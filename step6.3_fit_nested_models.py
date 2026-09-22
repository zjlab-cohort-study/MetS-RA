#!/usr/bin/env python3
"""Fit four leakage-resistant survival models by repeated nested CV.

All variable names and paths are supplied by a local JSON configuration.  The
script never reads a full-cohort discovery list when fitting an outer fold.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (  # noqa: E402
    TrainingTransformer,
    add_configured_transforms,
    administratively_censor,
    encode_categorical,
    make_survival,
    read_config,
    read_table,
    target_cohort,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prune_correlated(
    train: pd.DataFrame, screen: pd.DataFrame, limit: float
) -> list[str]:
    """Greedy rank-correlation pruning ordered by training-only screen P value."""
    selected_col = "fdr_selected" if "fdr_selected" in screen else "fdr_005"
    eligible = screen.loc[
        (screen["status"] == "ok")
        & screen[selected_col].fillna(False).astype(bool)
    ].sort_values(["p", "protein"])
    candidates = [str(p) for p in eligible["protein"] if p in train.columns]
    if not candidates:
        return []
    values = train[candidates].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(values.median())
    correlation = values.corr(method="spearman").abs().fillna(0.0)
    retained: list[str] = []
    for protein in candidates:
        if not retained or float(correlation.loc[protein, retained].max()) <= limit:
            retained.append(protein)
    return retained


def transformer(config: dict, features: list[str]) -> TrainingTransformer:
    return TrainingTransformer(
        features,
        set(config.get("inflammatory_predictors", [])),
        float(config.get("winsor_lower", 0.005)),
        float(config.get("winsor_upper", 0.995)),
    )


def derive_alpha_grid(
    frame: pd.DataFrame, y: np.ndarray, features: list[str],
    penalty_factor: np.ndarray, config: dict,
) -> np.ndarray:
    prep = transformer(config, features).fit(frame)
    model = CoxnetSurvivalAnalysis(
        n_alphas=int(config.get("n_alphas", 40)),
        alpha_min_ratio=float(config.get("alpha_min_ratio", 0.01)),
        l1_ratio=float(config.get("elastic_net_l1_ratio", 0.5)),
        penalty_factor=penalty_factor,
        normalize=False,
        max_iter=100000,
        tol=1e-7,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(prep.transform(frame), y)
    return np.asarray(model.alphas_, dtype=float)


def choose_alpha(
    frame: pd.DataFrame, y: np.ndarray, features: list[str],
    penalty_factor: np.ndarray, config: dict, seed: int,
) -> tuple[float, pd.DataFrame]:
    """Choose the strongest penalty within one SE of the best inner-fold C."""
    alphas = derive_alpha_grid(frame, y, features, penalty_factor, config)
    n_inner = int(config.get("inner_folds", 5))
    split = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
    scores = np.full((n_inner, len(alphas)), np.nan)
    event = y["event"].astype(int)
    for fold, (fit_index, validation_index) in enumerate(
        split.split(np.zeros(len(frame)), event)
    ):
        fit_frame = frame.iloc[fit_index]
        validation_frame = frame.iloc[validation_index]
        prep = transformer(config, features).fit(fit_frame)
        model = CoxnetSurvivalAnalysis(
            alphas=alphas,
            l1_ratio=float(config.get("elastic_net_l1_ratio", 0.5)),
            penalty_factor=penalty_factor,
            normalize=False,
            max_iter=100000,
            tol=1e-7,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(prep.transform(fit_frame), y[fit_index])
        fitted_alphas = np.asarray(model.alphas_)
        for index, alpha in enumerate(alphas):
            if alpha < fitted_alphas.min() - 1e-12 or alpha > fitted_alphas.max() + 1e-12:
                continue
            risk = model.predict(prep.transform(validation_frame), alpha=float(alpha))
            scores[fold, index] = concordance_index_censored(
                y[validation_index]["event"], y[validation_index]["time"], risk
            )[0]
    valid_n = np.isfinite(scores).sum(axis=0)
    mean = np.nanmean(scores, axis=0)
    standard_error = np.nanstd(scores, axis=0, ddof=1) / np.sqrt(valid_n)
    valid = (valid_n >= 3) & np.isfinite(mean)
    if not valid.any():
        raise RuntimeError("No alpha had at least three valid inner-fold scores")
    valid_indices = np.where(valid)[0]
    best = valid_indices[np.argmax(mean[valid])]
    eligible = np.where(valid & (mean >= mean[best] - standard_error[best]))[0]
    chosen = eligible[np.argmax(alphas[eligible])]
    table = pd.DataFrame({
        "alpha": alphas,
        "mean_inner_harrell_c": mean,
        "se_inner_harrell_c": standard_error,
        "valid_inner_folds": valid_n,
        "best_mean": np.arange(len(alphas)) == best,
        "chosen_one_se": np.arange(len(alphas)) == chosen,
    })
    return float(alphas[chosen]), table


def evaluate(
    y_train: np.ndarray, y_test: np.ndarray, risk: np.ndarray,
    survival_probability: np.ndarray, config: dict,
) -> dict[str, float]:
    horizons = np.asarray(config.get("horizons_years", [3, 5, 10]), dtype=float)
    tau = float(config.get("uno_tau_years", 10))
    train_tau = administratively_censor(y_train, tau)
    test_tau = administratively_censor(y_test, tau)
    result = {
        "harrell_c": float(concordance_index_censored(
            y_test["event"], y_test["time"], risk
        )[0]),
        "uno_c_10y": float(concordance_index_ipcw(
            train_tau, test_tau, risk, tau=tau
        )[0]),
    }
    auc, _ = cumulative_dynamic_auc(train_tau, test_tau, risk, horizons)
    _, brier = brier_score(train_tau, test_tau, survival_probability, horizons)
    for horizon, value in zip(horizons, auc):
        result[f"auc_{horizon:g}y"] = float(value)
    for horizon, value in zip(horizons, brier):
        result[f"brier_{horizon:g}y"] = float(value)
    return result


def fit_specification(
    model_name: str, train: pd.DataFrame, test: pd.DataFrame,
    features: list[str], unpenalized_count: int,
    repeat_id: int, fold_id: int, config: dict,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    y_train, y_test = make_survival(train, config), make_survival(test, config)
    penalty_factor = np.ones(len(features), dtype=float)
    penalty_factor[:unpenalized_count] = 0.0
    prep = transformer(config, features).fit(train)
    x_train, x_test = prep.transform(train), prep.transform(test)
    horizons = np.asarray(config.get("horizons_years", [3, 5, 10]), dtype=float)
    if unpenalized_count == len(features):
        alpha = 0.0
        model = CoxPHSurvivalAnalysis(alpha=1e-8, ties="breslow", n_iter=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
        risk = model.predict(x_test)
        functions = model.predict_survival_function(x_test)
        coefficients = np.asarray(model.coef_)
        tuning = pd.DataFrame({
            "alpha": [0.0], "fixed_prespecified_cox": [True],
            "chosen_one_se": [False],
        })
    else:
        seed = int(config.get("random_seed", 20260812)) + repeat_id * 100 + fold_id
        alpha, tuning = choose_alpha(
            train, y_train, features, penalty_factor, config, seed
        )
        model = CoxnetSurvivalAnalysis(
            alphas=[alpha],
            l1_ratio=float(config.get("elastic_net_l1_ratio", 0.5)),
            penalty_factor=penalty_factor,
            normalize=False,
            max_iter=100000,
            tol=1e-7,
            fit_baseline_model=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(x_train, y_train)
        risk = model.predict(x_test, alpha=alpha)
        functions = model.predict_survival_function(x_test, alpha=alpha)
        coefficients = np.asarray(model.coef_)[:, 0]
    survival = np.asarray([[fn(time) for time in horizons] for fn in functions])
    metrics = evaluate(y_train, y_test, risk, survival, config)
    protein_set = set(config["_protein_columns"])
    selected = pd.DataFrame({
        "model": model_name,
        "repeat_id": repeat_id,
        "outer_fold": fold_id,
        "feature": features,
        "feature_type": ["protein" if x in protein_set else "fixed_predictor" for x in features],
        "coefficient": coefficients,
        "selected_nonzero": np.abs(coefficients) > 1e-10,
        "unpenalized": penalty_factor == 0,
    })
    metrics.update({
        "model": model_name,
        "repeat_id": repeat_id,
        "outer_fold": fold_id,
        "test_n": len(test),
        "test_events": int(test[config["event_column"]].sum()),
        "candidate_features": len(features),
        "selected_proteins": int(selected.loc[
            selected.feature_type == "protein", "selected_nonzero"
        ].sum()),
        "alpha": alpha,
        "l1_ratio": float(config.get("elastic_net_l1_ratio", 0.5)),
    })
    prediction = pd.DataFrame({
        config["id_column"]: test[config["id_column"]].to_numpy(),
        "event": y_test["event"],
        "time": y_test["time"],
        "risk_score": risk,
        "model": model_name,
        "repeat_id": repeat_id,
        "outer_fold": fold_id,
    })
    for index, horizon in enumerate(horizons):
        prediction[f"survival_{horizon:g}y"] = survival[:, index]
    tuning.insert(0, "model", model_name)
    tuning.insert(1, "repeat_id", repeat_id)
    tuning.insert(2, "outer_fold", fold_id)
    return prediction, metrics, tuning, selected


def aggregate(output: Path) -> None:
    fold_dir = output / "fold_results"
    groups = {
        "predictions_": "all_outer_predictions.csv",
        "metrics_": "all_outer_fold_metrics.csv",
        "tuning_": "all_tuning_results.csv",
        "selected_": "all_selected_features.csv",
    }
    for prefix, destination in groups.items():
        paths = sorted(fold_dir.glob(f"{prefix}repeat*_fold*.csv"))
        if paths:
            pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).to_csv(
                output / destination, index=False
            )
    metrics_path = output / "all_outer_fold_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        numeric = [x for x in metrics if x.startswith(("harrell_", "uno_", "auc_", "brier_"))]
        metrics.groupby(["repeat_id", "model"])[numeric].mean().reset_index().to_csv(
            output / "repeat_specific_performance.csv", index=False
        )
    selected_path = output / "all_selected_features.csv"
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        protein = selected.loc[selected.feature_type == "protein"]
        stability = protein.groupby(["model", "feature"]).agg(
            selected_folds=("selected_nonzero", "sum"),
            candidate_folds=("selected_nonzero", "size"),
        ).reset_index()
        totals = selected[["model", "repeat_id", "outer_fold"]].drop_duplicates().groupby("model").size()
        stability["total_completed_folds"] = stability.model.map(totals)
        stability["selection_frequency"] = stability.selected_folds / stability.total_completed_folds
        stability.to_csv(output / "protein_selection_stability.csv", index=False)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    output = Path(config["output_dir"])
    fold_output = output / "fold_results"
    fold_output.mkdir(parents=True, exist_ok=True)
    data = target_cohort(read_table(config), config)
    data = add_configured_transforms(data, config)
    data, dummy_columns = encode_categorical(data, config)
    protein_columns = [x.strip() for x in Path(config["protein_columns_file"]).read_text().splitlines() if x.strip()]
    config["_protein_columns"] = protein_columns
    clinical_metabolic = (
        list(config.get("clinical_numeric_predictors", []))
        + dummy_columns
        + list(config.get("metabolic_predictors", []))
    )
    clinical_inflammatory = clinical_metabolic + list(config.get("inflammatory_predictors", []))
    splits = pd.read_csv(output / "outer_fold_assignments.csv")
    identifier = config["id_column"]
    manifest = {
        "cohort_n": len(data),
        "events": int(data[config["event_column"]].sum()),
        "full_cohort_discovery_list_used_in_outer_folds": False,
        "models": ["clinical_metabolic", "clinical_inflammatory", "protein_only", "integrated"],
        "outer_validation": f"{config['outer_repeats']} repeats x {config['outer_folds']} folds",
        "inner_validation": f"{config['inner_folds']}-fold; one-standard-error penalty rule",
        "censoring_distribution_for_outer_metrics": "corresponding outer-training fold",
    }
    (output / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2))
    for repeat_id in range(1, int(config["outer_repeats"]) + 1):
        assignment = splits.loc[splits.repeat_id == repeat_id, [identifier, "outer_fold"]]
        fold_map = assignment.set_index(identifier).outer_fold
        fold_for_row = data[identifier].map(fold_map)
        if fold_for_row.isna().any():
            raise ValueError(f"Incomplete outer-fold assignment for repeat {repeat_id}")
        for fold_id in range(1, int(config["outer_folds"]) + 1):
            marker = fold_output / f"predictions_repeat{repeat_id:02d}_fold{fold_id:02d}.csv"
            if marker.exists() and not args.overwrite:
                continue
            train = data.loc[fold_for_row != fold_id].copy()
            test = data.loc[fold_for_row == fold_id].copy()
            screen = pd.read_csv(output / "screens" / f"screen_repeat{repeat_id:02d}_fold{fold_id:02d}.csv")
            proteins = prune_correlated(train, screen, float(config.get("correlation_limit", 0.90)))
            if not proteins:
                raise RuntimeError(f"No proteins retained for repeat {repeat_id}, fold {fold_id}")
            specifications = [
                ("clinical_metabolic", clinical_metabolic, len(clinical_metabolic)),
                ("clinical_inflammatory", clinical_inflammatory, len(clinical_inflammatory)),
                ("protein_only", proteins, 0),
                ("integrated", clinical_inflammatory + proteins, len(clinical_inflammatory)),
            ]
            predictions, metrics, tuning, selected = [], [], [], []
            for name, features, unpenalized_count in specifications:
                p, m, t, s = fit_specification(
                    name, train, test, features, unpenalized_count,
                    repeat_id, fold_id, config,
                )
                predictions.append(p); metrics.append(m); tuning.append(t); selected.append(s)
            pd.concat(predictions, ignore_index=True).to_csv(marker, index=False)
            pd.DataFrame(metrics).to_csv(
                fold_output / f"metrics_repeat{repeat_id:02d}_fold{fold_id:02d}.csv", index=False
            )
            pd.concat(tuning, ignore_index=True).to_csv(
                fold_output / f"tuning_repeat{repeat_id:02d}_fold{fold_id:02d}.csv", index=False
            )
            pd.concat(selected, ignore_index=True).to_csv(
                fold_output / f"selected_repeat{repeat_id:02d}_fold{fold_id:02d}.csv", index=False
            )
            aggregate(output)
    aggregate(output)


if __name__ == "__main__":
    main()
