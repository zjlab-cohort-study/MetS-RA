#!/usr/bin/env python3
"""Summarise outer-fold performance and obtain paired conditional intervals.

The optional bootstrap samples participant identifiers once per replicate and
uses those same multiplicities across all repeats and models.  This preserves
the pairing and the dependence created by repeated out-of-fold predictions.
Model fitting is not repeated, so the intervals are explicitly conditional on
the completed development runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import brier_score, concordance_index_ipcw, cumulative_dynamic_auc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline_common import (  # noqa: E402
    administratively_censor, make_survival, read_config, read_table, target_cohort
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--comparison", default="integrated,clinical_inflammatory")
    return parser.parse_args()


def performance_summary(output: Path) -> pd.DataFrame:
    metrics = pd.read_csv(output / "all_outer_fold_metrics.csv")
    metric_columns = [
        x for x in metrics if x.startswith(("harrell_", "uno_", "auc_", "brier_"))
    ]
    repeat = metrics.groupby(["repeat_id", "model"])[metric_columns].mean().reset_index()
    repeat.to_csv(output / "repeat_specific_performance.csv", index=False)
    summary = repeat.groupby("model")[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(x).rstrip("_") for x in summary.columns.to_flat_index()]
    summary.to_csv(output / "model_performance_summary.csv", index=False)
    return repeat


def fold_metrics(
    y_train: np.ndarray, event: np.ndarray, time: np.ndarray,
    risk: np.ndarray, survival: np.ndarray, horizons: np.ndarray, tau: float,
) -> dict[str, float]:
    y_test = administratively_censor(
        np.array(list(zip(event.astype(bool), time.astype(float))),
                 dtype=[("event", "?"), ("time", "<f8")]),
        tau,
    )
    auc, _ = cumulative_dynamic_auc(y_train, y_test, risk, horizons)
    _, brier = brier_score(y_train, y_test, survival, horizons)
    result = {
        "uno_c_10y": float(concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0])
    }
    for horizon, value in zip(horizons, auc):
        result[f"auc_{horizon:g}y"] = float(value)
    for horizon, value in zip(horizons, brier):
        result[f"brier_{horizon:g}y"] = float(value)
    return result


def participant_cluster_bootstrap(
    config: dict, output: Path, n_bootstrap: int, new_model: str, reference_model: str
) -> pd.DataFrame:
    prediction = pd.read_csv(output / "all_outer_predictions.csv")
    cohort = target_cohort(read_table(config), config)
    splits = pd.read_csv(output / "outer_fold_assignments.csv")
    identifier = config["id_column"]
    horizons = np.asarray(config.get("horizons_years", [3, 5, 10]), dtype=float)
    tau = float(config.get("uno_tau_years", 10))
    survival_columns = [f"survival_{x:g}y" for x in horizons]
    base_ids = cohort[[identifier, config["event_column"]]].copy()
    event_ids = base_ids.loc[base_ids[config["event_column"]] == 1, identifier].to_numpy()
    nonevent_ids = base_ids.loc[base_ids[config["event_column"]] == 0, identifier].to_numpy()
    rng = np.random.default_rng(int(config.get("random_seed", 20260812)) + 4000)
    rows: list[dict] = []
    for bootstrap_id in range(n_bootstrap):
        sampled = np.concatenate([
            rng.choice(event_ids, len(event_ids), replace=True),
            rng.choice(nonevent_ids, len(nonevent_ids), replace=True),
        ])
        multiplicity = pd.Series(sampled).value_counts()
        repeat_differences: list[dict[str, float]] = []
        for repeat_id in sorted(prediction.repeat_id.unique()):
            repeat_prediction = prediction.loc[prediction.repeat_id == repeat_id]
            assignment = splits.loc[splits.repeat_id == repeat_id, [identifier, "outer_fold"]]
            cohort_repeat = cohort.merge(assignment, on=identifier, validate="one_to_one")
            fold_differences: list[dict[str, float]] = []
            for fold_id in sorted(repeat_prediction.outer_fold.unique()):
                fold_prediction = repeat_prediction.loc[repeat_prediction.outer_fold == fold_id]
                base = fold_prediction.loc[
                    fold_prediction.model == new_model,
                    [identifier, "event", "time"],
                ].sort_values(identifier)
                weights = base[identifier].map(multiplicity).fillna(0).astype(int).to_numpy()
                take = np.repeat(np.arange(len(base)), weights)
                if len(take) == 0 or base.event.to_numpy()[take].sum() == 0:
                    continue
                train = cohort_repeat.loc[cohort_repeat.outer_fold != fold_id]
                y_train = administratively_censor(make_survival(train, config), tau)
                model_values: dict[str, dict[str, float]] = {}
                for model_name in [new_model, reference_model]:
                    frame = fold_prediction.loc[
                        fold_prediction.model == model_name
                    ].set_index(identifier).loc[base[identifier]]
                    model_values[model_name] = fold_metrics(
                        y_train,
                        base.event.to_numpy()[take],
                        base.time.to_numpy()[take],
                        frame.risk_score.to_numpy()[take],
                        frame[survival_columns].to_numpy()[take],
                        horizons,
                        tau,
                    )
                difference = {}
                for metric, new_value in model_values[new_model].items():
                    raw = new_value - model_values[reference_model][metric]
                    difference[metric] = -raw if metric.startswith("brier_") else raw
                fold_differences.append(difference)
            if fold_differences:
                repeat_differences.append(pd.DataFrame(fold_differences).mean().to_dict())
        if repeat_differences:
            mean_difference = pd.DataFrame(repeat_differences).mean()
            rows.extend({
                "bootstrap_id": bootstrap_id,
                "new_model": new_model,
                "reference_model": reference_model,
                "metric": metric,
                "improvement_positive_is_better": value,
            } for metric, value in mean_difference.items())
    replicates = pd.DataFrame(rows)
    replicates.to_csv(output / "participant_cluster_bootstrap_replicates.csv.gz", index=False)
    intervals = replicates.groupby(["new_model", "reference_model", "metric"])[
        "improvement_positive_is_better"
    ].agg(
        estimate="mean",
        ci_low=lambda x: x.quantile(0.025),
        ci_high=lambda x: x.quantile(0.975),
    ).reset_index()
    intervals["interval_scope"] = "conditional on fixed out-of-fold predictions"
    intervals.to_csv(output / "paired_participant_cluster_bootstrap_ci.csv", index=False)
    return intervals


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    output = Path(config["output_dir"])
    performance_summary(output)
    if args.bootstrap > 0:
        comparison = [x.strip() for x in args.comparison.split(",")]
        if len(comparison) != 2:
            raise ValueError("--comparison must be new_model,reference_model")
        intervals = participant_cluster_bootstrap(
            config, output, args.bootstrap, comparison[0], comparison[1]
        )
        print(intervals.to_string(index=False))


if __name__ == "__main__":
    main()
