#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1: Cox proportional-hazards models for MetS, its components, and incident RA.

Reads a baseline CSV (one row per participant), derives the number of abnormal MetS
components (``components_total``) and the MetS indicator (>= 3 abnormal components),
and fits three adjustment models per exposure plus a dose-response model by
component count. See README.md for the input format.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


EVENT_COL = "RA_label"
TIME_COL = "RA_followup_time"
METS_EXPOSURE = "MetS"

COMPONENT_COLS = [
    "waist_circumference",
    "triglycerides",
    "blood_pressure",
    "blood_glucose",
    "HDL_cholesterol",
]

EXPOSURE_COLS = [
    "waist_circumference",
    "triglycerides",
    "blood_pressure",
    "blood_glucose",
    "HDL_cholesterol",
    "components_total",
    "MetS"
]

BASIC_COVS = ["Sex", "Age", "Race"]
FULL_COVS = BASIC_COVS + [
    "Educational_Level",
    "Smoking",
    "Alcohol_Drinking",
    "Townsend_deprivation_index",
    "Household_Income", "BMI"
]

# Covariates modelled as unordered categorical via dummy variables
# (lowest observed level = reference).
DUMMY_CATEGORICAL_COVS = ["Smoking", "Alcohol_Drinking", "Household_Income"]


def fit_single_exposure_cox(
    df: pd.DataFrame,
    exposure: str,
    covariates: list[str],
    model_name: str,
    penalizer: float = 0.0,
) -> tuple[dict, list[dict]]:
    use_cols = [TIME_COL, EVENT_COL, exposure] + covariates
    d = df[use_cols].dropna().copy()

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(d, duration_col=TIME_COL, event_col=EVENT_COL, show_progress=False)

    row = cph.summary.loc[exposure]
    exposure_row = {
        "exposure": exposure,
        "model": model_name,
        "covariates": ",".join(covariates) if covariates else "(none)",
        "N": int(d.shape[0]),
        "events": int(d[EVENT_COL].sum()),
        "coef": float(row["coef"]),
        "se": float(row["se(coef)"]),
        "HR": float(row["exp(coef)"]),
        "CI95_lower": float(row["exp(coef) lower 95%"]),
        "CI95_upper": float(row["exp(coef) upper 95%"]),
        "p": float(row["p"]),
        "n_exposed": int(d[exposure].sum()),
        "prevalence": float(d[exposure].mean()),
    }

    
    covariate_rows: list[dict] = []
    for term in covariates:
        if term not in cph.summary.index:
            continue
        crow = cph.summary.loc[term]
        covariate_rows.append(
            {
                "exposure": exposure,
                "model": model_name,
                "term": term,
                "N": int(d.shape[0]),
                "events": int(d[EVENT_COL].sum()),
                "coef": float(crow["coef"]),
                "se": float(crow["se(coef)"]),
                "HR": float(crow["exp(coef)"]),
                "CI95_lower": float(crow["exp(coef) lower 95%"]),
                "CI95_upper": float(crow["exp(coef) upper 95%"]),
                "p": float(crow["p"]),
            }
        )

    return exposure_row, covariate_rows


def fit_components_count_cox(
    df: pd.DataFrame,
    covariates: list[str],
    model_name: str,
    penalizer: float = 0.0,
) -> pd.DataFrame:
    d = df[[TIME_COL, EVENT_COL, "components_total"] + covariates].dropna().copy()

    dummy = pd.get_dummies(d["components_total"].astype(int), prefix="components_total")
    dummy = dummy.drop(columns=["components_total_0"], errors="ignore")

    model_df = pd.concat([d[[TIME_COL, EVENT_COL] + covariates], dummy], axis=1)
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(model_df, duration_col=TIME_COL, event_col=EVENT_COL, show_progress=False)

    counts = (
        df.groupby("components_total", dropna=False)[EVENT_COL]
        .agg(total_n="count", events="sum")
        .reset_index()
        .rename(columns={"components_total": "components"})
    )

    rows = []
    rows.append(
        {
            "components": 0,
            "model": model_name,
            "N_used": int(d.shape[0]),
            "events_used": int(d[EVENT_COL].sum()),
            "total_n": int(counts.loc[counts["components"] == 0, "total_n"].iloc[0]),
            "events": int(counts.loc[counts["components"] == 0, "events"].iloc[0]),
            "HR": 1.0,
            "CI95_lower": np.nan,
            "CI95_upper": np.nan,
            "p": np.nan,
            "label": "Ref",
        }
    )

    for level in range(1, 6):
        term = f"components_total_{level}"
        row = cph.summary.loc[term]
        rows.append(
            {
                "components": level,
                "model": model_name,
                "N_used": int(d.shape[0]),
                "events_used": int(d[EVENT_COL].sum()),
                "total_n": int(counts.loc[counts["components"] == level, "total_n"].iloc[0]),
                "events": int(counts.loc[counts["components"] == level, "events"].iloc[0]),
                "HR": float(row["exp(coef)"]),
                "CI95_lower": float(row["exp(coef) lower 95%"]),
                "CI95_upper": float(row["exp(coef) upper 95%"]),
                "p": float(row["p"]),
                "label": None,
            }
        )

    out = pd.DataFrame(rows)
    out["hr_ci_text"] = out.apply(
        lambda r: "Ref"
        if r["components"] == 0
        else f"{r['HR']:.2f}({r['CI95_lower']:.2f}-{r['CI95_upper']:.2f})",
        axis=1,
    )
    out["p_text"] = out["p"].apply(format_p)
    out.loc[out["components"] == 0, "p_text"] = ""
    return out


def format_p(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 1e-3:
        base, exp = f"{value:.2e}".split("e")
        return f"{base}x10^{int(exp)}"
    return f"{value:.2f}"


def add_derived_exposures(df: pd.DataFrame) -> pd.DataFrame:
    
    out = df.copy()
    n_abnormal = out[COMPONENT_COLS].sum(axis=1, min_count=len(COMPONENT_COLS))
    out["components_total"] = n_abnormal
    out[METS_EXPOSURE] = np.where(n_abnormal >= 3, 1.0, np.where(n_abnormal.notna(), 0.0, np.nan))
    return out


def expand_to_dummies(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    
    dummy_cols: list[str] = []
    for col in cols:
        present = df[col].notna()
        levels = sorted(int(v) for v in df[col].dropna().unique())
        for v in levels[1:]:  # lowest level (levels[0]) is the reference, gets no dummy
            name = f"{col}_{v}"
            df[name] = (df[col] == v).astype(float).where(present)
            dummy_cols.append(name)
    return df, dummy_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Step1 Cox models.")
    parser.add_argument(
        "--input",
        default="input_file.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--out_dir",
        default="output",
        help="Output directory.",
    )
    parser.add_argument("--penalizer", type=float, default=0.0, help="L2 penalizer for CoxPHFitter.")
    parser.add_argument(
        "--no-bmi",
        action="store_true",
        help="Exclude BMI from the Model-3 adjustment covariates (run the no-BMI variant).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    required_cols = [EVENT_COL, TIME_COL] + COMPONENT_COLS + FULL_COVS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [EVENT_COL, TIME_COL] + COMPONENT_COLS + FULL_COVS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_derived_exposures(df)

    # (dummies, lowest observed level = reference).
    df, cat_dummy_cols = expand_to_dummies(df, DUMMY_CATEGORICAL_COVS)
    model_full_covs = [c for c in FULL_COVS if not (args.no_bmi and c == "BMI")]
    full_covs_expanded = [c for c in model_full_covs if c not in DUMMY_CATEGORICAL_COVS] + cat_dummy_cols
    model_specs = [
        ("Model-1", []),
        ("Model-2", BASIC_COVS),
        ("Model-3", full_covs_expanded),
    ]

    exposure_rows = []
    covariate_rows = []
    for exposure in EXPOSURE_COLS:
        for model_name, covariates in model_specs:
            row, cov_rows = fit_single_exposure_cox(
                df, exposure, covariates, model_name, penalizer=args.penalizer
            )
            exposure_rows.append(row)
            covariate_rows.extend(cov_rows)

    exposure_results = pd.DataFrame(exposure_rows)
    exposure_results["p_text"] = exposure_results["p"].apply(format_p)
    exposure_results.to_csv(out_dir / "step1_single_exposure_cox_results.csv", index=False)

    covariate_results = pd.DataFrame(covariate_rows)
    if not covariate_results.empty:
        covariate_results["p_text"] = covariate_results["p"].apply(format_p)
    covariate_results.to_csv(out_dir / "step1_covariate_cox_results.csv", index=False)

    dose_rows = []
    for model_name, covariates in model_specs:
        dose_rows.append(fit_components_count_cox(df, covariates, model_name, penalizer=args.penalizer))
    dose_results_all = pd.concat(dose_rows, ignore_index=True)
    dose_results_all.to_csv(out_dir / "step1_components_count_cox_results.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "input_rows",
                "input_events",
                "mets_prevalence_pct",
                "mets_model1_hr",
                "mets_model2_hr",
                "mets_model3_hr",
                "bmi_in_model3",
                "model3_covariates",
            ],
            "value": [
                int(df.shape[0]),
                int(df[EVENT_COL].sum()),
                round(float(df[METS_EXPOSURE].mean() * 100), 2),
                round(float(exposure_results.query("exposure == @METS_EXPOSURE and model == 'Model-1'")["HR"].iloc[0]), 4),
                round(float(exposure_results.query("exposure == @METS_EXPOSURE and model == 'Model-2'")["HR"].iloc[0]), 4),
                round(float(exposure_results.query("exposure == @METS_EXPOSURE and model == 'Model-3'")["HR"].iloc[0]), 4),
                "no" if args.no_bmi else "yes",
                ",".join(full_covs_expanded),
            ],
        }
    )
    summary.to_csv(out_dir / "step1_run_summary.csv", index=False)

    print(f"[DONE] Outputs saved to: {out_dir}")
    print(exposure_results[["exposure", "model", "HR", "CI95_lower", "CI95_upper", "p"]].to_string(index=False))
    if not covariate_results.empty:
        print("\nCovariate associations (Model-3, MetS exposure):")
        view = covariate_results[
            (covariate_results["model"] == "Model-3") & (covariate_results["exposure"] == METS_EXPOSURE)
        ][["term", "HR", "CI95_lower", "CI95_upper", "p"]]
        print(view.to_string(index=False))


if __name__ == "__main__":
    main()
