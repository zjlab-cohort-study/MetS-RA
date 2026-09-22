#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2: sensitivity analyses — landmark (lag) and time-window Cox models.

Reads the same baseline CSV as Step 1 and re-fits the same Cox models after
(a) excluding the first 1/2/3 years of follow-up (landmark analyses, addressing
reverse causation) and (b) splitting follow-up into 0-5y, 5-10y, and >10y windows.
See README.md for the input format.
"""

import argparse
from pathlib import Path

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

ALL_EXPOSURES = COMPONENT_COLS + [METS_EXPOSURE]

BASIC_COVS = ["Sex", "Age", "Race"]
FULL_COVS = BASIC_COVS + [
    "Educational_Level",
    "Smoking",
    "Alcohol_Drinking",
    "Townsend_deprivation_index",
    "Household_Income"
]

# Covariates modelled as unordered categorical via dummy variables
# (lowest observed level = reference).
DUMMY_CATEGORICAL_COVS = ["Smoking", "Alcohol_Drinking", "Household_Income"]


def add_derived_exposures(df: pd.DataFrame) -> pd.DataFrame:
    """Derive components_total (0-5) and MetS (>= 3 abnormal components).

    ``min_count`` makes the sum NaN whenever any component is missing, so participants
    with incomplete component data are excluded from the derived exposures instead of
    being assigned a count based on the non-missing components only.
    """
    out = df.copy()
    n_abnormal = out[COMPONENT_COLS].sum(axis=1, min_count=len(COMPONENT_COLS))
    out["components_total"] = n_abnormal
    out[METS_EXPOSURE] = (n_abnormal >= 3).astype(float).where(n_abnormal.notna())
    return out


def expand_to_dummies(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Expand integer categorical covariates into 0/1 dummy columns.

    The LOWEST observed level of each variable is used as the reference and gets no
    dummy, so Smoking/Alcohol_Drinking (0,1,2) -> ref 0 and Household_Income (1-5)
    -> ref 1. Rows where the source value is missing become NaN in ALL of that
    variable's dummies, so that the downstream ``dropna()`` drops them instead of
    silently coding them as the reference level.
    """
    dummy_cols: list[str] = []
    for col in cols:
        present = df[col].notna()
        levels = sorted(int(v) for v in df[col].dropna().unique())
        for v in levels[1:]:  # lowest level (levels[0]) is the reference, gets no dummy
            name = f"{col}_{v}"
            df[name] = (df[col] == v).astype(float).where(present)
            dummy_cols.append(name)
    return df, dummy_cols


def fit_single_exposure_cox(df: pd.DataFrame, time_col: str, event_col: str, exposure: str, covariates: list[str], model_name: str, penalizer: float = 0.0) -> dict:
    use_cols = [time_col, event_col, exposure] + covariates
    d = df[use_cols].dropna().copy()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(d, duration_col=time_col, event_col=event_col, show_progress=False)
    row = cph.summary.loc[exposure]
    return {
        "exposure": exposure,
        "model": model_name,
        "N": int(d.shape[0]),
        "events": int(d[event_col].sum()),
        "HR": float(row["exp(coef)"]),
        "CI95_lower": float(row["exp(coef) lower 95%"]),
        "CI95_upper": float(row["exp(coef) upper 95%"]),
        "p": float(row["p"]),
    }


def apply_landmark_exclusion(df: pd.DataFrame, landmark_years: float) -> tuple[pd.DataFrame, dict]:
    keep = df[TIME_COL] > landmark_years
    out = df.loc[keep].copy()
    dropped_total = int((~keep).sum())
    dropped_events = int(df.loc[~keep, EVENT_COL].sum())
    dropped_non_events = int(dropped_total - dropped_events)
    return out, {
        "landmark_years": landmark_years,
        "dropped_total": dropped_total,
        "dropped_events": dropped_events,
        "dropped_non_events": dropped_non_events,
    }


def build_window_dataset(df: pd.DataFrame, window_key: str) -> pd.DataFrame:
    d = df.copy()
    if window_key == "0_5":
        d["time_w"] = d[TIME_COL].clip(upper=5)
        d["event_w"] = ((d[EVENT_COL] == 1) & (d[TIME_COL] <= 5)).astype(int)
        return d[d["time_w"] > 0].copy()
    if window_key == "5_10":
        d = d[d[TIME_COL] > 5].copy()
        d["time_w"] = d[TIME_COL].clip(upper=10) - 5
        d["event_w"] = ((d[EVENT_COL] == 1) & (d[TIME_COL] > 5) & (d[TIME_COL] <= 10)).astype(int)
        return d[d["time_w"] > 0].copy()
    if window_key == "gt_10":
        d = d[d[TIME_COL] > 10].copy()
        d["time_w"] = d[TIME_COL] - 10
        d["event_w"] = ((d[EVENT_COL] == 1) & (d[TIME_COL] > 10)).astype(int)
        return d[d["time_w"] > 0].copy()
    raise ValueError(f"Unknown window: {window_key}")


def format_p(val: float) -> str:
    if pd.isna(val):
        return ""
    if val < 1e-3:
        base, exp = f"{val:.2e}".split("e")
        return f"{base}x10^{int(exp)}"
    return f"{val:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: landmark and time-window sensitivity Cox analyses.")
    parser.add_argument("--input", default="input_file.csv", help="Input CSV path.")
    parser.add_argument("--out_dir", default="output", help="Output directory.")
    parser.add_argument("--penalizer", type=float, default=0.0, help="L2 penalizer for CoxPHFitter.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lag_mid_dir = out_dir / "processed_lag"
    win_mid_dir = out_dir / "processed_timewindow"
    lag_mid_dir.mkdir(parents=True, exist_ok=True)
    win_mid_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required_cols = [EVENT_COL, TIME_COL] + COMPONENT_COLS + FULL_COVS
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in [EVENT_COL, TIME_COL] + COMPONENT_COLS + FULL_COVS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = add_derived_exposures(df)

    # Model Smoking/Alcohol_Drinking/Household_Income as unordered categorical
    # (dummies, lowest observed level = reference).
    df, cat_dummy_cols = expand_to_dummies(df, DUMMY_CATEGORICAL_COVS)
    full_covs_expanded = [c for c in FULL_COVS if c not in DUMMY_CATEGORICAL_COVS] + cat_dummy_cols
    model_specs = [
        ("Model1_unadjusted", []),
        ("Model2_basic", BASIC_COVS),
        ("Model3_full", full_covs_expanded),
    ]

    lag_results = []
    lag_summary = []
    for lag in [1.0, 2.0, 3.0]:
        df_lag, info = apply_landmark_exclusion(df, lag)
        lag_tag = str(int(lag))
        df_lag.to_csv(lag_mid_dir / f"cohort_landmark_{lag_tag}y.csv", index=False)
        lag_summary.append({
            "landmark_years": lag,
            "N_after_exclusion": int(df_lag.shape[0]),
            "events_after_exclusion": int(df_lag[EVENT_COL].sum()),
            **info,
        })
        for exposure in ALL_EXPOSURES:
            for model_name, covs in model_specs:
                row = fit_single_exposure_cox(df_lag, TIME_COL, EVENT_COL, exposure, covs, model_name, args.penalizer)
                row.update(info)
                lag_results.append(row)

    lag_df = pd.DataFrame(lag_results)
    lag_df["p_text"] = lag_df["p"].apply(format_p)
    lag_df.to_csv(out_dir / "step2_lag_cox_results_v2.csv", index=False)
    pd.DataFrame(lag_summary).to_csv(out_dir / "step2_lag_exclusion_summary_v2.csv", index=False)

    window_results = []
    for window_key, label in [("0_5", "0-5y"), ("5_10", "5-10y"), ("gt_10", ">10y")]:
        df_win = build_window_dataset(df, window_key)
        df_win.to_csv(win_mid_dir / f"cohort_window_{window_key}.csv", index=False)
        for exposure in ALL_EXPOSURES:
            for model_name, covs in model_specs:
                row = fit_single_exposure_cox(df_win, "time_w", "event_w", exposure, covs, model_name, args.penalizer)
                row["window"] = label
                window_results.append(row)

    win_df = pd.DataFrame(window_results)
    win_df["p_text"] = win_df["p"].apply(format_p)
    win_df.to_csv(out_dir / "step2_timewindow_cox_results_v2.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": ["input_rows", "input_events", "lag_rows_result", "timewindow_rows_result"],
            "value": [int(df.shape[0]), int(df[EVENT_COL].sum()), int(lag_df.shape[0]), int(win_df.shape[0])],
        }
    )
    summary.to_csv(out_dir / "step2_run_summary_v2.csv", index=False)

    print(f"[DONE] Outputs saved to: {out_dir}")
    view_cols = ["N", "events", "HR", "CI95_lower", "CI95_upper", "p"]
    print("\nMetS (Model3_full) by landmark:")
    view = lag_df[(lag_df["exposure"] == METS_EXPOSURE) & (lag_df["model"] == "Model3_full")]
    print(view[["landmark_years"] + view_cols].to_string(index=False))
    print("\nMetS (Model3_full) by time window:")
    view = win_df[(win_df["exposure"] == METS_EXPOSURE) & (win_df["model"] == "Model3_full")]
    print(view[["window"] + view_cols].to_string(index=False))


if __name__ == "__main__":
    main()
