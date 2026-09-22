#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5.1: stratified univariate Cox models, one per plasma protein.

Within each MetS stratum (non-MetS / MetS), a covariate-adjusted Cox model is
fitted for every protein column: proteins are z-scored within the stratum and
tested one at a time, with Bonferroni and Benjamini-Hochberg correction across
the tested proteins. BMI is present in the data but deliberately not adjusted
for (it lies on the causal pathway between MetS and RA).

Any column not listed in NON_PROTEIN_COLS is treated as a protein.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


TIME_COL = "RA_followup_time"
EVENT_COL = "RA_label"
STRATA_COL = "MetS"

COVARIATES = [
    "Sex",
    "Age",
    "Race",
    "Educational.Level",
    "Smoking",
    "Alcohol_Drinking",
    "Townsend_deprivation_index",
    "Household_Income",
]

# Columns that are not proteins; everything else is treated as a protein.
NON_PROTEIN_COLS = {
    "eid",
    EVENT_COL,
    TIME_COL,
    STRATA_COL,
    "waist_circumference",
    "triglycerides",
    "blood_pressure",
    "blood_glucose",
    "HDL_cholesterol",
    "both_1",
    "both_2",
    "both_3",
    "both_4",
    "both_5",
    "components_total",
    "BMI",
    *COVARIATES,
}


def one_hot_covariates(df: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    covs_expanded: list[str] = []
    for c in covariates:
        ser = out[c]
        is_obj = (ser.dtype == "object") or pd.api.types.is_string_dtype(ser)
        nunique = ser.dropna().nunique()
        is_small_cat = (nunique <= 10) and (not pd.api.types.is_float_dtype(ser))
        if is_obj or is_small_cat:
            dummies = pd.get_dummies(ser, prefix=c, dummy_na=False, drop_first=True)
            out = out.drop(columns=[c])
            out = pd.concat([out, dummies], axis=1)
            covs_expanded.extend(list(dummies.columns))
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            covs_expanded.append(c)
    return out, covs_expanded


def infer_protein_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_PROTEIN_COLS]


def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    mu = np.nanmean(x.values)
    sd = np.nanstd(x.values, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return x * np.nan
    return (x - mu) / sd


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.array(pvals, dtype=float)
    q = np.full(p.size, np.nan)
    ok = np.isfinite(p)
    p_ok = p[ok]
    m = p_ok.size
    if m == 0:
        return q
    order = np.argsort(p_ok)
    ranks = np.arange(1, m + 1)
    q_ok = p_ok[order] * m / ranks
    q_ok = np.minimum.accumulate(q_ok[::-1])[::-1]
    q_ok = np.clip(q_ok, 0, 1)
    tmp = np.empty_like(p_ok)
    tmp[order] = q_ok
    q[ok] = tmp
    return q


def fit_univariate_cox(df: pd.DataFrame, cov_cols: list[str], protein_col: str) -> dict:
    use_cols = [TIME_COL, EVENT_COL] + cov_cols + [protein_col]
    d = df[use_cols].dropna().copy()
    n = d.shape[0]
    events = int(d[EVENT_COL].sum()) if n > 0 else 0
    if n < 50 or events < 5:
        return {
            "protein": protein_col, "beta": np.nan, "se": np.nan, "HR": np.nan,
            "CI95_lower": np.nan, "CI95_upper": np.nan, "p": np.nan,
            "N_used": n, "events_used": events, "status": "skip_smallN",
        }

    cph = CoxPHFitter(penalizer=0.0)
    try:
        cph.fit(d, duration_col=TIME_COL, event_col=EVENT_COL, show_progress=False)
    except Exception:
        return {
            "protein": protein_col, "beta": np.nan, "se": np.nan, "HR": np.nan,
            "CI95_lower": np.nan, "CI95_upper": np.nan, "p": np.nan,
            "N_used": n, "events_used": events, "status": "fit_error",
        }

    summ = cph.summary
    if protein_col not in summ.index:
        return {
            "protein": protein_col, "beta": np.nan, "se": np.nan, "HR": np.nan,
            "CI95_lower": np.nan, "CI95_upper": np.nan, "p": np.nan,
            "N_used": n, "events_used": events, "status": "protein_not_in_model",
        }

    row = summ.loc[protein_col]
    return {
        "protein": protein_col, "beta": float(row["coef"]), "se": float(row["se(coef)"]),
        "HR": float(row["exp(coef)"]), "CI95_lower": float(row["exp(coef) lower 95%"]),
        "CI95_upper": float(row["exp(coef) upper 95%"]), "p": float(row["p"]),
        "N_used": n, "events_used": events, "status": "ok",
    }


def run_stratum(df_all: pd.DataFrame, mets_value: int, label: str, out_dir: Path) -> pd.DataFrame:
    d0 = df_all.loc[df_all[STRATA_COL] == mets_value].copy()
    for c in [TIME_COL, EVENT_COL] + COVARIATES:
        d0[c] = pd.to_numeric(d0[c], errors="coerce")

    protein_cols = infer_protein_cols(d0)
    d1, cov_cols = one_hot_covariates(d0, COVARIATES)
    base_cols = [TIME_COL, EVENT_COL] + cov_cols
    d_base = d1[base_cols + protein_cols].dropna(subset=base_cols, how="any").copy()

    for p in protein_cols:
        d_base[p] = zscore_series(d_base[p])

    rows = []
    for i, p in enumerate(protein_cols, start=1):
        if i == 1 or i % 500 == 0 or i == len(protein_cols):
            print(f"[PROGRESS] {label}: {i}/{len(protein_cols)}", flush=True)
        rows.append(fit_univariate_cox(d_base, cov_cols, p))

    res = pd.DataFrame(rows)
    m_tests = max(1, len(protein_cols))
    bonf_p_thr = 0.05 / m_tests
    res["p_bonferroni"] = np.clip(res["p"].values * m_tests, 0, 1)
    res["fdr_bh"] = benjamini_hochberg(res["p"].values)
    res["bonf_p_thr"] = bonf_p_thr
    res["sig_bonferroni"] = res["p_bonferroni"] < 0.05
    res["sig_fdr_005"] = res["fdr_bh"] < 0.05
    res.insert(0, "stratum", label)
    res["mets_value"] = mets_value

    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "cox_results.csv", index=False)
    res.loc[res["sig_bonferroni"], "protein"].to_csv(
        out_dir / "significant_proteins_bonferroni_p05.txt", index=False, header=False)
    res.loc[res["sig_fdr_005"], "protein"].to_csv(
        out_dir / "significant_proteins_fdr005.txt", index=False, header=False)

    summary = pd.DataFrame([{
        "stratum": label, "mets_value": mets_value,
        "n_total_in_stratum": int(d0.shape[0]),
        "events_total_in_stratum": int(pd.to_numeric(d0[EVENT_COL], errors="coerce").fillna(0).sum()),
        "n_after_drop_base": int(d_base.shape[0]),
        "events_after_drop_base": int(d_base[EVENT_COL].sum()),
        "n_proteins": int(len(protein_cols)), "bonf_p_thr": float(bonf_p_thr),
        "n_sig_bonf": int(res["sig_bonferroni"].sum()),
        "n_sig_fdr": int(res["sig_fdr_005"].sum()),
        "covariates": ",".join(COVARIATES),
    }])
    summary.to_csv(out_dir / "summary_counts.csv", index=False)
    return res


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 5.1: stratified univariate Cox models per plasma protein.")
    p.add_argument("--input", default="input_file.csv", help="Path to the input CSV.")
    p.add_argument("--out_dir", default="output", help="Output directory (created if missing).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    needed = [TIME_COL, EVENT_COL, STRATA_COL] + COVARIATES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    non_mets = run_stratum(df, mets_value=0, label="non_mets", out_dir=out_root / "non_mets")
    mets = run_stratum(df, mets_value=1, label="mets", out_dir=out_root / "mets")

    all_results = pd.concat([non_mets, mets], ignore_index=True)
    all_results.to_csv(out_root / "all_results_mets_vs_non_mets.csv", index=False)

    print(f"[DONE] Saved outputs to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
