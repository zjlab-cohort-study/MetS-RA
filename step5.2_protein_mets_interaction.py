#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5.2: protein x MetS interaction analysis (pooled Cox model).

For every protein column, a Cox model is fitted on the pooled cohort
(MetS + non-MetS together), stratified by MetS:

    h(t | MetS) = h0(t | MetS) * exp(
        b1 * protein_z              # protein effect in non-MetS (reference)
      + b3 * protein_z x MetS      # interaction: effect difference in MetS
      + g  * covariates)

Stratifying gives each group its own baseline hazard, so the estimate of b3
does not rely on proportional hazards for the strong MetS -> RA association;
b3 itself remains estimable and tests whether the protein-RA association
differs between the two groups.

"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


TIME_COL = "RA_followup_time"
EVENT_COL = "RA_label"
METS_COL = "MetS"  

COVARIATES = [
    "Sex",
    "Age",
    "Race",
    "Educational_Level",
    "Smoking",
    "Alcohol_Drinking",
    "Townsend_deprivation_index",
    "Household_Income",
]

# Columns that are not proteins; everything else is treated as a protein.
NON_PROTEIN_COLS = {
    "sample_id",
    EVENT_COL,
    TIME_COL,
    METS_COL,
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


def infer_protein_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_PROTEIN_COLS]


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


def fit_interaction(df_base: pd.DataFrame, cov_cols: list[str], protein: str) -> dict:
    inter_col = f"{protein}__x__mets"
    d = pd.DataFrame({
        TIME_COL: df_base[TIME_COL],
        EVENT_COL: df_base[EVENT_COL],
        METS_COL: df_base[METS_COL],
        protein: df_base[protein],
        inter_col: df_base[protein] * df_base[METS_COL],
    })
    d = pd.concat([d, df_base[cov_cols]], axis=1)
    d = d.dropna()

    n = int(d.shape[0])
    events = int(d[EVENT_COL].sum())
    base = {
        "protein": protein,
        "beta_int": np.nan, "se_int": np.nan, "HR_int": np.nan,
        "HR_int_lower": np.nan, "HR_int_upper": np.nan, "p_int": np.nan,
        "beta_protein": np.nan, "p_protein": np.nan,
        "N_used": n, "events_used": events, "status": "ok",
    }
    if n < 50 or events < 5:
        base["status"] = "skip_smallN"
        return base

    cph = CoxPHFitter(penalizer=0.0)
    try:
        cph.fit(d, duration_col=TIME_COL, event_col=EVENT_COL,
                strata=[METS_COL], show_progress=False)
    except Exception:
        base["status"] = "fit_error"
        return base

    summ = cph.summary
    if inter_col not in summ.index:
        base["status"] = "interaction_not_in_model"
        return base

    r_int = summ.loc[inter_col]
    base["beta_int"] = float(r_int["coef"])
    base["se_int"] = float(r_int["se(coef)"])
    base["HR_int"] = float(r_int["exp(coef)"])
    base["HR_int_lower"] = float(r_int["exp(coef) lower 95%"])
    base["HR_int_upper"] = float(r_int["exp(coef) upper 95%"])
    base["p_int"] = float(r_int["p"])
    if protein in summ.index:
        r_p = summ.loc[protein]
        base["beta_protein"] = float(r_p["coef"])
        base["p_protein"] = float(r_p["p"])
    return base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step 5.2: protein x MetS Cox interaction analysis.")
    p.add_argument("--input", default="input_file.csv", help="Path to the input CSV.")
    p.add_argument("--out_dir", default="output_interaction",
                   help="Output directory (created if missing).")
    p.add_argument("--union", default=None,
                   help="Optional text file of protein names to test (one per line); "
                        "default: all protein columns in the data.")
    p.add_argument("--stratified", default=None,
                   help="Optional step 5.1 all_results CSV (per-stratum protein effects) "
                        "merged to define the MetS-specific flags.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.union:
        union_file = Path(args.union)
        with union_file.open() as fh:
            wanted = [ln.strip().lower() for ln in fh if ln.strip()]
        print(f"[INFO] Union file: {union_file}", flush=True)
    else:
        wanted = None
        print("[INFO] Testing ALL protein columns in data", flush=True)
    print(f"[INFO] Output dir: {out_dir}", flush=True)

    df = pd.read_csv(args.input)
    have = set(df.columns)
    missing = [c for c in [TIME_COL, EVENT_COL, METS_COL] + COVARIATES if c not in have]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    all_proteins = infer_protein_cols(df)
    if wanted is None:
        proteins = all_proteins
    else:
        wanted_set = set(wanted)
        proteins = [p for p in all_proteins if p in wanted_set]
        print(f"[INFO] Requested {len(wanted)} proteins; "
              f"{len(wanted_set - set(all_proteins))} not found in data", flush=True)
    print(f"[INFO] Proteins to test: {len(proteins)}", flush=True)

    for c in [TIME_COL, EVENT_COL, METS_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df, cov_cols = one_hot_covariates(df, COVARIATES)

    for p in proteins:
        df[p] = zscore_series(df[p])

    base_cols = [TIME_COL, EVENT_COL, METS_COL] + cov_cols
    df_base = df.dropna(subset=base_cols, how="any").copy()
    print(f"[INFO] Pooled analysis N after covariate dropna: {len(df_base)} "
          f"(MetS={int((df_base[METS_COL]==1).sum())}, "
          f"non-MetS={int((df_base[METS_COL]==0).sum())})", flush=True)

    rows = []
    for i, p in enumerate(proteins, start=1):
        if i == 1 or i % 50 == 0 or i == len(proteins):
            print(f"[PROGRESS] {i}/{len(proteins)}", flush=True)
        rows.append(fit_interaction(df_base, cov_cols, p))

    res = pd.DataFrame(rows)

    m = res["p_int"].notna().sum()
    bonf_thr = 0.05 / max(1, m)
    res["p_int_fdr_bh"] = benjamini_hochberg(res["p_int"].values)
    res["p_int_bonf_thr"] = bonf_thr
    res["sig_interaction_bonf"] = res["p_int"] < bonf_thr
    res["sig_interaction_fdr"] = res["p_int_fdr_bh"] < 0.05

    if args.stratified and Path(args.stratified).exists():
        ar = pd.read_csv(args.stratified)
        keep = ["protein", "HR", "p", "sig_bonferroni", "sig_fdr_005"]
        mets = ar.loc[ar["stratum"] == "mets", keep].rename(
            columns={"HR": "HR_mets", "p": "p_mets",
                     "sig_bonferroni": "sig_bonf_mets", "sig_fdr_005": "sig_fdr_mets"})
        nmets = ar.loc[ar["stratum"] == "non_mets", keep].rename(
            columns={"HR": "HR_nonmets", "p": "p_nonmets",
                     "sig_bonferroni": "sig_bonf_nonmets", "sig_fdr_005": "sig_fdr_nonmets"})
        res = res.merge(mets, on="protein", how="left").merge(nmets, on="protein", how="left")
    else:
        print(f"[WARN] Stratified results not provided or not found; "
              f"stratified columns and MetS-specific flags will be NaN.", flush=True)
        for col in ["HR_mets", "p_mets", "sig_bonf_mets", "sig_fdr_mets",
                    "HR_nonmets", "p_nonmets", "sig_bonf_nonmets", "sig_fdr_nonmets"]:
            res[col] = np.nan

    res["mets_stronger"] = (
        (np.log(res["HR_mets"]).abs()) > (np.log(res["HR_nonmets"]).abs())
    )

    res["sig_in_mets"] = res["sig_fdr_mets"].fillna(False)
    res["sig_in_nonmets"] = res["sig_fdr_nonmets"].fillna(False)
    res["mets_specific"] = res["sig_interaction_fdr"] & res["sig_in_mets"]
    res["sig_interaction_nominal"] = res["p_int"] < 0.05

    # Relaxed, biology-driven definition: present in MetS, absent in non-MetS,
    # statistically different, and stronger in MetS.
    res["mets_specific_nominal"] = (
        res["sig_interaction_nominal"] & res["sig_in_mets"]
        & (~res["sig_in_nonmets"]) & res["mets_stronger"]
    )

    front = ["protein", "p_int", "p_int_fdr_bh", "sig_interaction_fdr",
             "sig_interaction_bonf", "beta_int", "HR_int", "HR_int_lower",
             "HR_int_upper", "HR_mets", "p_mets", "HR_nonmets", "p_nonmets",
             "mets_stronger", "sig_in_mets", "mets_specific", "N_used", "events_used"]
    res = res[[c for c in front if c in res.columns] +
              [c for c in res.columns if c not in front]]
    res = res.sort_values("p_int", na_position="last").reset_index(drop=True)

    res.to_csv(out_dir / "interaction_results_mets_vs_nonmets.csv", index=False)

    
if __name__ == "__main__":
    main()
