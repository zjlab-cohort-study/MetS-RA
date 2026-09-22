# Analysis code — metabolic syndrome and incident rheumatoid arthritis

Reproducible code for analysing the prospective association of metabolic syndrome
(MetS) with incident rheumatoid arthritis (RA) in a cohort with baseline covariates.

| Script | Analysis |
|---|---|
| `step1_Cox_MetS_RA.py` | Main Cox models: MetS, its five components, and dose–response by component count |
| `step2_Lag_timewindow.py` | Sensitivity analyses: 1/2/3-year landmark (lag) exclusions and 0–5 / 5–10 / >10-year time windows |
| `step3.1_MR.R` | Two-sample Mendelian randomisation: MetS → RA (IVW, MR-Egger, weighted median, modes, MR-PRESSO) |
| `step3.2_MVMR.R` | Multivariable MR: the five MetS components → RA, each conditional on the other four |
| `step4_mediation_multi_mediator.R` | Multi-mediator causal mediation analysis with full bootstrap |
| `step5.1_Cox_protein_RA.py` | Stratified univariate Cox models, one per plasma protein (MetS vs non-MetS strata) |
| `step5.2_protein_mets_interaction.py` | Protein × MetS interaction Cox models on the pooled cohort, stratified by MetS |
| `step6.1`–`step6.4` | Leakage-resistant protein prediction pipeline: outer splits → per-fold protein screening → nested elastic-net Cox fitting → performance summary with paired bootstrap |

## Requirements

Python ≥ 3.9:

```bash
pip install numpy pandas lifelines scikit-learn scikit-survival
```

R (steps 3–4, 6.2):

```r
install.packages(c("remotes", "data.table", "jsonlite", "dplyr",
                   "survival", "VariantAnnotation", "MRPRESSO"))
remotes::install_github(c("MRCIEU/TwoSampleMR",
                          "mrcieu/gwasvcf",            # version 0.1.5
                          "explodecomputer/genetics.binaRies"))
```

`MRPRESSO` is optional (enables MR-PRESSO in step 3.1); `plink` and `bcftools`
are managed automatically by `genetics.binaRies`.

---

## Step 1 — main Cox models (`step1_Cox_MetS_RA.py`)

Fits one Cox model per exposure — `MetS`, each of the five components, and
`components_total` (continuous, HR per additional abnormal component, plus
dummies for 1–5 components vs 0) — under three adjustments: Model-1 (crude),
Model-2 (+ Sex, Age, Race), Model-3 (+ education, smoking, alcohol, Townsend
deprivation, income, BMI).

```bash
python step1_Cox_MetS_RA.py --input data.csv --out_dir output
```

`--penalizer` adds an L2 term if a fit fails to converge; `--no-bmi` drops BMI
from Model-3 (sensitivity analysis without adjusting for a variable on the
causal pathway). Outputs, in `--out_dir`: `step1_single_exposure_cox_results.csv`
(HR/95% CI/p per exposure × model), `step1_covariate_cox_results.csv`,
`step1_components_count_cox_results.csv` (dose–response), `step1_run_summary.csv`.

---

## Step 2 — landmark and time-window sensitivity analyses (`step2_Lag_timewindow.py`)

Re-fits the Step 1 models (Model-3 without BMI) under two designs addressing
reverse causation: landmark analyses excluding participants whose follow-up ends
within the first 1/2/3 years, and time-window analyses splitting follow-up into
0–5 / 5–10 / >10-year windows (later windows restricted to participants still
observed at the window start).

```bash
python step2_Lag_timewindow.py --input data.csv --out_dir output
```

Outputs, in `--out_dir`: `step2_lag_cox_results_v2.csv`,
`step2_timewindow_cox_results_v2.csv`, `step2_lag_exclusion_summary_v2.csv`,
`step2_run_summary_v2.csv`, plus the per-design analysis cohorts under
`processed_lag/` and `processed_timewindow/`.

---

## Step 3 — Mendelian randomisation (`step3.1_MR.R`, `step3.2_MVMR.R`)

Two-sample MR using only public GWAS summary statistics. `step3.1_MR.R` estimates
the total effect of MetS on RA; `step3.2_MVMR.R` estimates the direct effect of
each component conditional on the other four. Instruments are selected at
p < 5e-8 and clumped locally with `plink` against the 1000 Genomes EUR reference
(r² < 0.001, 10 Mb window); the MVMR script jointly clumps the merged
per-component sets so instruments are independent across components.

Input data (public downloads, file names configured at the top of each script):

| File | Dataset |
|---|---|
| `GCST90444487.h.tsv.gz` | MetS shared-component genetic score, N = 1,384,348 (3.1 exposure; GWAS Catalog) |
| `finn-b-M13_RHEUMA.vcf.gz` | RA, FinnGen (3.1 outcome; any GWAS-VCF outcome can be substituted) |
| `ieu-a-299.vcf.gz` | HDL cholesterol |
| `ukb-b-9405.vcf.gz` | Waist circumference |
| `ebi-a-GCST005186.vcf.gz` | Fasting glucose |
| `ebi-a-GCST90038604.vcf.gz` | Hypertension |
| `ieu-a-302.vcf.gz` | Triglycerides |
| `ieu-a-832.vcf.gz` | RA, Okada 2014 (3.2 outcome) |

All VCFs are IEU OpenGWAS downloads in GWAS-VCF format (INFO fields
`ES`/`SE`/`LP`/`AF`/`SS`; ALT = effect allele). `EXPOSURE_N` in `step3.1_MR.R`
must match the exposure GWAS sample size. The LD reference (~1.5 GB) is
downloaded automatically on first run into `ld_ref/` and reused afterwards.

```bash
Rscript step3.1_MR.R    # univariate MR: MetS -> RA
Rscript step3.2_MVMR.R  # multivariable MR: five components -> RA
```

Outputs are CSV tables only: `MR_results/` (harmonised instrument data;
per-method estimates, heterogeneity, Egger intercept, single-SNP and
leave-one-out, Steiger) and `MVMR_results/` (primary `MVMR_residual.csv`,
alternative estimators, first-stage F statistics, exposure matrix).

---

## Step 4 — multi-mediator mediation analysis (`step4_mediation_multi_mediator.R`)

Estimates how much of the exposure → RA association is transmitted through a set
of continuous mediators, with uncertainty from a full bootstrap (default
`B = 2000`; every resample refits the a-path linear models and the joint
multi-mediator Cox model — reduce `B` for a quick test run).

Edit the **User configuration** block at the top before running — in particular
replace the placeholder `mediators` (`mediator1`, …) with the actual mediator
column names in your input CSV. Input is one CSV, one row per participant, with
the exposure, survival outcome (`time_var` / `event_var`), mediators and
covariates; the analysis is complete-case, and mediators are standardised
automatically (z-score, with a log transform first when non-normal).

```bash
Rscript step4_mediation_multi_mediator.R
```

Run from the directory containing the input CSV. Outputs:
`MultiMediator_distribution_summary_<event_var>.csv` (transform applied per
mediator), `MultiMediation_per_mediator_B<B>_*.csv` (a/b paths, indirect
effects, bootstrap CIs, FDR/Bonferroni p) and `MultiMediation_total_B<B>_*.csv`
(total/direct effects, proportion mediated).

---

## Step 5 — protein-level analyses (`step5.1_Cox_protein_RA.py`, `step5.2_protein_mets_interaction.py`)

Input is one CSV, one row per participant: the survival outcome, covariates, a
pre-derived `MetS` column (0/1) and one column per protein — every column
outside the reserved set is treated as a protein. Proteins are z-scored
internally (within stratum in 5.1, pooled in 5.2) and missing values are dropped
per model. BMI is deliberately not adjusted for (causal pathway).

- **5.1** fits one covariate-adjusted Cox model per protein separately within
  each MetS stratum, with Bonferroni and BH correction across proteins.
- **5.2** fits one pooled model per protein, stratified by MetS, with a
  protein × MetS interaction term (stratification avoids relying on
  proportional hazards for MetS). A protein is flagged **MetS-specific** when
  the interaction is FDR-significant and the protein is FDR-significant in the
  MetS stratum (Step 5.1 results, linked via `--stratified`).

```bash
python step5.1_Cox_protein_RA.py --input protein_data.csv --out_dir output_step51
python step5.2_protein_mets_interaction.py --input protein_data.csv \
    --out_dir output_step52 --stratified output_step51/all_results_mets_vs_non_mets.csv
```

In 5.2, `--union` restricts testing to protein names listed in a text file
(default: all proteins). Outputs: per-stratum `cox_results.csv`, significance
lists and `summary_counts.csv`, plus combined `all_results_mets_vs_non_mets.csv`
(5.1); `interaction_results_mets_vs_nonmets.csv` and protein-name lists per
threshold / MetS-specificity (5.2).

---

## Step 6 — protein prediction pipeline (`step6.1`–`step6.4`)

Repeated nested cross-validation comparing four model tiers
(`clinical_metabolic`, `clinical_inflammatory`, `protein_only`, `integrated`).
Every data-dependent step — protein screening, correlation pruning, imputation,
winsorisation, scaling, penalty tuning — is re-estimated inside each
outer-training fold only; outer-test folds are untouched until evaluation.

| Script | Role |
|---|---|
| `pipeline_common.py` | Shared Python module (config, cohort selection, preprocessing); keep next to the step6 scripts |
| `step6.1_make_outer_splits.py` | Event-stratified repeated outer-fold assignments |
| `step6.2_outer_fold_protein_screen.R` | Covariate-adjusted univariate Cox screen of every candidate protein within each outer-training fold (FDR selection) |
| `step6.3_fit_nested_models.py` | Correlation pruning, inner-CV elastic-net penalty selection (one-SE rule), outer-test evaluation (Harrell C, Uno C, time-dependent AUC, Brier) |
| `step6.4_summarize_and_bootstrap.py` | Aggregate outer-fold performance; paired participant-cluster bootstrap CIs for incremental value |

All scripts are driven by one JSON file: copy `config.example.json` and adapt
the paths/column names, predictor lists (categorical predictors take
prespecified `categorical_levels`, first level = reference), screening
thresholds (`screen_fdr`, missingness/event minima) and CV settings
(`outer_repeats`, `outer_folds`, `inner_folds`, `horizons_years`, …). The R
screen reads CSV input only.

```bash
python step6.1_make_outer_splits.py --config config.json
Rscript step6.2_outer_fold_protein_screen.R config.json   # SCREEN_CORES=8 optional
python step6.3_fit_nested_models.py --config config.json  # re-entrant; --overwrite to redo
python step6.4_summarize_and_bootstrap.py --config config.json \
    --bootstrap 1000 --comparison integrated,clinical_inflammatory
```

Outputs under `output_dir`: `outer_fold_assignments.csv`,
`screens/screen_repeat<rr>_fold<ff>.csv`, `prediction_manifest.json`,
`all_outer_predictions.csv` (out-of-fold scores and survival probabilities),
per-fold / per-repeat / overall performance (`model_performance_summary.csv`
and companions), `all_tuning_results.csv`, feature coefficients and protein
selection stability, and the paired bootstrap replicates and 95% CIs
(`paired_participant_cluster_bootstrap_ci.csv`; positive = the new model
improves on the reference). The bootstrap resamples participants rather than
folds, preserving the pairing created by repeated out-of-fold predictions.
