#!/usr/bin/env Rscript
###############################################################################
#  Step 3.1: two-sample Mendelian randomisation (TwoSampleMR), MetS -> RA
#  ---------------------------------------------------------------------------
#  Exposure : GCST90444487, MetS shared-component genetic score
#             (GWAS Catalog harmonized TSV; Nat Genet 2024, N = 1,384,348,
#             European, continuous trait).
#  Outcome  : RA GWAS summary statistics in GWAS-VCF format (OUTCOME_FILE;
#             the default is FinnGen finn-b-M13_RHEUMA, any GWAS-VCF outcome
#             can be substituted).
#
#  Pipeline
#   - Instruments: exposure p < 5e-8, clumped locally with plink against the
#     1000 Genomes EUR reference (r2 < 0.001, 10 Mb window).
#   - Outcome statistics extracted by rsid from the local VCF via gwasvcf.
#   - harmonise_data aligns alleles and drops incompatible / palindromic SNPs.
#   - Estimation: IVW, MR-Egger, weighted median, simple/weighted mode; plus
#     heterogeneity (Cochran's Q), MR-Egger intercept, single-SNP Wald ratios,
#     leave-one-out, Steiger directionality test, and (if installed) MR-PRESSO.
#
#  Dependencies: TwoSampleMR, gwasvcf (0.1.5), genetics.binaRies,
#                VariantAnnotation, data.table. Data sources: see README.md.
###############################################################################

## 0. Binaries, file paths, parameters ========================================
suppressPackageStartupMessages({
  library(TwoSampleMR)
  library(gwasvcf)
  library(genetics.binaRies)
  library(VariantAnnotation)
  library(data.table)
})

# genetics.binaRies downloads plink / bcftools automatically on first use
plink_bin <- get_plink_binary()
bcftools  <- get_bcftools_binary()
set_bcftools(bcftools)

# Input files (download sources: see README.md)
EXPOSURE_FILE <- "GCST90444487.h.tsv.gz"
OUTCOME_FILE  <- "finn-b-M13_RHEUMA.vcf.gz"
# Sample column in the VCF header (follows the file name; adjust if it differs)
OUTCOME_ID    <- sub("\\.vcf\\.gz$", "", basename(OUTCOME_FILE))
OUTCOME_BUILD <- "GRCh37"

OUTDIR <- "MR_results"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# 1000 Genomes LD reference panel (~1.5 GB, downloaded once and reused)
REF_DIR <- "ld_ref"
REF_URL <- "http://fileserve.mrcieu.ac.uk/ld/1kg.v3.tgz"

P_THR      <- 5e-8     # instrument significance threshold
R2         <- 0.001    # clumping r2 threshold
KB         <- 10000    # clumping window (kb)
POP        <- "EUR"    # LD reference population
EXPOSURE_N <- 1384348  # exposure GWAS sample size (absent from the harmonized TSV; used by the Steiger test)

## 1. Local LD reference panel 
get_ld_bfile <- function(ref_dir, url, pop = "EUR") {
  find_prefix <- function() {
    bed <- list.files(ref_dir, pattern = paste0(pop, "\\.bed$"),
                      recursive = TRUE, full.names = TRUE)[1]
    if (is.na(bed)) NA_character_ else sub("\\.bed$", "", bed)
  }
  prefix <- find_prefix()
  if (!is.na(prefix)) { message("  [LD ref] already present: ", prefix); return(prefix) }

  dir.create(ref_dir, showWarnings = FALSE, recursive = TRUE)
  tgz <- file.path(ref_dir, basename(url))
  if (!file.exists(tgz)) {
    message("  [LD ref] downloading 1000G reference (~1.5 GB) ...")
    download.file(url, tgz, mode = "wb")
  }
  message("  [LD ref] extracting ...")
  untar(tgz, exdir = ref_dir)
  prefix <- find_prefix()
  if (is.na(prefix)) stop(pop, ".bed not found under ", ref_dir)
  prefix
}
message("\n==== 1. Preparing the LD reference panel ====")
bfile <- get_ld_bfile(REF_DIR, REF_URL, POP)

## 2. Read and format the exposure 
read_exposure <- function(file, p_thr) {
  message("  Reading the exposure GWAS (may take a few minutes) ...")
  raw <- as.data.frame(data.table::fread(file))  # format_data does not accept data.table
  message(sprintf("  SNPs in file: %s", format(nrow(raw), big.mark = ",")))

  # Pre-filter on significance to keep the clumping input small
  raw <- raw[!is.na(raw$p_value) & raw$p_value < p_thr, ]
  message(sprintf("  SNPs with p < %g: %s", p_thr, format(nrow(raw), big.mark = ",")))
  if (nrow(raw) == 0) stop("No SNP reaches the significance threshold; lower P_THR")

  exp <- TwoSampleMR::format_data(
    raw, type = "exposure",
    chr_col = "chromosome", pos_col = "base_pair_location",
    beta_col = "beta", se_col = "standard_error",
    pval_col = "p_value", log_pval = FALSE,  # p_value is the raw p, not -log10(p)
    effect_allele_col = "effect_allele", other_allele_col = "other_allele",
    eaf_col = "effect_allele_frequency", snp_col = "rsid")
  exp$exposure    <- "Metabolic syndrome"
  exp$id.exposure <- "GCST90444487"
  exp$samplesize.exposure <- EXPOSURE_N  # needed by the Steiger test
  exp
}
message("\n==== 2. Reading and formatting the exposure ====")
exposure_dat <- read_exposure(EXPOSURE_FILE, P_THR)

## 3. Local LD clumping 
message("\n==== 3. LD clumping ====")
exposure_dat <- clump_data(
  exposure_dat, clump_kb = KB, clump_r2 = R2, pop = POP,
  bfile = bfile, plink_bin = plink_bin)
message(sprintf("  Independent instruments: %d", nrow(exposure_dat)))
if (nrow(exposure_dat) < 3)
  warning("Fewer than 3 independent instruments; most MR methods cannot be estimated. Consider relaxing P_THR.")

## 4. Read the outcome from the GWAS-VCF 
ensure_tbi <- function(vcffile, bcf) {
  tbi <- paste0(vcffile, ".tbi")
  if (file.exists(tbi)) return(invisible(NULL))
  message("  Creating the tabix index for the VCF (~1-2 min) ...")
  out <- system2(bcf, c("index", "-t", shQuote(vcffile)), stdout = TRUE, stderr = TRUE)
  if (!file.exists(tbi))
    stop("tabix index creation failed:\n", paste(out, collapse = "\n"))
  invisible(NULL)
}

read_outcome <- function(snps, vcffile, id, build) {
  ensure_tbi(vcffile, bcftools)
  message(sprintf("  Extracting %d instruments from the outcome VCF ...", length(snps)))
  vcf <- gwasvcf::query_rsid_file(rsid = snps, vcffile = vcffile, id = id, build = build)
  tb <- gwasvcf::vcf_to_tibble(vcf)
  if (nrow(tb) == 0) stop("No instrument matched in the VCF; check rsid / build / id")
  message(sprintf("  Matched %d / %d SNPs", nrow(tb), length(snps)))

  # GWAS-VCF convention: ALT = effect allele, REF = other allele, LP = -log10(p)
  o <- data.frame(
    SNP           = as.character(tb$rsid),
    beta          = as.numeric(tb$ES),
    se            = as.numeric(tb$SE),
    pval          = 10 ^ (-as.numeric(tb$LP)),
    effect_allele = as.character(tb$ALT),
    other_allele  = as.character(tb$REF),
    eaf           = as.numeric(tb$AF),
    samplesize    = as.numeric(tb$SS),
    chr           = as.character(tb$seqnames),
    pos           = as.numeric(tb$start),
    stringsAsFactors = FALSE)
  o <- o[!is.na(o$beta) & !is.na(o$se) & !is.na(o$pval), ]

  o <- TwoSampleMR::format_data(
    o, type = "outcome",
    beta_col = "beta", se_col = "se", pval_col = "pval", log_pval = FALSE,
    effect_allele_col = "effect_allele", other_allele_col = "other_allele",
    eaf_col = "eaf", samplesize_col = "samplesize",
    chr_col = "chr", pos_col = "pos", snp_col = "SNP")
  o$outcome    <- "Rheumatoid arthritis"
  o$id.outcome <- id
  o
}
message("\n==== 4. Reading the outcome VCF ====")
outcome_dat <- read_outcome(exposure_dat$SNP, OUTCOME_FILE, OUTCOME_ID, OUTCOME_BUILD)

## 5. Harmonisation =
message("\n==== 5. Harmonising exposure and outcome ====")
dat <- harmonise_data(exposure_dat, outcome_dat, action = 2)
message(sprintf("  SNPs retained after harmonisation: %d", nrow(dat)))
write.csv(dat, file.path(OUTDIR, "harmonised_data.csv"), row.names = FALSE)

## 6. Main MR analysis 
message("\n==== 6. MR analysis ====")
res        <- mr(dat)                  # IVW / Egger / weighted median / modes
het        <- mr_heterogeneity(dat)    # Cochran's Q
pleio      <- mr_pleiotropy_test(dat)  # MR-Egger intercept (directional pleiotropy)
res_single <- mr_singlesnp(dat)        # Wald ratio per SNP
res_loo    <- mr_leaveoneout(dat)      # leave-one-out

# Approximate instrument strength (first-stage F statistic)
fstat <- (dat$beta.exposure / dat$se.exposure) ^ 2
message(sprintf("  Approximate F statistic: median %.1f (range %.1f-%.1f), all > 10: %s",
                median(fstat), min(fstat), max(fstat), all(fstat > 10)))

## 7. Steiger directionality test 
message("\n==== 7. Steiger directionality test ====")
dir_test <- tryCatch(directionality_test(dat),
                     error = function(e) { message("  Skipped: ", e$message); NULL })

## 8. MR-PRESSO 
presso <- NULL
if (requireNamespace("MRPRESSO", quietly = TRUE)) {
  message("\n==== 8. MR-PRESSO (may take a few minutes) ====")
  presso <- tryCatch(
    MRPRESSO::mr_presso(BetaOutcome = "beta.outcome", BetaExposure = "beta.exposure",
                        SdOutcome = "se.outcome", SdExposure = "se.exposure",
                        data = dat, OUTCOME = "outcome", EXPOSURE = "exposure",
                        NbDistribution = 1000, SignifThreshold = 0.05),
    error = function(e) { message("  MR-PRESSO failed: ", e$message); NULL })
} else {
  message("\n  (MRPRESSO package not installed, skipping; install.packages('MRPRESSO') to enable)")
}

## 9. Save result tables
message("\n==== 9. Saving results ====")
write.csv(res,        file.path(OUTDIR, "MR_main.csv"),            row.names = FALSE)
write.csv(het,        file.path(OUTDIR, "MR_heterogeneity.csv"),   row.names = FALSE)
write.csv(pleio,      file.path(OUTDIR, "MR_egger_intercept.csv"), row.names = FALSE)
write.csv(res_single, file.path(OUTDIR, "MR_singlesnp.csv"),       row.names = FALSE)
write.csv(res_loo,    file.path(OUTDIR, "MR_leaveoneout.csv"),     row.names = FALSE)
if (!is.null(dir_test)) write.csv(dir_test, file.path(OUTDIR, "MR_steiger.csv"), row.names = FALSE)

cat("\n================ MR main results ================\n")
print(res)
cat("\n--- Heterogeneity (Cochran's Q) ---\n");     print(het)
cat("\n--- Pleiotropy (MR-Egger intercept) ---\n"); print(pleio)
if (!is.null(dir_test)) { cat("\n--- Steiger directionality ---\n"); print(dir_test) }
if (!is.null(presso))    { cat("\n--- MR-PRESSO ---\n");              print(presso$MainMR) }

message("\nDone. Results saved to:\n  ", OUTDIR,
        "\n  - harmonised_data.csv",
        "\n  - MR_main / MR_heterogeneity / MR_egger_intercept / MR_singlesnp /",
        "\n    MR_leaveoneout / MR_steiger .csv")
