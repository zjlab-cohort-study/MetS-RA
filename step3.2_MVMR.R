#!/usr/bin/env Rscript
###############################################################################
#  Step 3.2: multivariable MR (TwoSampleMR), five MetS components -> RA
#  ---------------------------------------------------------------------------
#
#  Exposures (GWAS-VCF, GRCh37):
#    HDL cholesterol      ieu-a-299           continuous
#    Waist circumference  ukb-b-9405          continuous
#    Fasting glucose      ebi-a-GCST005186    continuous
#    Hypertension         ebi-a-GCST90038604  binary (log OR)
#    Triglycerides        ieu-a-302           continuous
#  Outcome: RA (ieu-a-832)
#
#  Dependencies: TwoSampleMR, gwasvcf (0.1.5), genetics.binaRies,
#                VariantAnnotation, data.table. Data sources: see README.md.
###############################################################################

## 0. Binaries, file paths, parameters 
suppressPackageStartupMessages({
  library(TwoSampleMR)
  library(gwasvcf)
  library(genetics.binaRies)
  library(VariantAnnotation)
  library(data.table)
})
bcftools <- get_bcftools_binary()
set_bcftools(bcftools)
plink_bin <- get_plink_binary()

OUTCOME_FILE <- "ieu-a-832.vcf.gz"
OUTDIR <- "MVMR_results"
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

# 1000 Genomes LD reference 
REF_DIR <- "ld_ref"
REF_URL <- "http://fileserve.mrcieu.ac.uk/ld/1kg.v3.tgz"

P_THR  <- 5e-8
LP_THR <- -log10(P_THR)  
R2     <- 0.001
KB     <- 10000
POP    <- "EUR"

TRAITS <- list(
  list(name = "HDL cholesterol",     id = "ieu-a-299",          file = "ieu-a-299.vcf.gz"),
  list(name = "Waist circumference", id = "ukb-b-9405",         file = "ukb-b-9405.vcf.gz"),
  list(name = "Fasting glucose",     id = "ebi-a-GCST005186",   file = "ebi-a-GCST005186.vcf.gz"),
  list(name = "Hypertension",        id = "ebi-a-GCST90038604", file = "ebi-a-GCST90038604.vcf.gz"),
  list(name = "Triglycerides",       id = "ieu-a-302",          file = "ieu-a-302.vcf.gz")
)

extract_vcf <- function(vcffile, filter_expr) {
  tmp <- tempfile(fileext = ".vcf.gz")
  system2(bcftools, c("view", "-i", shQuote(filter_expr), "-Oz", "-o", tmp, vcffile),
          stdout = FALSE, stderr = FALSE)
  v <- readVcf(tmp); unlink(tmp); v
}

# Create the tabix index for a VCF if missing
ensure_tbi <- function(vcffile) {
  if (file.exists(paste0(vcffile, ".tbi"))) return(invisible(NULL))
  message("    Creating tabix index: ", basename(vcffile))
  system2(bcftools, c("index", "-t", vcffile), stdout = FALSE, stderr = FALSE)
}

# GWAS-VCF convention: ALT = effect allele, REF = other allele, LP = -log10(p)
to_df <- function(v) {
  tb <- gwasvcf::vcf_to_tibble(v)
  data.frame(
    SNP           = as.character(tb$rsid),
    beta          = as.numeric(tb$ES),
    se            = as.numeric(tb$SE),
    pval          = 10 ^ (-as.numeric(tb$LP)),
    effect_allele = as.character(tb$ALT),
    other_allele  = as.character(tb$REF),
    eaf           = as.numeric(tb$AF),
    samplesize    = as.numeric(tb$SS),
    stringsAsFactors = FALSE)
}

# Format a data.frame as TwoSampleMR exposure / outcome data
fmt <- function(d, type, name, id) {
  d <- d[!is.na(d$beta) & !is.na(d$se) & !is.na(d$pval), ]
  d <- TwoSampleMR::format_data(
    d, type = type,
    beta_col = "beta", se_col = "se", pval_col = "pval", log_pval = FALSE,
    effect_allele_col = "effect_allele", other_allele_col = "other_allele",
    eaf_col = "eaf", samplesize_col = "samplesize", snp_col = "SNP")
  d[[type]] <- name
  d[[paste0("id.", type)]] <- id
  d
}

## 1. LD reference 
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
    message("  [LD ref] downloading ~1.5 GB ...")
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

## 2. Select instruments: p < 5e-8 per component -> union -> joint clumping 
message("\n==== 2. Selecting instruments and joint clumping ====")
inst_all <- list()  # significant SNPs per component (formatted)
trait_N  <- setNames(numeric(length(TRAITS)), sapply(TRAITS, `[[`, "name"))
for (tr in TRAITS) {
  ensure_tbi(tr$file)
  v <- extract_vcf(tr$file, sprintf("FMT/LP>%f", LP_THR))
  d <- fmt(to_df(v), "exposure", tr$name, tr$id)
  message(sprintf("    %-20s SNPs with p < %g: %d", tr$name, P_THR, nrow(d)))
  if (nrow(d) == 0) stop(tr$name, ": no significant SNP; check the data / threshold")
  inst_all[[tr$name]] <- d
  if (any(!is.na(d$samplesize.exposure)))
    trait_N[tr$name] <- as.integer(names(which.max(table(d$samplesize.exposure))))
}
all_inst <- do.call(rbind, inst_all)

clump_in <- aggregate(pval.exposure ~ SNP, data = all_inst[, c("SNP", "pval.exposure")], FUN = min)
clumped <- clump_data(clump_in, clump_kb = KB, clump_r2 = R2, pop = POP,
                      bfile = bfile, plink_bin = plink_bin)
combined_snps <- clumped$SNP
message(sprintf("  Merged independent instruments: %d", length(combined_snps)))

## 3. Build the balanced exposure matrix 
message("\n==== 3. Building the balanced exposure matrix ====")
snp_file <- file.path(OUTDIR, ".combined_snps.txt")
writeLines(combined_snps, snp_file)
id_filter <- sprintf("ID=@%s", snp_file)

exp_list <- list()
for (tr in TRAITS) {
  dd <- fmt(to_df(extract_vcf(tr$file, id_filter)), "exposure", tr$name, tr$id)
  exp_list[[tr$name]] <- dd
  message(sprintf("    %-20s matched %d / %d", tr$name, nrow(dd), length(combined_snps)))
}
exposure_dat <- do.call(rbind, exp_list)

# Complete-case: keep only SNPs with data in every component
keep <- Reduce(intersect, lapply(unique(exposure_dat$exposure),
                                 function(nm) exposure_dat$SNP[exposure_dat$exposure == nm]))
exposure_dat <- exposure_dat[exposure_dat$SNP %in% keep, ]
message(sprintf("  Complete SNPs (present in all %d components): %d", length(TRAITS), length(keep)))

## 4. Outcome extraction 
message("\n==== 4. Extracting the RA outcome ====")
ensure_tbi(OUTCOME_FILE)
od <- fmt(to_df(extract_vcf(OUTCOME_FILE, id_filter)), "outcome", "Rheumatoid arthritis", "ieu-a-832")
message(sprintf("  Outcome SNPs matched: %d", nrow(od)))

## 5. MVMR harmonisation 
message("\n==== 5. mv_harmonise_data ====")
mvdat <- mv_harmonise_data(exposure_dat, od, harmonise_strictness = 2)
n_iv <- nrow(mvdat$exposure_beta)
message(sprintf("  Instruments after harmonisation: %d x %d exposures", n_iv, ncol(mvdat$exposure_beta)))
if (n_iv < 3) stop("Fewer than 3 instruments after harmonisation; MVMR not estimable")

## 6. MVMR estimation 
message("\n==== 6. MVMR estimation ====")
# pval_threshold = 1 -> estimate all exposures using the full merged SNP set (standard union approach)
res_multiple <- mv_multiple(mvdat, pval_threshold = 1)$result
res_residual  <- mv_residual (mvdat, pval_threshold = 1)$result
res_ivw       <- mv_ivw      (mvdat, pval_threshold = 1)$result
res_lasso <- tryCatch(mv_lasso_feature_selection(mvdat),
                      error = function(e) { message("  lasso skipped: ", e$message); NULL })

cat("\n--- mv_residual (primary analysis: direct effect conditional on the other components) ---\n")
print(res_residual[, c("exposure", "nsnp", "b", "se", "pval")])

## 7. Diagnostics: instrument strength
message("\n==== 7. Diagnostics ====")
# Conventional (univariate) first-stage F per component; an upper bound under
# MVMR. Conditional F (Sanderson) would require the MVMR package.
Fstat <- sapply(inst_all, function(d) median((d$beta.exposure / d$se.exposure) ^ 2, na.rm = TRUE))
Ftab  <- data.frame(exposure = names(Fstat), median_F = round(Fstat, 1),
                    samplesize = trait_N[names(Fstat)], row.names = NULL)
print(Ftab)

## 8. Save result tables 
message("\n==== 8. Saving results ====")
write.csv(res_residual, file.path(OUTDIR, "MVMR_residual.csv"), row.names = FALSE)
write.csv(res_multiple, file.path(OUTDIR, "MVMR_multiple.csv"), row.names = FALSE)
write.csv(res_ivw,      file.path(OUTDIR, "MVMR_ivw.csv"),      row.names = FALSE)
write.csv(Ftab,         file.path(OUTDIR, "MVMR_Fstat.csv"),    row.names = FALSE)
write.csv(data.frame(SNP = rownames(mvdat$exposure_beta), mvdat$exposure_beta,
                     check.names = FALSE),
          file.path(OUTDIR, "MVMR_exposure_beta_matrix.csv"), row.names = FALSE)
if (!is.null(res_lasso)) write.csv(res_lasso, file.path(OUTDIR, "MVMR_lasso.csv"), row.names = FALSE)

cat("\n================ MVMR summary (mv_residual) ================\n")
print(res_residual[, c("exposure", "nsnp", "b", "se", "pval")])
message("\nDone. Results saved to:\n  ", OUTDIR,
        "\n  - MVMR_residual / MVMR_multiple / MVMR_ivw / MVMR_lasso / MVMR_Fstat /",
        "\n    MVMR_exposure_beta_matrix .csv")
