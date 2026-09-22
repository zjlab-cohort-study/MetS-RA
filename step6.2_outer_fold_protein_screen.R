#!/usr/bin/env Rscript

# Outcome-associated protein screening performed separately inside each outer
# training fold. The script receives all paths and variable mappings from JSON.

suppressPackageStartupMessages({
  library(data.table)
  library(jsonlite)
  library(parallel)
  library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("Usage: Rscript 02_outer_fold_protein_screen.R config.json")
cfg <- fromJSON(args[[1]], simplifyVector = TRUE)

read_input <- function(path, format) {
  if (tolower(format) %in% c("csv", "csv.gz")) return(fread(path))
  stop("The public R screening script currently accepts CSV input. Export an authorised, processed CSV locally or adapt read_input() for an approved format.")
}

d <- read_input(cfg$data_path, cfg$data_format)
d <- d[get(cfg$target_population_column) == cfg$target_population_value]
id_col <- cfg$id_column
event_col <- cfg$event_column
time_col <- cfg$time_column
if (anyDuplicated(d[[id_col]]) || anyNA(d[[id_col]])) stop("Identifier must be complete and unique")

proteins <- trimws(readLines(cfg$protein_columns_file, warn = FALSE))
proteins <- proteins[nzchar(proteins)]
if (anyDuplicated(proteins)) stop("Duplicate protein column names")
missing_proteins <- setdiff(proteins, names(d))
if (length(missing_proteins)) stop("Protein columns missing from input: ", paste(head(missing_proteins), collapse = ", "))

splits <- fread(file.path(cfg$output_dir, "outer_fold_assignments.csv"))
screen_dir <- file.path(cfg$output_dir, "screens")
dir.create(screen_dir, recursive = TRUE, showWarnings = FALSE)

categorical <- unlist(cfg$categorical_predictors)
for (v in categorical) {
  levels_v <- unlist(cfg$categorical_levels[[v]])
  d[[v]] <- factor(d[[v]], levels = levels_v)
}
adjustment <- unlist(cfg$screen_adjustment_predictors)
formula_rhs <- paste(adjustment, collapse = " + ")

screen_one <- function(train, common, protein) {
  raw <- as.numeric(train[[protein]])
  ok <- complete.cases(common) & is.finite(raw)
  n <- sum(ok)
  events <- sum(train[[event_col]][ok] == 1)
  miss <- mean(!is.finite(raw))
  sd_raw <- if (n > 1L) sd(raw[ok]) else NA_real_
  result <- data.table(protein = protein, status = "ok", n = n, events = events,
                       missing_fraction_train = miss, beta = NA_real_, se = NA_real_,
                       hr = NA_real_, p = NA_real_)
  if (miss > cfg$protein_missingness_max || n < cfg$protein_min_observations ||
      events < cfg$protein_min_events || !is.finite(sd_raw) || sd_raw <= 0) {
    result[, status := "insufficient_information"]
    return(result)
  }
  z <- (raw[ok] - mean(raw[ok])) / sd_raw
  x <- cbind(protein_z = z, common[ok, , drop = FALSE])
  y <- Surv(train[[time_col]][ok], train[[event_col]][ok])
  fit <- tryCatch(coxph.fit(x = x, y = y, strata = NULL, offset = NULL, init = NULL,
                            control = coxph.control(iter.max = 30), weights = NULL,
                            method = "efron", rownames = NULL, resid = FALSE),
                  error = function(e) e)
  if (inherits(fit, "error") || !is.finite(fit$coefficients[[1]]) || !is.finite(fit$var[1, 1])) {
    result[, status := "fit_error"]
    return(result)
  }
  b <- fit$coefficients[[1]]
  se_estimate <- sqrt(fit$var[1, 1])
  result[, `:=`(beta = b, se = se_estimate, hr = exp(b),
                p = 2 * pnorm(-abs(b / se_estimate)))]
  result
}

cores <- max(1L, as.integer(Sys.getenv("SCREEN_CORES", "1")))
run_screen <- function(train) {
  frame <- model.frame(as.formula(paste("~", formula_rhs)), data = train, na.action = na.pass)
  common <- model.matrix(as.formula(paste("~", formula_rhs)), data = frame)[, -1, drop = FALSE]
  result <- rbindlist(mclapply(proteins, function(p) screen_one(train, common, p),
                               mc.cores = cores, mc.preschedule = TRUE), fill = TRUE)
  tested <- result$status == "ok" & is.finite(result$p)
  result[, fdr_bh := NA_real_]
  result[tested, fdr_bh := p.adjust(p, method = "BH")]
  result[, fdr_selected := tested & fdr_bh < cfg$screen_fdr]
  setorder(result, p)
  result
}

for (r in seq_len(cfg$outer_repeats)) {
  map <- splits[repeat_id == r, c(id_col, "outer_fold"), with = FALSE]
  assignment <- map[match(d[[id_col]], map[[id_col]]), outer_fold]
  if (anyNA(assignment)) stop("Incomplete fold assignment")
  for (f in seq_len(cfg$outer_folds)) {
    train <- d[assignment != f]
    result <- run_screen(train)
    fwrite(result, file.path(screen_dir, sprintf("screen_repeat%02d_fold%02d.csv", r, f)))
  }
}
