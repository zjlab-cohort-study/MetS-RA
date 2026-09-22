#!/usr/bin/env Rscript
########################################################################
# Step 4: multi-mediator causal mediation analysis (full bootstrap)
# - Exposure: categorical
# - Mediators: multiple continuous variables, entered jointly into a
#   single outcome model
# - Outcome: survival (time + event)
# - Bootstrap: every resample refits the a-path models and the joint
#   multi-mediator outcome model from scratch
# - Multiple testing: mediator-level FDR (BH) and Bonferroni
########################################################################

library(dplyr)
library(survival)

# -------------------------------
# 0. User configuration
# -------------------------------
data_path <- "input_data.csv"
exposure  <- "MetS"
time_var  <- "RA_followup_time"
event_var <- "RA_label"

# Replace with the actual mediator column names in your data
mediators <- c("mediator1", "mediator2", "mediator3")

continuous_vars <- c("Age", "Townsend_deprivation_index")

categorical_vars <- c(
  "Sex", "Race", "Educational.Level",
  "Smoking", "Alcohol_Drinking",
  "Household_Income"
)

B <- 2000  # number of bootstrap replicates
set.seed(12345)

# -------------------------------
# 1. Read and check the data
# -------------------------------
dat0 <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)

covars <- c(continuous_vars, categorical_vars)
need_cols <- c(exposure, time_var, event_var, mediators, covars)
miss <- setdiff(need_cols, names(dat0))
if (length(miss) > 0) stop("Missing columns: ", paste(miss, collapse = ", "))

dat <- dat0

dat[categorical_vars] <- lapply(dat[categorical_vars], as.factor)
dat[continuous_vars] <- lapply(dat[continuous_vars], as.numeric)

event_values <- sort(unique(na.omit(dat[[event_var]])))
if (!all(event_values %in% c(0, 1))) {
  stop("event_var must be coded 0/1; current values: ", paste(event_values, collapse = ", "))
}
dat[[event_var]] <- as.numeric(as.character(dat[[event_var]]))

exposure_values <- sort(unique(na.omit(dat[[exposure]])))
if (length(exposure_values) < 2) {
  stop("exposure needs at least two levels; current values of ", exposure, ": ",
       paste(exposure_values, collapse = ", "))
}
dat[[exposure]] <- factor(dat[[exposure]], levels = exposure_values)

# Fetch a coefficient by term name (model term names may carry backticks)
extract_named_coef <- function(model, target_name) {
  coef_vec <- coef(model)
  coef_names <- names(coef_vec)

  if (is.null(coef_names)) return(NA_real_)

  clean_names <- gsub("`", "", coef_names)
  match_idx <- match(target_name, clean_names)
  if (is.na(match_idx)) return(NA_real_)

  as.numeric(coef_vec[[match_idx]])
}

# -------------------------------
# 2. Mediator standardisation
# -------------------------------
# Shapiro-Wilk normality check (on up to sample_n values)
check_normality <- function(x, sample_n = 5000, alpha = 0.05) {
  x <- x[!is.na(x)]

  out <- list(
    normal = FALSE,
    p_value = NA_real_,
    n_non_missing = length(x),
    n_tested = NA_integer_
  )

  if (length(x) < 3) return(out)
  if (length(unique(x)) < 3) return(out)

  x_test <- x
  if (length(x_test) > sample_n) x_test <- sample(x_test, sample_n)

  p_val <- tryCatch(
    shapiro.test(x_test)$p.value,
    error = function(e) NA_real_
  )

  out$p_value <- p_val
  out$n_tested <- length(x_test)
  out$normal <- !is.na(p_val) && p_val >= alpha
  out
}

mediator_distribution_summary <- data.frame(
  mediator = mediators,
  n_non_missing = NA_integer_,
  n_tested = NA_integer_,
  shapiro_p = NA_real_,
  normal_before = NA,
  transform = NA_character_,
  stringsAsFactors = FALSE
)

for (i in seq_along(mediators)) {
  m <- mediators[i]

  if (is.factor(dat[[m]])) {
    dat[[m]] <- as.numeric(as.character(dat[[m]]))
  } else if (is.character(dat[[m]])) {
    dat[[m]] <- as.numeric(dat[[m]])
  }

  x <- as.numeric(dat[[m]])
  normality_res <- check_normality(x)
  normal_flag <- normality_res$normal
  x_trans <- x
  method <- "zscore"

  if (!normal_flag) {
    method <- "log_then_zscore"
    min_x <- suppressWarnings(min(x, na.rm = TRUE))
    if (is.finite(min_x) && min_x <= 0) {
      x_trans <- log(x - min_x + 1)
    } else {
      x_trans <- log(x)
    }
  }

  m_mean <- mean(x_trans, na.rm = TRUE)
  m_sd <- sd(x_trans, na.rm = TRUE)

  if (is.na(m_sd) || m_sd == 0) {
    dat[[m]] <- ifelse(is.na(x_trans), NA_real_, 0)
  } else {
    dat[[m]] <- (x_trans - m_mean) / m_sd
  }

  mediator_distribution_summary$n_non_missing[i] <- normality_res$n_non_missing
  mediator_distribution_summary$n_tested[i] <- normality_res$n_tested
  mediator_distribution_summary$shapiro_p[i] <- normality_res$p_value
  mediator_distribution_summary$normal_before[i] <- normal_flag
  mediator_distribution_summary$transform[i] <- method
}

write.csv(
  mediator_distribution_summary,
  paste0("MultiMediator_distribution_summary_", event_var, ".csv"),
  row.names = FALSE
)

cat("Mediator distribution summary saved.\n")
cat("Normal -> zscore:", sum(mediator_distribution_summary$transform == "zscore"), "\n")
cat("Non-normal -> log_then_zscore:", sum(mediator_distribution_summary$transform == "log_then_zscore"), "\n")

# -------------------------------
# 3. Multi-mediator main analysis
# -------------------------------
all_model_cols <- c(exposure, time_var, event_var, covars, mediators)
df_full <- dat %>% dplyr::select(all_of(all_model_cols)) %>% na.omit()

if (nrow(df_full) < 50) {
  stop("Fewer than 50 complete cases after joint modelling; check the missing values.")
}

# a-path: one linear model per mediator (same complete-case sample)
a_fit_list <- list()
for (m in mediators) {
  f_a <- as.formula(paste0("`", m, "` ~ ", exposure, " + ", paste(covars, collapse = " + ")))
  fit_a <- tryCatch(lm(f_a, data = df_full), error = function(e) NULL)
  if (is.null(fit_a)) {
    stop("a-path model failed to fit: ", m)
  }
  a_fit_list[[m]] <- fit_a
}

# Use the first mediator's a-path model to determine the exposure coefficient name(s)
exposure_terms <- grep(paste0("^", exposure), names(coef(a_fit_list[[mediators[1]]])), value = TRUE)
if (length(exposure_terms) == 0) {
  stop("No exposure coefficient found in the a-path models; check the exposure coding.")
}

# c: total-effect model (mediators not included)
f_c <- as.formula(paste0(
  "Surv(", time_var, ",", event_var, ") ~ ", exposure, " + ", paste(covars, collapse = " + ")
))
fit_c <- tryCatch(coxph(f_c, data = df_full), error = function(e) NULL)
if (is.null(fit_c)) {
  stop("Total-effect model failed to fit.")
}
coef_c_all <- coef(fit_c)

# b and c': joint mediator model (all mediators entered at once)
mediator_terms_formula <- paste0("`", mediators, "`", collapse = " + ")
f_b <- as.formula(paste0(
  "Surv(", time_var, ",", event_var, ") ~ ", exposure,
  " + ", mediator_terms_formula,
  " + ", paste(covars, collapse = " + ")
))
fit_b <- tryCatch(coxph(f_b, data = df_full), error = function(e) NULL)
if (is.null(fit_b)) {
  stop("Joint multi-mediator Cox model failed to fit.")
}
coef_b_all <- coef(fit_b)

# Observed b_j per mediator
b_obs_map <- setNames(rep(NA_real_, length(mediators)), mediators)
for (m in mediators) {
  b_obs_map[[m]] <- extract_named_coef(fit_b, m)
}

# Observed a_jk and indirect_jk per mediator x contrast, plus the total indirect effect
obs_per_mediator <- list()
obs_total <- list()

for (term_name in exposure_terms) {
  total_indirect <- 0

  for (m in mediators) {
    a_jk <- tryCatch(coef(a_fit_list[[m]])[[term_name]], error = function(e) NA_real_)
    b_j <- b_obs_map[[m]]

    indirect_jk <- ifelse(is.na(a_jk) || is.na(b_j), NA_real_, as.numeric(a_jk * b_j))

    if (!is.na(indirect_jk)) total_indirect <- total_indirect + indirect_jk

    obs_per_mediator[[paste(m, term_name, sep = "__")]] <- list(
      mediator = m,
      contrast = term_name,
      a = as.numeric(a_jk),
      b = as.numeric(b_j),
      indirect = as.numeric(indirect_jk)
    )
  }

  c_total <- tryCatch(coef_c_all[[term_name]], error = function(e) NA_real_)
  c_prime <- tryCatch(coef_b_all[[term_name]], error = function(e) NA_real_)
  prop_med <- ifelse(is.na(c_total) || c_total == 0, NA_real_, as.numeric(total_indirect / c_total))

  obs_total[[term_name]] <- list(
    contrast = term_name,
    c_total = as.numeric(c_total),
    c_prime = as.numeric(c_prime),
    total_indirect = as.numeric(total_indirect),
    prop_mediated = as.numeric(prop_med)
  )
}

# -------------------------------
# 4. Bootstrap (full refit)
# -------------------------------
cat("Starting the multi-mediator bootstrap: B =", B, "\n")

# Bootstrap sampling distributions
boot_indirect_map <- list()   # per mediator x contrast
boot_total_map <- list()      # per contrast, total indirect
boot_cprime_map <- list()     # per contrast, c'
boot_ctotal_map <- list()     # per contrast, c

for (term_name in exposure_terms) {
  boot_total_map[[term_name]] <- rep(NA_real_, B)
  boot_cprime_map[[term_name]] <- rep(NA_real_, B)
  boot_ctotal_map[[term_name]] <- rep(NA_real_, B)

  for (m in mediators) {
    boot_indirect_map[[paste(m, term_name, sep = "__")]] <- rep(NA_real_, B)
  }
}

for (bi in 1:B) {
  sidx <- sample.int(nrow(df_full), size = nrow(df_full), replace = TRUE)
  df_b <- df_full[sidx, , drop = FALSE]

  # Refit the a-path for every mediator
  a_coef_boot <- vector("list", length(mediators))
  names(a_coef_boot) <- mediators
  a_ok <- TRUE

  for (m in mediators) {
    f_a <- as.formula(paste0("`", m, "` ~ ", exposure, " + ", paste(covars, collapse = " + ")))
    fa_b <- tryCatch(lm(f_a, data = df_b), error = function(e) NULL)
    if (is.null(fa_b)) {
      a_ok <- FALSE
      break
    }
    a_coef_boot[[m]] <- coef(fa_b)
  }

  if (!a_ok) next

  # Refit the total-effect model
  fc_b <- tryCatch(coxph(f_c, data = df_b), error = function(e) NULL)
  if (is.null(fc_b)) next
  c_all_b <- coef(fc_b)

  # Refit the multi-mediator outcome model
  fb_b <- tryCatch(coxph(f_b, data = df_b), error = function(e) NULL)
  if (is.null(fb_b)) next

  b_boot_map <- setNames(rep(NA_real_, length(mediators)), mediators)
  for (m in mediators) {
    b_boot_map[[m]] <- extract_named_coef(fb_b, m)
  }

  coef_fb_b <- coef(fb_b)

  for (term_name in exposure_terms) {
    total_indirect_b <- 0
    valid_all <- TRUE

    for (m in mediators) {
      a_jk_b <- tryCatch(a_coef_boot[[m]][[term_name]], error = function(e) NA_real_)
      b_j_b <- b_boot_map[[m]]

      if (is.na(a_jk_b) || is.na(b_j_b)) {
        valid_all <- FALSE
        next
      }

      ind_jk_b <- as.numeric(a_jk_b * b_j_b)
      boot_indirect_map[[paste(m, term_name, sep = "__")]][bi] <- ind_jk_b
      total_indirect_b <- total_indirect_b + ind_jk_b
    }

    if (valid_all) {
      boot_total_map[[term_name]][bi] <- total_indirect_b
    }

    boot_cprime_map[[term_name]][bi] <- tryCatch(as.numeric(coef_fb_b[[term_name]]), error = function(e) NA_real_)
    boot_ctotal_map[[term_name]][bi] <- tryCatch(as.numeric(c_all_b[[term_name]]), error = function(e) NA_real_)
  }

  if (bi %% 50 == 0) cat("bootstrap:", bi, "/", B, "\n")
}

# -------------------------------
# 5. Summarise the per-mediator results
# -------------------------------
per_mediator_rows <- lapply(obs_per_mediator, function(x) {
  key <- paste(x$mediator, x$contrast, sep = "__")
  boot_vals <- boot_indirect_map[[key]]
  boot_vals <- boot_vals[!is.na(boot_vals)]

  n_valid <- length(boot_vals)
  if (n_valid > 0) {
    ci_low <- as.numeric(quantile(boot_vals, probs = 0.025, na.rm = TRUE))
    ci_high <- as.numeric(quantile(boot_vals, probs = 0.975, na.rm = TRUE))
    p_boot <- 2 * min(mean(boot_vals <= 0), mean(boot_vals >= 0))
    p_boot <- min(p_boot, 1)
  } else {
    ci_low <- NA_real_
    ci_high <- NA_real_
    p_boot <- NA_real_
  }

  data.frame(
    mediator = x$mediator,
    contrast = x$contrast,
    a = x$a,
    b = x$b,
    indirect = x$indirect,
    indirect_CI_low = ci_low,
    indirect_CI_high = ci_high,
    indirect_p = p_boot,
    boot_valid = n_valid,
    stringsAsFactors = FALSE
  )
})

res_mediator <- do.call(rbind, per_mediator_rows)
res_mediator$indirect_p_FDR <- p.adjust(res_mediator$indirect_p, method = "BH")
res_mediator$indirect_p_Bonferroni <- p.adjust(res_mediator$indirect_p, method = "bonferroni")

# -------------------------------
# 6. Summarise the total (all mediators combined) results
# -------------------------------
total_rows <- lapply(obs_total, function(x) {
  term_name <- x$contrast

  boot_total <- boot_total_map[[term_name]]
  boot_total <- boot_total[!is.na(boot_total)]

  n_valid <- length(boot_total)
  if (n_valid > 0) {
    ci_low <- as.numeric(quantile(boot_total, probs = 0.025, na.rm = TRUE))
    ci_high <- as.numeric(quantile(boot_total, probs = 0.975, na.rm = TRUE))
    p_boot <- 2 * min(mean(boot_total <= 0), mean(boot_total >= 0))
    p_boot <- min(p_boot, 1)
  } else {
    ci_low <- NA_real_
    ci_high <- NA_real_
    p_boot <- NA_real_
  }

  data.frame(
    contrast = x$contrast,
    c_total = x$c_total,
    c_prime = x$c_prime,
    total_indirect = x$total_indirect,
    total_indirect_CI_low = ci_low,
    total_indirect_CI_high = ci_high,
    total_indirect_p = p_boot,
    total_boot_valid = n_valid,
    prop_mediated = x$prop_mediated,
    stringsAsFactors = FALSE
  )
})

res_total <- do.call(rbind, total_rows)
res_total$total_indirect_p_FDR <- p.adjust(res_total$total_indirect_p, method = "BH")
res_total$total_indirect_p_Bonferroni <- p.adjust(res_total$total_indirect_p, method = "bonferroni")

# -------------------------------
# 7. Save the results
# -------------------------------
out_mediator <- paste0("MultiMediation_per_mediator_B", B, "_", exposure, "_", event_var, ".csv")
out_total <- paste0("MultiMediation_total_B", B, "_", exposure, "_", event_var, ".csv")

write.csv(res_mediator, out_mediator, row.names = FALSE)
write.csv(res_total, out_total, row.names = FALSE)

cat("\nMulti-mediator results saved:\n")
cat("- ", out_mediator, "\n", sep = "")
cat("- ", out_total, "\n", sep = "")

cat("\n=== per mediator ===\n")
print(res_mediator)

cat("\n=== total indirect ===\n")
print(res_total)
