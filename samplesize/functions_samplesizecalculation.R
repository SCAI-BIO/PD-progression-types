library(longpower)
library(plyr)
library(dplyr)
source("longpower_fix.R") # fixes an error in lmmpowwer (see https://github.com/mcdonohue/longpower/issues/6)

#' Calculates the power or sample size of a simulated clinical trial using longpower package and a linear mixed model on a given clinical dataset. The calculation can be performed directly for the given clinical cohort (resample=FALSE) or for an resampled cohort with adapted fraction of fast-progressing patients (resample=TRUE).
#'
#' @param outcome The outcome used as primary outcome for sample size/power calculation in the simulated randomized controlled trial. Needs to be a column in data_visits. UPDRS I-III sum was used as primary outcome in the publication.
#' @param data_visits A data.frame containing the longitudinal clinical data used for fitting the linear mixed model. The following columns are required: Patient_ID, Timepoint_scaled (time on individual time scale in 10-years scale, not common disease timescale) and cluster.
#' @param t_visits Vector of times of visits (in years) where the outcome should be measured in the simulated study. By default, visits each 60 days for one year are used.
#' @param treatment_eff Expected treatment effect, i.e. how much disease progression is slowed by the treatment compared to placebo.
#' @param power The desired statistical power of the clinical trial to calculate the sample size. Either power or n needs to be given. The other value will be calculated and returned.
#' @param n The desired sample size of the clinical trial to calculate the statistical power. Either power or n needs to be given. The other value will be calculated and returned.
#' @param sig_level The desired significance level of the simulated study.
#' @param resample Whether the calculation should be performed directly for the given clinical cohort (resample=FALSE) or for an resampled cohort with adapted fraction of fast-progressing patients (resample=TRUE).
#' @param resample_ratio The fraction of fast-progressing patients in the simulated cohort. Patients will be resampled from the given clinical cohort and the cluster-column such that the desired fraction of fast-progressing patients is achieved. (only used if resample=TRUE)
#' @param resample_n_bootstrap How often resampling should be performed (only used if resample=TRUE)
#' @param method The longpower method used for sample size or power calculation
#' @return  An object of class \code{power.htest} giving the calculated sample size and power (resample=FALSE) or a list of such objects with length resample_n_bootstrap (resample=TRUE).
calc_power_lmm <- function(outcome = "UPDRS13",
                           data_visits,
                           t_visits = seq(0, 360, by = 60) / 365,
                           treatment_eff = 0.3,
                           power = 0.9,
                           n = NULL,
                           sig_level = 0.1,
                           resample = FALSE,
                           resample_ratio = NULL,
                           resample_n_bootstrap = 100,
                           method = "edland") {

  # convert Timpepoint from 10-year-scale back to years (scaling was applied for LTJMM to speed up MCMC mixing)
  data_visits$Timepoint_scaled <- data_visits$Timepoint_scaled * 10
  data_visits <- data_visits[c("Timepoint_scaled", "Patient_ID", "cluster", outcome)]

  # remove all patients with only one measurement as we want to calculate slopes
  data_visits <- data_visits %>% group_by(Patient_ID) %>% filter(n() > 1)

  if (resample) {
    # Resample patients from the cohort to achieve the given resample_ratio fraction of fast-progressing patients; this is reapeated resample_n_bootstrap times

    # identify patients from fast-progressing and slow-progressing subtype
    n_patients <- length(unique(data_visits$Patient_ID))
    patIDs_cl1 <- unique(data_visits[data_visits$cluster == 1, ]$Patient_ID)
    patIDs_cl2 <- unique(data_visits[data_visits$cluster == 2, ]$Patient_ID)

    # if no resample_ratio is provided, we retrieve the fraction of fast-progressing patients from the original cohort
    if (is.null(resample_ratio)){
      resample_ratio <- length(patIDs_cl1) / n_patients
    }

    # perform repeated sample size / power analysis
    pwr_list <- list()
    for (j in 1:resample_n_bootstrap) {
      # sample patients from the original cohort with the given fraction of fast-progressing patients
      patIDs_cl1_sampled <- sample(patIDs_cl1, size = n_patients * resample_ratio, replace = TRUE)
      patIDs_cl2_sampled <- sample(patIDs_cl2, size = n_patients * (1 - resample_ratio), replace = TRUE)
      patids_sampled <- c(patIDs_cl1_sampled, patIDs_cl2_sampled)

      # assign new patient IDs to the resampled cohort (to prevent the same ID used for multiple patients)
      patids_sampled_df <- tibble(Original_ID = patids_sampled,  New_ID = paste0("New_", seq_along(patids_sampled)))
      data_visits_sampled <- patids_sampled_df %>% left_join(data_visits, by = c("Original_ID" = "Patient_ID"), multiple = "all")

      # create a linear mixed model of outcome progression on which we base the sample size/power estimation
      lmm <- lmer(as.formula(paste(outcome, "~ Timepoint_scaled + (1 + Timepoint_scaled | New_ID)")), data = data_visits_sampled)

      # calculate power / sample size using longpower package
      pwr_list[[j]] <- lmmpower(lmm, pct.change = treatment_eff, t = t_visits, power = power, sig.level = sig_level, n = n, method = method)
    }
    return(pwr_list)

  } else {
    # Without resampling

    # create a linear mixed model of outcome progression on which we base the sample size/power estimation
    lmm <- lmer(as.formula(paste(outcome, "~ Timepoint_scaled + (1 + Timepoint_scaled | Patient_ID)")), data = data_visits)

    # calculate power / sample size using longpower package
    pwr <- lmmpower(lmm, pct.change = treatment_eff, t = t_visits, power = power, sig.level = sig_level, n = n, method = method)
    return(pwr)
  }
}

#' Calculates the power - sample size curve of a simulated clinical trial using longpower package and a linear mixed model on a given clinical dataset. The calculation can be performed directly for the given clinical cohort (resample=FALSE) or for an resampled cohort with adapted fraction of fast-progressing patients (resample=TRUE).
#'
#' @param outcome The outcome used as primary outcome for sample size/power calculation in the simulated randomized controlled trial. Needs to be a column in data_visits. UPDRS I-III sum was used as primary outcome in the publication.
#' @param data_visits A data.frame containing the longitudinal clinical data used for fitting the linear mixed model. The following columns are required: Patient_ID, Timepoint_scaled (time on individual time scale in 10-years scale, not common disease timescale) and cluster.
#' @param t_visits Vector of times of visits (in years) where the outcome should be measured in the simulated study. By default, visits each 60 days for one year are used.
#' @param treatment_eff Expected treatment effect, i.e. how much disease progression is slowed by the treatment compared to placebo.
#' @param n A vector of sample size for which the power should be calculated, i.e. the x-variable of the power-sample size plot.
#' @param sig_level The desired significance level of the simulated study.
#' @param ci_probs Probabilities used for condifence interval calculation (default is c(0.025, 0.975) for 95% CI).
#' @param resample Whether the calculation should be performed directly for the given clinical cohort (resample=FALSE) or for an resampled cohort with adapted fraction of fast-progressing patients (resample=TRUE).
#' @param resample_ratio The fraction of fast-progressing patients in the simulated cohort. Patients will be resampled from the given clinical cohort and the cluster-column such that the desired fraction of fast-progressing patients is achieved. (only used if resample=TRUE)
#' @param resample_n_bootstrap How often resampling should be performed (only used if resample=TRUE)
#' @param method The longpower method used for sample size or power calculation
#' @return  An data.frame containing values of the power-sample size curve. The following columns are contained in the data.frame: outcome (outcome used as primary outcome, as provided), n (n for that power was calculated, as provided), power (calculated power), power_upper (calculated confidence interval limit of the power), power_lower (calculated lower confidence interval limit of the power)
calc_samplesize_curve <- function(outcome = "UPDRS13",
                                  data_visits,
                                  t_visits = seq(0, 360, by = 60) / 365,
                                  treatment_eff = 0.3,
                                  n,
                                  sig_level = 0.1,
                                  ci_probs = c(0.025, 0.975),
                                  resample = FALSE,
                                  resample_ratio = NULL,
                                  resample_n_bootstrap = 100,
                                  method = "edland") {
  if (resample) {
    # Resample patients from the cohort to achieve the given resample_ratio fraction of fast-progressing patients; this is done and reapeated resample_n_bootstrap times within calc_power_lmm
    df_power <- data.frame()

    # calculate power for each given n
    for (n_i in n){
      pwr_list <- calc_power_lmm(outcome = outcome, data_visits = data_visits, t_visits = t_visits, treatment_eff = treatment_eff, n = n_i, sig_level = sig_level, resample = TRUE, power = NULL, method = method, resample_ratio = resample_ratio, resample_n_bootstrap = resample_n_bootstrap)

      # extract the power from the object returned
      pwr_vec <- laply(pwr_list, function(x) {
        x$power
      })

      df_power <- rbind(df_power, data.frame(outcome = outcome, n = n_i, power = median(pwr_vec), power_lower = quantile(pwr_vec, probs = ci_probs[1])[[1]], power_upper = quantile(pwr_vec, probs = ci_probs[2])[[1]]))
    }
    return(df_power)

  } else {
    # Without resampling
    df_power <- data.frame()

    # calculate power for each given n
    for (n_i in n){
      pwr_obj <- calc_power_lmm(outcome = outcome, data_visits = data_visits, t_visits = t_visits, treatment_eff = treatment_eff, n = n_i, sig_level = sig_level, resample = FALSE, power = NULL, method = method)
      df_power <- rbind(df_power, data.frame(outcome = outcome, n = n_i, power = pwr_obj$power))
    }
    return(df_power)
  }
}