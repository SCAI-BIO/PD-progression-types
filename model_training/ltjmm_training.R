library(rstan)
library(ltjmm) # from https://bitbucket.org/mdonohue/ltjmm/src/master/
library(doParallel)

# rstan settings
options(mc.cores = parallel::detectCores())
options(auto_write = TRUE)

# Read data
# The data needs to be in long format with following columns: Patient_ID, Time.since.diag_scaled, Age_at_diagnosis_scaled, Sex, variable, value
# The column 'variable' contains the outcomes, i.e. the following values: UPDRS1_scaled, UPDRS2_scaled, UPDRS3_scaled, UPDRS4_scaled, PIGD_scaled, MCATOT_scaled, SCOPA_scaled
# All outcomes were min-max scaled regarding the theoratical minimum and maximum of the scores.
# Additionally, Age_at_diagnosis_scaled and Time.since.diag_scaled are calculated as the original times (in years) divided by 10. This speeds up the convergence of the chains.
data <- read.csv(file = "data/ppmi_ltjmm_melted.csv") # needs to be replaced

# fit LTJMM
fit <- ltjmm_stan(
  value ~ Time.since.diag_scaled |
    1 + Age_at_diagnosis_scaled + Sex |
    Patient_ID |
    variable ,
  random_effects = "multivariate",
  data = data,
  pars = c(
    "beta", "delta", "alpha0", "alpha1", "gamma",
    "sigma_delta", "sigma_y", "log_lik"
  ),
  open_progress = FALSE, chains = 4, iter = 25000, thin = 5, warmup = 12500, cores = detectCores() - 1, seed = 1, refresh = 10
)

# save the fit
save(fit, file = "fit.rdata")