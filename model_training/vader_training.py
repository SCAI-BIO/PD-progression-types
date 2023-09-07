import numpy as np
import pandas as pd
from collections import OrderedDict

# VaDER imports, VaDER can be obtained from https://github.com/yalchik/VaDER
from vader.utils.data_utils import map_xdict_to_xtensor, generate_wtensor_from_xtensor
from vader.utils.clustering_utils import ClusteringUtils
from vader.hp_opt.vader_hyperparameters_optimizer import VADERHyperparametersOptimizer
from vader.hp_opt.interface.abstract_grid_search_params_factory import AbstractGridSearchParamsFactory

class ParamsFactory(AbstractGridSearchParamsFactory):
    """
    Class used for random sampling of hyperparameters from a hyperparameter grid.
    An instance of this class is passed to the VaDER hyperparameter optimizer.
    """

    def get_full_param_dict(self):
        """
        Returns an dictionary containing the possible values of all hyperparameters from which hyperparameters are sampled during the hyperparameter training.

        Returns:
            dict: dictionary containing hyperparameter names as keys and list of possible values as values
        """
        param_dict = {
            "k": list(range(2, 5)),
            "n_hidden": self.gen_list_of_combinations([0, 1, 2, 3, 4, 5, 6]), # all combinations of 1, 2, 4, 8, 16, 32, 64 nodes per layer; 2 layers
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64], 
            "alpha": [1.0]
        }
        return param_dict

def predict_outcome(data_visits: pd.DataFrame, data_params: dict, outcome_id: int, covariates_list: list, timevariable: str) -> np.array:
    """
    Predicts one outcome for given patients and times using parameters from a LTJMM fit.

    Args:
        data_visits (pd.DataFrame): A pandas DataFrame containing all patient data and the times for which predictions of outcome n should be made. Following columns are needed:fit_latenttime (latent time in 10-year scale), fit_alpha0_n (random intercept for the patient of outcome n), fit_alpha1_n (random slope for the patient of outcome n), all covariates provided in covariates_list. Counting of n is zero-based here.
        data_params (dict): A dictionary storing the LTJMM parameters shared between patients. The following keys are required: covariate coefficients for all outcomes: beta[n+1,i], mean slope for all outcomes: gamma[n+1]. (For each i in 1...number of covariates and with n as the ID of the outcome. 
        outcome_id (int): ID of the outcome in the LTJMM model.
        covariates_list (list(str)): List of all covariates used for LTJMM fitting. 
        timevariable (str): Name of the timevariable column used as timescale in LTJMM.

    Returns:
        np.array: Array of predictions in the same order as the input data_visits.
    """

    # initialize predictions with value 0
    pred = np.zeros(data_visits.shape[0])

    # fixed covariates / betas
    # add fixed intercept
    pred += data_params["beta[" + str(outcome_id+1) + ",1]"]

    # add fixed covariate effects
    for i in range(0, len(covariates_list)):
        pred += data_params["beta[" + str(outcome_id+1) + "," + str(i+2) + "]"] * data_visits[covariates_list[i]]

    # add mean progression / gamma
    pred += data_visits["fit_latenttime"] * data_params["gamma[" + str(outcome_id+1) + "]"]

    # add random intercept / alpha0
    pred += data_visits["fit_alpha0_" + str(outcome_id + 1)]

    # add random slope / alpha1
    pred += data_visits["fit_alpha1_" + str(outcome_id + 1)] * data_visits[timevariable]

    return pred.values

# converts longitudinal data into tensors as required for VaDER
def get_vader_tensors(data_patients: pd.DataFrame, timevariable: str, covariates: list, outcomes: list, params: dict, times: list = [-1, 0, 1], verbose: bool = False) -> (np.ndarray, np.ndarray):
    """
    Calculates LTJMM predictions for VaDER. Based on these predictions, the X-tensor (outcome progression scores) and W-Tensor (non-missing indicator) required for VaDER fitting are calculated and returned.

    Args:
        data_patients (pd.DataFrame):  A pandas DataFrame containing patient information with one patient per row and following columns: fit_delta (timeshift from LTJMM), fit_latenttime (latent time from LTJMM), fit_alpha0_i (random intercepts for all outcomes i), fit_alpha1_i (random slopes for all outcomes i).
        timevariable (str): Name of the timevariable column used as timescale in LTJMM.
        covariates (list): List of all covariates used for LTJMM fitting. 
        outcomes (list): List of all outcomes used in LTJMM, needs to be in correct order.
        params (dict): A dictionary storing the LTJMM parameters shared between patients. The following keys are required: covariate coefficients for all outcomes: beta[n+1,i], mean slope for all outcomes: gamma[n+1]. (For each i in 1...number of covariates and with n as the ID of the outcome. 
        times (list, optional): List of times on the latent time scale for which LTJMM predictions should be calculated. Defaults to [-1, 0, 1].

    Returns:
        (np.ndarray, np.ndarray): Tuple of X-tensor (3D numpy array, where 1st dimension is samples, 2nd dimension is time points, 3rd dimension is feature vectors) and W-tensor (same structure as X-tensor, but 0/1-values as indicators for non-missing values) as required for VaDER fitting. 
    """

    # construct a dataframe with the LTJMM-predicted outcomes for patients for given timepoints 
    data_pred = pd.DataFrame() 
    data_pred["fit_latenttime"] = np.tile(times, data_patients.shape[0])

    # copy timeshift
    data_pred["fit_delta"] = np.repeat(data_patients["fit_delta"].values, len(times))

    # calculate original time (latenttime = original time + timeshift => original time = latenttime - timeshift)
    data_pred[timevariable] = data_pred["fit_latenttime"] - data_pred["fit_delta"]

    # copy covariates into this dataframe
    for covariate in covariates:
        data_pred[covariate] = np.repeat(data_patients[covariate].values, len(times))

    # copy random effect variables into this dataframe
    for outcome_id in range(1, len(outcomes)+1):
        data_pred["fit_alpha0_" + str(outcome_id)] = np.repeat(data_patients["fit_alpha0_" + str(outcome_id)].values, len(times))
        data_pred["fit_alpha1_" + str(outcome_id)] = np.repeat(data_patients["fit_alpha1_" + str(outcome_id)].values, len(times))

    # predict the outcomes using LTJMM
    for outcome_id in range(0, len(outcomes)):
        data_pred[outcomes[outcome_id]] = predict_outcome(data_visits = data_pred, data_params = params, outcome_id = outcome_id, covariates_list = covariates, timevariable = timevariable) 

    # convert dataframe to a ordered dict
    data_dict = OrderedDict.fromkeys(outcomes)
    for feature in outcomes:
        data_dict[feature] = OrderedDict.fromkeys(data_pred["fit_latenttime"].unique())
        for time in data_pred["fit_latenttime"].unique():
            data_dict[feature][time] = data_pred[data_pred["fit_latenttime"] == time][feature].values

    # convert to tensors using VaDER methods
    # Dimensionen: patients, times, outcomes
    x_tensor_with_nans_orig = map_xdict_to_xtensor(data_dict)

    # calc z-scores ('outcome progression scores'): z-score as deviation based on std of baseline values
    data_std = data_pred[data_pred["fit_latenttime"] == data_pred["fit_latenttime"].min()][outcomes].std()
    x_tensor_with_nans = ClusteringUtils.calc_z_scores(x_tensor_with_nans_orig, data_std)

    # remove NaN's and create weights (we don't have NANs in this situation, could be skipped)
    x_tensor_no_na = np.nan_to_num(x_tensor_with_nans)

    # generate w_tensor (non-missing value indicator)
    w_tensor = generate_wtensor_from_xtensor(x_tensor_with_nans)
    
    return x_tensor_no_na, w_tensor


# load clinical data and LTJMM fit, define outcomes and LTJMM covariates
covariates = ["Age_at_diagnosis_scaled", "Sex"] # list of all covariates as used in LTJMM, included in correct order
outcomes = ["UPDRS1_scaled", "UPDRS2_scaled", "UPDRS3_scaled", "UPDRS4_scaled", "PIGD_scaled", "MCATOT_scaled", "SCOPA_scaled"] # list of all outcomes used for VaDER fitting; outcomes were min-max scaled before regarding the theoratical minimum and maximum of the scores before
data_patients =  pd.read_csv("data/ppmi_bl.csv") # A file containing outcomes, covariates and random effects and timeshift of all patients in wide data format. One row per patient.
params = pd.read_csv("data/ppmi_ltjmm_params.csv").set_index("parameter")["value"].to_dict() # dataset containing all LTJMM parameter estimates, converted to dictionary
timevariable = "Time.since.diag_scaled" # timevariable used for LTJMM fitting

# Calculate LTJMM predictions for VaDER. Based on these predictions, the X-tensor (outcome progression scores) and W-Tensor (non-missing indicator) required for VaDER fitting are calculated in this method.
x_tensor, w_tensor = get_vader_tensors(data_patients=data_patients, timevariable=timevariable, covariates=covariates, outcomes=outcomes, params=params, verbose=False)

# Run VaDER hyperparameter optimization
# Hyperparameter optimization results are saved in the specified folder
optimizer = VADERHyperparametersOptimizer(
            params_factory=ParamsFactory(),
            n_repeats=20, # value used in the original ADNI VaDER paper
            n_proc=6, # number of cores used
            n_sample=360, # number of hyperparameter samples from the grid
            n_consensus=1, # no consensus clustering in hyperparameter optimization
            n_epoch=50, # value used in the original ADNI VaDER paper
            n_splits=2, # value used in the original ADNI VaDER paper
            n_perm=1000, # value used in the original ADNI VaDER paper
            seed=0,
            early_stopping_ratio=0.03, # value used in the original ADNI VaDER paper
            early_stopping_batch_size=8, # value used in the original ADNI VaDER paper
            enable_cv_loss_reports=True, # generates additional output files
            output_folder="vader_output" # directory where VaDER reports are created
        )
optimizer.run(x_tensor, w_tensor)
