import numpy as np
from SurvivalEVAL import SurvivalEvaluator

np.random.seed(42)
# get some synthetic data
n_data = 1000
n_features = 10
event_times = np.random.weibull(a=1, size=n_data).round(1)
censoring_times = np.random.lognormal(mean=1, sigma=1, size=n_data).round(1)
event_indicators = event_times < censoring_times
observed_times = np.minimum(event_times, censoring_times)

# get some synthetic predictions
n_times = 100
time_grid = np.linspace(1, 5, n_times)
predictions = np.random.rand(n_data, n_times)
# normalize the predictions to sum to 1, meaning the probability mass function
pmf = predictions / predictions.sum(axis=1)[:, None]
survival_curves = 1 - np.cumsum(pmf, axis=1)

# initialize the evaluator
evaluator = SurvivalEvaluator(survival_curves, time_grid, observed_times, event_indicators)
print("Successfully initialized the evaluator.")

# calculate the concordance index
cindex, _, _ = evaluator.concordance()
print(f"The concordance index is {cindex}.")

# calculate the MAE
mae = evaluator.mae(method="Hinge", weighted=False)
print(f"The MAE is {mae}.")

# calculate the d-calibration
pvalue, _ = evaluator.d_calibration(num_bins=10)
print(f"The p-value of the D-Calibration test is {pvalue}.")

# calculate the ibs
ibs = evaluator.integrated_brier_score(num_points=None, IPCW_weighted=False)
print(f"The IBS is {ibs}.")

# calculate the brier score
brier_score = evaluator.brier_score(target_time=2, IPCW_weighted=False)
print(f"The Brier score at time 2 is {brier_score}.")

# calculate the one calibration
one_cal = evaluator.one_calibration(target_time=2)
print(f"The one calibration at time 2 is {one_cal}.")

# calculate the AUC
auc = evaluator.auc(target_time=2)
print(f"The AUC at time 2 is {auc}.")
