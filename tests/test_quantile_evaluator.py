import numpy as np
from SurvivalEVAL import QuantileRegEvaluator

np.random.seed(42)
# get some synthetic data
n_data = 100
n_features = 10
event_times = np.random.weibull(a=1, size=n_data).round(1)
censoring_times = np.random.lognormal(mean=1, sigma=1, size=n_data).round(1)
event_indicators = event_times < censoring_times
observed_times = np.minimum(event_times, censoring_times)

# 0.1, 0.2, ..., 0.9
quantile_levels = np.linspace(0.1, 0.9, 9)

# generate some random time points as quantile predictions
time_grid = np.random.uniform(0, 6, size=900).reshape(100, 9)
quantile_pred = np.cumsum(time_grid, axis=1)

# initialize the evaluator
evaluator = QuantileRegEvaluator(quantile_pred, quantile_levels, observed_times, event_indicators, predict_time_method="RMST")
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
one_cal = evaluator.one_calibration(target_time=0.6)
print(f"The one calibration at time 2 is {one_cal}.")

# calculate the AUC
auc = evaluator.auc(target_time=0.6)
print(f"The AUC at time 2 is {auc}.")
