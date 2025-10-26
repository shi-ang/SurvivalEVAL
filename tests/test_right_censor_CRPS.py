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
predictions_cdf = np.cumsum(pmf, axis=1)
evaluator = SurvivalEvaluator(survival_curves, time_grid, observed_times, event_indicators, predict_time_method="Mean")
print("Successfully initialized the evaluator.")

print ("Test calibration_points_right_censor")

p_list, obsevation_list, slope = evaluator.calibration_slope_right_censor()
print("slope:", slope)
print("p_list:", p_list)
print("obsevation_list:", obsevation_list)

print ("Test Well calibrated Synthetic Data")

import numpy as np

rng = np.random.default_rng(42)
n_data = 1000
n_features = 10
X = rng.normal(size=(n_data, n_features))
beta = rng.normal(scale=0.3, size=n_features)    
shape_k = 1.3                                    
b0 = 0.2
lambda_i = np.exp(b0 + X @ beta)
U = rng.uniform(size=n_data)
event_times = lambda_i * (-np.log(U))**(1.0/shape_k)
censoring_times = rng.lognormal(mean=1.0, sigma=1.0, size=n_data)
event_indicators = event_times <= censoring_times  
observed_times   = np.minimum(event_times, censoring_times)

t_max = np.percentile(event_times, 99.5) * 1.5
n_times = 200
time_grid = np.linspace(0.0, t_max, n_times)
tt = time_grid[None, :]                 
ratio_k = (tt / lambda_i[:, None])**shape_k
predictions_cdf = 1.0 - np.exp(-ratio_k)            
predictions_surv = 1.0 - predictions_cdf            
print (predictions_surv.shape)
print (time_grid.shape)
evaluator = SurvivalEvaluator(predictions_surv, time_grid, observed_times, event_indicators, predict_time_method="Mean")

p_list = [0.1, 0.3, 0.5, 0.7, 0.9]
p_list, obsevation_list, slope = evaluator.calibration_slope_right_censor(ps=p_list)
print("slope:", slope)
print("p_list:", p_list)
print("obsevation_list:", obsevation_list)

print ("Test CoV")

# from IntervalCensor import cov_from_cdf_grid

cov_list = evaluator.cov_from_cdf_grid()
print("Mean coverage from evaluator:", np.mean(cov_list))

print ("Test survival-auprc right censor")

Survival_AUPRC = evaluator.auprc(n_quad=256)
print("Mean Survival-AUPRC (interval) from evaluator:", np.mean(Survival_AUPRC))