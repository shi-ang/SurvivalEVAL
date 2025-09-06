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

print ("Test calibration_points_right_censor")

from IntervalCensor import calibration_slope_right_censor
p_list = (0.1,0.3,0.5,0.7,0.9)
p_list, obsevation_list, slope = calibration_slope_right_censor(
    event_indicators=event_indicators,     # bool
    observed_times=observed_times,    # float
    predictions=predictions_cdf,      # (N,T), CDF
    time_grid=time_grid,
    ps=p_list,
)
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

p_list = (0.1,0.3,0.5,0.7,0.9)
p_list, obsevation_list, slope = calibration_slope_right_censor(
    event_indicators=event_indicators,     # bool
    observed_times=observed_times,    # float
    predictions=predictions_cdf,      # (N,T), CDF
    time_grid=time_grid,
    ps=p_list,
)
print("slope:", slope)
print("p_list:", p_list)
print("obsevation_list:", obsevation_list)

print ("Test CoV")

from IntervalCensor import cov_from_cdf_grid

cov_list = cov_from_cdf_grid(cdf = predictions_cdf, t_grid = time_grid)
print ('CoV:', np.nanmean(cov_list))

print ("Test survival-auprc right censor")

from IntervalCensor import survival_auprc_right

scores = survival_auprc_right(event_indicators = event_indicators,
                              observed_times = observed_times,
                              predictions_cdf = predictions_cdf, 
                              time_grid = time_grid)
print (np.mean(scores))