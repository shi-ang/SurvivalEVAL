import numpy as np
import pandas as pd
from IntervalCensorDGP import convert_right_censor_to_interval_censor 

print ('Test convert_right_censor_to_interval_censor df.head()')

np.random.seed(42)
# get some synthetic data
n_data = 1000
n_features = 10
event_times = np.random.weibull(a=1, size=n_data).round(1)
censoring_times = np.random.lognormal(mean=1, sigma=1, size=n_data).round(1)
event_indicators = event_times < censoring_times
observed_times = np.minimum(event_times, censoring_times)
predicted_times = np.random.uniform(0, 5, size=n_data).round(1)

# get some synthetic predictions
n_times = 100
time_grid = np.linspace(1, 5, n_times)
predictions = np.random.rand(n_data, n_times)
# normalize the predictions to sum to 1, meaning the probability mass function
pmf = predictions / predictions.sum(axis=1)[:, None]
survival_curves = 1 - np.cumsum(pmf, axis=1)
predictions_cdf = np.cumsum(pmf, axis=1)

left, right = convert_right_censor_to_interval_censor(
    event_indicators=event_indicators,
    observed_times=observed_times,
)

df = pd.DataFrame({
    "left": left,
    "right": right,
    "e": event_indicators,
    "t": observed_times
})

print (df.head())

from SurvivalEVAL import IntervalCenEvaluator

evaluator = IntervalCenEvaluator(pred_survs = survival_curves, 
                                 time_coordinates = time_grid, 
                                 left_limits = left,
                                 right_limits = right)

print("Successfully initialized the evaluator.")

print ('Test  Survival AUPRC')

Survival_AUPRC = evaluator.survival_auprc_interval(n_quad=256)
print("Mean Survival-AUPRC (interval) from evaluator:", np.mean(Survival_AUPRC))

print ('Test calibration_slope_interval_censor')

p_arr, o_arr, slope = evaluator.calibration_slope_interval_censor(ps = (0.1, 0.3, 0.5, 0.7, 0.9))
print ("interval_censor slope:", slope)

print ('Test cov_from_cdf_grid')

cov_list = evaluator.cov_from_cdf_grid()
print("Mean coverage from evaluator:", np.mean(cov_list))

print ('Median survival time in interval consistency')

p_out, d_out = evaluator.median_in_interval_from_point()
print ("p_out:", p_out)
print ("d_out:", d_out)