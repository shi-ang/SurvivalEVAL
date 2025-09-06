import numpy as np
import pandas as pd
from interval_censor_sythentic_data import convert_right_censor_to_interval_censor 

print ('Test convert_right_censor_to_interval_censor df.head()')

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

print ('Test calibration_slope_interval_censor')

from IntervalCensor import calibration_slope_interval_censor

p_list, obs_list, slope = calibration_slope_interval_censor(
    left, right, predictions_cdf, time_grid,
    ps=(0.1, 0.3, 0.5, 0.7, 0.9),
    quantile_method="Linear",
    through_origin=True
)
print ("slope:", slope)

print ('Test  Survival AUPRC')

from IntervalCensor import survival_auprc_interval

Survival_AUPRC = survival_auprc_interval(
    left, right, predictions_cdf, time_grid, n_quad=256
)

print("Mean Survival-AUPRC (interval):", np.mean(Survival_AUPRC))
