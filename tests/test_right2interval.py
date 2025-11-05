import numpy as np
import pandas as pd
from IntervalCensorDGP import right_censor_to_interval
from matplotlib import pyplot as plt

from SurvivalEVAL import IntervalCenEvaluator
from SurvivalEVAL.Evaluations.util import check_monotonicity

print("Test convert_right_censor_to_interval_censor")

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
# clip the survival curves to be between 0 and 1
survival_curves = np.clip(survival_curves, 0.0, 1.0)
print("Monotonicity of survival curves:", check_monotonicity(survival_curves))

left, right = right_censor_to_interval(
    event_indicators=event_indicators,
    observed_times=observed_times,
)

df = pd.DataFrame(
    {"left": left, "right": right, "e": event_indicators, "t": observed_times}
)

print(df.head())


evaluator = IntervalCenEvaluator(
    pred_survs=survival_curves,
    time_coordinates=time_grid,
    left_limits=left,
    right_limits=right,
)

print("Successfully initialized the evaluator.")

print("Test  Survival AUPRC")

Survival_AUPRC = evaluator.auprc(n_quad=256)
print("Mean Survival-AUPRC (interval) from evaluator:", np.mean(Survival_AUPRC))

print("Test calibration_slope_interval_censor")

# p_value, details = evaluator.d_calibration(return_details=True)
# print ("interval_censor d_calibration p-value:", p_value)
# print ("Details:", details)
# plt.show()

# p_value, details = evaluator.one_calibration(target_time=2.5, return_details=True)
# print ("interval_censor one_calibration p-value at time 2.5:", p_value)
# print ("Details:", details)
# plt.show()

i_r = evaluator.inclusion_rate()
print("Inclusion Rate in interval censoring:", i_r)

mae = evaluator.mae()
print("Mean Absolute Error in interval censoring:", mae)

coverage, cov_gap, avg_width = evaluator.coverage(cov_level=0.8, method="linear")
print("Coverage at 80% level in interval censoring:", coverage)
print("Coverage gap at 80% level in interval censoring:", cov_gap)
print(
    "Average width of prediction intervals at 80% level in interval censoring:",
    avg_width,
)
