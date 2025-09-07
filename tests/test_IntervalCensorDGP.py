import numpy as np
import pandas as pd
from IntervalCensorDGP import interval_censor_DGP_from_synthetic_times

print ('Test convert_right_censor_to_interval_censor')

def show_df(left, right, event_indicators, observed_times):
    df = pd.DataFrame({
        "left": left,
        "right": right,
        "e": event_indicators,
        "t": observed_times
    })
    print (df.head(10))

np.random.seed(42)
# get some synthetic data
n_data = 1000
n_features = 10
event_times = np.random.weibull(a=1, size=n_data).round(1)
censoring_times = np.random.lognormal(mean=1, sigma=1, size=n_data).round(1)
event_indicators = event_times < censoring_times
observed_times = np.minimum(event_times, censoring_times)

print ("Fixed interval, step = 2")
left, right, n_visits = interval_censor_DGP_from_synthetic_times(event_times = event_indicators, censoring_times = censoring_times, method = 'fixed', step = 2.0)
show_df(left, right, event_times, censoring_times)

print ('Possion Interval, rate = 1.0')
left, right, n_visits = interval_censor_DGP_from_synthetic_times(event_times = event_indicators, censoring_times = censoring_times, method = 'poisson', rate = 1.0)
show_df(left, right, event_times, censoring_times)

print ('lognormal Interval, mean, sigma = 0, 1.0')
left, right, n_visits = interval_censor_DGP_from_synthetic_times(event_times = event_indicators, censoring_times = censoring_times, method = 'lognormal', ln_mean = 0.0, ln_sigma = 1.0)
show_df(left, right, event_times, censoring_times)
