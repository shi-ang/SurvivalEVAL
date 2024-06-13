import numpy as np
from SurvivalEVAL import PointEvaluator

np.random.seed(1)
# get some synthetic data
n_data = 1000
n_features = 10
event_times = np.random.uniform(0, 6, size=n_data).round(1)
censoring_times = np.random.lognormal(mean=1, sigma=1, size=n_data).round(1)
event_indicators = event_times < censoring_times
observed_times = np.minimum(event_times, censoring_times)

# get training data
n_train = 800
train_event_times = event_times[:n_train]
train_event_indicators = event_indicators[:n_train]

# get testing data
test_event_times = event_times[n_train:]
test_event_indicators = event_indicators[n_train:]

# get some synthetic predictions
predicted_times = np.random.uniform(0, 6, size=200).round(1)

# initialize the evaluator
evaluator = PointEvaluator(predicted_times, test_event_times, test_event_indicators,
                           train_event_times, train_event_indicators)
print("Successfully initialized the evaluator.")

# calculate the concordance index
cindex, _, _ = evaluator.concordance(pair_method="Margin")
print(f"The concordance index is {cindex}.")

# calculate the MAE
mae = evaluator.mae(method="Pseudo_obs", weighted=True)
print(f"The MAE is {mae}.")

# calculate the rmse
rmse = evaluator.rmse(method="Margin", weighted=False)
print(f"The RMSE is {rmse}.")
