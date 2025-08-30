# test interval censoring evaluation script
import numpy as np
from SurvivalEVAL.Evaluations.BrierScore import brier_score_ic

# generate some random left limit and right limit times for the test data
n_samples = 100
left_limit_times = np.random.uniform(0, 10, n_samples)
right_limit_times = left_limit_times + np.random.uniform(0, 5, n_samples)
x = np.random.binomial(1, 0.5, n_samples)

train_samples = 50
tr_left_limit_times = np.random.uniform(0, 10, train_samples)
tr_right_limit_times = tr_left_limit_times + np.random.uniform(0, 5, train_samples)
x_train = np.random.binomial(1, 0.5, train_samples)
# generate some random predicted survival probabilities for the test data
predicted_probs = np.random.uniform(0, 1, (n_samples, ))

bs = brier_score_ic(predicted_probs, left_limit_times, right_limit_times,
                    train_left_limits=tr_left_limit_times, train_right_limits=tr_right_limit_times,
                    x = x, x_train=x_train,
                    method="Tsouprou-conditional")