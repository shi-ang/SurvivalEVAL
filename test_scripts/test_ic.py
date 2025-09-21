# test interval censoring evaluation script
import numpy as np
from SurvivalEVAL import IntervalCenEvaluator
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
# generate some random points
logits = np.random.weibull(a=1, size=2000).reshape(100, 20)
# geenrate pdf using softmax
pdf = np.exp(logits) / np.exp(logits).sum(axis=1)[:, None]
surv = 1 - np.cumsum(pdf, axis=1)
surv = surv[:, :10]

# generate some random time points
time_points = np.random.uniform(0, 6, size=1000).reshape(100, 10)
time_points = np.cumsum(time_points, axis=1)
time_points = time_points[0, :]
target_time = np.quantile(time_points, 0.5)

# create an evaluator instance
evaluator = IntervalCenEvaluator(
    pred_survs=surv,
    time_coordinates=time_points,
    left_limits=left_limit_times,
    right_limits=right_limit_times,
    train_left_limits=tr_left_limit_times,
    train_right_limits=tr_right_limit_times,
    predict_time_method="Median",
    interpolation="Linear"
)
bs = evaluator.brier_score(method="Tsouprou-marginal", x=x, x_train=x_train)
one_cal = evaluator.one_calibration(target_time=target_time)
d_cal = evaluator.d_calibration()

# bs2 = brier_score_ic(predicted_probs, left_limit_times, right_limit_times,
#                     train_left_limits=tr_left_limit_times, train_right_limits=tr_right_limit_times,
#                     x = x, x_train=x_train,
#                     method="Tsouprou-conditional")