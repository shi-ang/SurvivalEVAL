from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

import numpy as np
from SurvivalEVAL.Evaluator import LifelinesEvaluator

np.random.seed(42)
# Load the data
rossi = load_rossi()
rossi = rossi.sample(frac=1.0)

# Split train/test set
train = rossi.iloc[:300, :]
test = rossi.iloc[300:, :]
train_event_times = train.week.values
train_event_indicators = train.arrest.values
test_event_times = test.week.values
test_event_indicators = test.arrest.values

# Fit the model
cph = CoxPHFitter()
cph.fit(train, duration_col='week', event_col='arrest')

survival_curves = cph.predict_survival_function(test)

eval = LifelinesEvaluator(survival_curves, test_event_times, test_event_indicators,
                          train_event_times, train_event_indicators, predict_time_method="Median")

# Make the evaluation
cindex, _, _ = eval.concordance(method="Margin")

bs = eval.brier_score(IPCW_weighted=True)
# The largest event time is 52. So we use 53 time points (0, 1, ..., 52) to calculate the IBS
ibs = eval.integrated_brier_score(num_points=53, IPCW_weighted=False, draw_figure=True)

l1 = eval.mae(method="Margin")

one_cal = eval.one_calibration(target_time=25)

d_cal = eval.d_calibration()
