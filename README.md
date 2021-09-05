# SurvivalEVAL

This python-based package contains the most completeness evaluation methods for Survival Algorithms. 
These evaluation metrics can be divided into five aspect
(according to Haider's paper: [Effective Ways to Build and Evaluate Individual Survival Distributions](https://jmlr.org/papers/volume21/18-772/18-772.pdf)): 
Concordance, Brier score, L1 loss, 1-Calibration, and D-Calibration. Please also refer to [Haider's repository](https://github.com/haiderstats/ISDEvaluation) for the original R-based implementation.

## Concordance Index (C-index)
Concordance Index identifies the “comparable” pairs of patients and calculates the percentage of correctly ranked pairs to assess a survival model’s performance. 
Given two predicted survival curves of a paired patients, it compares the predicted median survival times and marks that pair as correct if the model's prediction about who died first matches with the reality. 
Please note that there are multiple definition for the concordance index, . 

While every pair of the uncensored patients are “comparable”, only those censored patients whose censoring time is greater than the event time of any uncensored patient are considered comparable to that uncensored patient. 
Thus, C-index ignores censored patients who fail to form “comparable” pairs. 
Please note that C-index does not measure how close a model’s estimated survival time is to the actual survival time. 

## Brier Score (BS) and Integrated Brier Score (IBS)
IBS measures both discrimination and calibration of survival prediction models across several time points. 
It is computed as the integration of the (single-time) Brier score across all time points and a smaller IBS value is preferred over the larger value. 
The Brier score, at a specific time-point, is computed as the mean squared error between the observed event (binary indicator variable) and the predicted event probability at that time-point. 
It is meaningful in the sense that the square root of the Brier score is the distance between the observed and predicted event on the probability scale. 
This python implementation uses IPCW weighting to handle the censored instances. Please refer to [Assessment and Comparison of Prognostic Classification Schemes for Survival Data](https://pubmed.ncbi.nlm.nih.gov/10474158/) for the detail of IPCW weighting.
Please also note that IBS is identical to the [Continuous Ranked Probability Score(CRPS)](https://arxiv.org/abs/1806.08324).

## L1 loss
An obvious metric would be “L1 loss” –  the absolute difference between the actual and predicted survival times (e.g. median of an ISD).
This requires using the “actual survival time”, which is trivial for uncensored instances, but problematic for censored individuals. 
This python package implemented three different l1 loss metrics for different ways of handling censored instances. 
1. L1-uncensored loss simply discard all the censored individuals and compute the l1 loss for all the uncensored instances.
2. L1-hinge loss calculates the censored l1-loss using the following manner: If the predicted survival time is smaller than the censored time, then `l1_loss = censor_time - predict_time`. If the predicted survival time is equal or larger than the censored time, then `l1-loss = 0`. 
3. L1-margin loss “de-censors” the censored patients, by using their expected survival time (based on the Kaplan-Meier distribution on the training set). The expected survival time is estimated by the censored time plus the median residual time of the KM curve starting from the censored time.

## One-time Calibration (1-calibration)
Additionally we have the observed and expected probabilities (see Figure 7 in Haider et al.). Note there is an error in the text, it claims the plotted values are Oj and njpj -- it is actually Oj/nj and pj.


## Distribution Calibration (D-calibration)
[Haider et al.](https://jmlr.org/papers/volume21/18-772/18-772.pdf) proposed distribution calibration (D-calibration) test for determining if a model that produces ISDs is meaningful. 
D-calibration splits the time-axis into a fixed number of intervals and compares the actual number of events with the predicted number of events within each interval. 
A well D-calibrated model is the one where the predicted number of events within each time interval is statistically similar to the observed number.

D-calibration quantifies this comparison of predicted and actual events within each time interval. 
The details of D-calibration calculations and ways to incorporate censored instances into D-calibration computation appear in Appendix B and in [Effective Ways to Build and Evaluate Individual Survival Distributions](https://jmlr.org/papers/volume21/18-772/18-772.pdf).

## Quickstart Example
```python
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

from Evaluator import LifelinesEvaluator

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

# Make the evaluation
eval = LifelinesEvaluator(survival_curves, test_event_times, test_event_indicators,
                          train_event_times, train_event_indicators)

cindex, _, _ = eval.concordance()

# The largest event time is 52. So we use 53 time points (0, 1, ..., 52) to calculate the IBS
ibs = eval.integrated_brier_score(num_points=53, draw_figure=True)

l1 = eval.l1_loss(method="Margin")

one_cal = eval.one_calibration(target_time=25)

d_cal = eval.d_calibration()

```
See the [Examples](Examples) for more usage examples.


## Expected Deliveries in the Future
1. Time-dependent concordance index (AUC)
2. Graphical calibration curves
3. IPCW weighted L1 loss
4. Expand l1 loss to l2 loss.
