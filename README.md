# SurvivalEVAL

This python-based package contains the most completeness evaluation methods for Survival Algorithms. 
These evaluation metrics can be divided into five aspect
(according to Haider's paper: [Effective Ways to Build and Evaluate Individual Survival Distributions](https://jmlr.org/papers/volume21/18-772/18-772.pdf)): 
Concordance, Brier score, MAE loss, 1-Calibration, and D-Calibration.

## Concordance Index (C-index)
Concordance Index identifies the “comparable” pairs of patients and calculates the percentage of correctly ranked pairs to assess a survival model’s performance. 
Given two predicted survival curves of a paired patients, it compares the predicted median/mean survival times and marks that pair as correct if the model's prediction about who died first matches with the reality. 
There are multiple definition for the concordance index, 
[pycox's](https://github.com/havakv/pycox) C-index compares the predicted probabilities, 
[PySurvival's](https://square.github.io/pysurvival/index.html) compares the risk score. 
However, our c-index definition uses the predicted median/mean survival times as the risk score.  

While every pair of the uncensored patients are “comparable”, only those censored patients whose censoring time is greater than the event time of any uncensored patient are considered comparable to that uncensored patient. 
Thus, C-index ignores censored patients who fail to form “comparable” pairs. 
Please note that C-index does not measure how close a model’s estimated survival time is to the actual survival time. 

## Brier Score (BS) and Integrated Brier Score (IBS)
IBS measures both discrimination and calibration of survival prediction models across several time points. 
It is computed as the integration of the (single-time) Brier score across all the time points. 
A smaller IBS value is preferred over the larger value. 
The Brier score, at a specific time-point, is computed as the mean squared error between the observed event (binary indicator variable) and the predicted event probability at that time-point. 
It is meaningful in the sense that the square root of the Brier score is the distance between the observed and predicted event on the probability scale. 
This python implementation uses IPCW weighting to handle the censored instances. Please refer to [Assessment and Comparison of Prognostic Classification Schemes for Survival Data](https://pubmed.ncbi.nlm.nih.gov/10474158/) for the detail of IPCW weighting.
Please also note that IBS is identical to the [Continuous Ranked Probability Score(CRPS)](https://arxiv.org/abs/1806.08324).

## Mean Absolute Error (MAE) and Mean Squared Error (MSE)
One straightforward metric would be “MAE” –  the absolute difference between the actual and predicted survival times (e.g. median of an ISD).
This requires using the “actual survival time”, which is trivial for uncensored instances, but problematic for censored individuals. 
This python package implemented three different MAE (or MSE) loss metrics for different ways of handling censored instances. 
1. MAE-Uncensored simply discards all the censored individuals and compute the MAE for all the uncensored instances.
2. MAE-Hinge calculates the early prediction error. For a censored instance, if the predicted survival time is smaller than the censored time, then `l1_loss = censor_time - predict_time`. If the predicted survival time is equal or larger than the censored time, then `l1-loss = 0`. 
3. MAE-Margin “de-censors” the censored patients, by using their expected survival time (based on the Kaplan-Meier distribution on the training set). The expected survival time is estimated by the censored time plus the median residual time of the KM curve starting from the censored time.
4. MAE-IPCW-D 
5. MAE-IPCW-T
6. MAE-PO


## One-time Calibration (1-calibration)
Calibration measures the confidence of the model. 
The detailed explanation for the algorithm implementation can be found in [Effective Ways to Build and Evaluate Individual Survival Distributions](https://jmlr.org/papers/volume21/18-772/18-772.pdf) and [A tutorial on calibration measurements and calibration models for clinical prediction models](https://academic.oup.com/jamia/article/27/4/621/5762806).
The output is a p-value of Hosmer-Lemeshow goodness-of-fit test at a target time. 
Models with p-value higher than 0.05 can be considered as well-calibrated model at that time.

## Distribution Calibration (D-calibration)
[Haider et al.](https://jmlr.org/papers/volume21/18-772/18-772.pdf) proposed distribution calibration (D-calibration) test for determining if a model that produces ISDs is meaningful. 
D-calibration splits the time-axis into a fixed number of intervals and compares the actual number of events with the predicted number of events within each interval. 
A well D-calibrated model is the one where the predicted number of events within each time interval is statistically similar to the observed number.
Models with p-value higher than 0.05 can be considered as well-calibrated model across the survival distribution.

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

mae_score = eval.mae(method="Pseudo_obs")

one_cal = eval.one_calibration(target_time=25)

d_cal = eval.d_calibration()

```
See the [Examples](Examples) for more usage examples.


## Expected Deliveries in the Future
1. Time-dependent c-index (by Antolini) 
2. time-specific c-index (AUC)

Please create an issue if you want me to implement any other evaluation metrics.