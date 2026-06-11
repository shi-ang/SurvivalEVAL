<p align="center">
  <a href="https://github.com/shi-ang/SurvivalEVAL/">
    <img
      alt="SurvivalEVAL logo"
      src="https://raw.githubusercontent.com/shi-ang/SurvivalEVAL/main/logo.png"
      width="300"
      height="300"></a>
</p>

-----------------

<p align="center">
  <a href="https://pypi.org/project/SurvivalEVAL/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/SurvivalEVAL"></a>
  <a href="https://pypi.org/project/SurvivalEVAL/">
    <img alt="Python Version" src="https://img.shields.io/badge/python-3.9%2B-blue.svg"></a>
  <a href="https://opensource.org/license/gpl-3-0">
    <img alt="License" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
  <a href="https://github.com/shi-ang/SurvivalEVAL/issues">
    <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg"></a>
</p>

SurvivalEVAL is a Python 3.9+ package for evaluating survival analysis predictions.
It supports right-censored and interval-censored outcomes, predicted survival
curves, point predictions, single-time probabilities, and quantile-regression
outputs.

The package is designed around evaluator classes:

- `SurvivalEvaluator`: general right-censored survival-curve predictions.
- `LifelinesEvaluator`, `PycoxEvaluator`, and `ScikitSurvivalEvaluator`: adapters
  for common model-output formats.
- `IntervalCenEvaluator`: interval-censored survival-curve predictions.
- `PointEvaluator`: point survival-time predictions.
- `SingleTimeEvaluator`: probabilities at one target time.
- `QuantileRegEvaluator`: predicted event-time quantiles.

The public tests and examples show typical model integrations. See
[examples](examples) for notebooks covering lifelines, pycox, scikit-survival,
quantile prediction, point prediction, interpolation choices, and monotonicity
handling.

## Installation

Install from PyPI:

```bash
pip install SurvivalEVAL
```

For local development:

```bash
git clone https://github.com/shi-ang/SurvivalEVAL.git
cd SurvivalEVAL
python -m pip install -r requirements.txt
python -m pip install -e .
```

Optional development dependencies are available with:

```bash
python -m pip install -e ".[dev]"
```

## Input Conventions

For right-censored data, `event_indicators` uses:

- `1`: observed event
- `0`: right-censored observation

For interval-censored data, pass finite non-negative `left_limits` and
`right_limits`. Use `right_limits=np.inf` for right-censored observations.
Exact events can be represented with `left_limits == right_limits`, and
left-censored observations can use `left_limits == 0`.

Predicted survival curves are passed as survival probabilities over time:

- `pred_survs`: shape `(n_samples, n_time_points)` or `(n_time_points,)`
- `time_coordinates`: shape `(n_time_points,)` or
  `(n_samples, n_time_points)`

At least one of `pred_survs` or `time_coordinates` must be two-dimensional so
the evaluator can infer the number of testing samples. If the time grid does
not start at zero, SurvivalEVAL prepends time zero and survival probability one.

## Quickstart: Right-Censored Survival Curves

```python
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

from SurvivalEVAL import LifelinesEvaluator

rossi = load_rossi().sample(frac=1.0, random_state=0)
train = rossi.iloc[:300, :]
test = rossi.iloc[300:, :]

cph = CoxPHFitter()
cph.fit(train, duration_col="week", event_col="arrest")

survival_curves = cph.predict_survival_function(test)

evl = LifelinesEvaluator(
    survival_curves,
    test.week.values,
    test.arrest.values,
    train.week.values,
    train.arrest.values,
)

c_index, concordant, total = evl.concordance(method="Harrell")
mae = evl.mae(method="Pseudo_obs")
ibs = evl.integrated_brier_score(num_points=53)
d_cal_p, d_cal_hist = evl.d_calibration()

auc = evl.auc(target_time=25)
brier = evl.brier_score(target_time=25)
one_cal_p, observed, expected = evl.one_calibration(target_time=25)
```

## Quickstart: Interval-Censored Survival Curves

```python
import numpy as np

from SurvivalEVAL import IntervalCenEvaluator

time_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
pred_survs = np.array(
    [
        [1.0, 0.82, 0.58, 0.34, 0.12],
        [1.0, 0.76, 0.51, 0.25, 0.08],
        [1.0, 0.90, 0.72, 0.48, 0.22],
    ]
)

left = np.array([0.5, 1.0, 2.5])
right = np.array([1.5, np.inf, 3.5])

train_left = np.array([0.0, 1.0, 1.5, 2.0])
train_right = np.array([1.0, 2.0, np.inf, 3.0])

evl = IntervalCenEvaluator(
    pred_survs,
    time_grid,
    left,
    right,
    train_left_limits=train_left,
    train_right_limits=train_right,
)

c_index, concordant, total = evl.concordance(method="comparable")
brier = evl.brier_score(target_time=2.0, method="uncensored")
ibs = evl.integrated_brier_score(
    target_times=np.array([1.0, 2.0, 3.0]),
    method="uncensored",
)

one_cal_p, observed, expected = evl.one_calibration(
    target_time=2.0,
    method="MidPoint",
)
d_cal_p, d_cal_hist = evl.d_calibration()
coverage, coverage_gap, average_width = evl.coverage(
    cov_level=0.8,
    method="linear",
)
```

## Right-Censored Metrics

Right-censored survival-curve evaluators include `SurvivalEvaluator`,
`LifelinesEvaluator`, `PycoxEvaluator`, `ScikitSurvivalEvaluator`, and
`QuantileRegEvaluator`.

### Point Prediction Metrics

These metrics compare predicted survival times with observed event/censoring
times. Predicted times are derived from survival curves using the configured
`predict_time_method`: `"Median"`, `"Mean"`, or `"RMST"`.

- Concordance index: `evl.concordance(method="Harrell")`
- Margin concordance: `evl.concordance(method="Margin")`
- Mean absolute error: `evl.mae(method=...)`
- Mean squared error: `evl.mse(method=...)`
- Root mean squared error: `evl.rmse(method=...)`
- Log-rank and weighted log-rank tests: `evl.log_rank(...)`

Error methods include `"Uncensored"`, `"Hinge"`, `"Margin"`, `"IPCW-T"`,
`"IPCW-D"`, and `"Pseudo_obs"`.

### Single-Time Probability Metrics

These metrics evaluate survival probabilities at a specified target time:

- AUROC/AUC: `evl.auc(target_time=...)` or `evl.auroc(target_time=...)`
- Brier score: `evl.brier_score(target_time=..., IPCW_weighted=True)`
- Hosmer-Lemeshow calibration: `evl.one_calibration(target_time=...)`
- Integrated calibration index: `evl.integrated_calibration_index(...)`

Use `SingleTimeEvaluator` when the model only outputs one survival probability
per patient at one target time.

### Survival Distribution Metrics

These metrics evaluate the full predicted survival curve:

- Integrated Brier score: `evl.integrated_brier_score(...)`
- Survival-AUPRC: `evl.auprc()`
- D-calibration: `evl.d_calibration()`
- K-S D-calibration: `evl.ksd_calibration()`
- KM calibration: `evl.km_calibration()`
- Cox-Snell, modified Cox-Snell, Martingale, and Deviance residuals:
  `evl.residuals(method=...)`

## Interval-Censored Metrics

`IntervalCenEvaluator` evaluates predicted survival curves against interval
endpoints. It supports exact events, left-censored observations, interval
censoring, and right censoring within one interface.

### Discrimination

- Comparable-pair interval concordance:
  `evl.concordance(method="comparable")`
- Probability-weighted interval concordance using a Turnbull estimator:
  `evl.concordance(method="probability")`
- Midpoint-imputed concordance:
  `evl.concordance(method="midpoint")`
- Survival-AUPRC for interval-censored outcomes: `evl.auprc()`

### Error And Scoring Metrics

- Single-time interval Brier score:
  `evl.brier_score(target_time=..., method=...)`
- Multiple-time interval Brier score:
  `evl.brier_score_multiple_points(target_times=..., method=...)`
- Integrated Brier score:
  `evl.integrated_brier_score(target_times=..., method=...)`
- Continuous Ranked Probability Score:
  `evl.crps(...)`
- Point-prediction MAE/MSE/RMSE:
  `evl.mae()`, `evl.mse()`, and `evl.rmse()`
- Inclusion rate of point predictions in observed intervals:
  `evl.inclusion_rate()`
- Prediction-interval coverage:
  `evl.coverage(cov_level=...)`

Interval Brier score methods include `"uncensored"`, `"Tsouprou-marginal"`,
and `"Tsouprou-conditional"`. The conditional method additionally requires
test and train covariates through `x` and `x_train`.

### Calibration

- One-calibration with interval handling:
  `evl.one_calibration(target_time=..., method="Turnbull")`
- Midpoint one-calibration:
  `evl.one_calibration(target_time=..., method="MidPoint")`
- Interval D-calibration: `evl.d_calibration()`
- Interval K-S D-calibration: `evl.ksd_calibration()`

## Other Evaluators And Helper APIs

- `PointEvaluator` evaluates already-computed point survival-time predictions
  with concordance, MAE, MSE, RMSE, and log-rank tests.
- `SingleTimeEvaluator` evaluates already-computed survival probabilities at a
  single target time with AUC, Brier score, one-calibration, and ICI.
- `QuantileRegEvaluator` converts event-time quantile predictions into survival
  curves and reuses the right-censored survival-curve metrics.
- `SurvivalEVAL.Evaluations.OtherMetrics` includes lower-level research helpers
  such as calibration slope and coefficient of variation.

Most evaluator methods are also available as lower-level functions under
`SurvivalEVAL.Evaluations` for advanced use cases.

## Nonparametric Estimators

The `SurvivalEVAL.NonparametricEstimator.SingleEvent` module includes:

- Kaplan-Meier estimators: `KaplanMeier`, `KaplanMeierArea`
- Nelson-Aalen estimator: `NelsonAalen`
- Copula Graphic estimator: `CopulaGraphic`
- Turnbull estimators: `TurnbullEstimator`, `TurnbullEstimatorLifelines`
- Fiducial interval-censoring fitter:
  `SurvivalEVAL.NonparametricEstimator.SingleEvent.Fiducial.fit_fiducial_interval_censor`

## Citing This Work

We recommend you use the following to cite `SurvivalEVAL` in your publications:

```bibtex
@article{qi2024survivaleval,
year = {2024},
month = {01},
pages = {453-457},
title = {{SurvivalEVAL}: A Comprehensive Open-Source Python Package for Evaluating Individual Survival Distributions},
author = {Qi, Shi-ang and Sun, Weijie and Greiner, Russell},
volume = {2},
journal = {Proceedings of the AAAI Symposium Series},
doi = {10.1609/aaaiss.v2i1.27713}
}
```
