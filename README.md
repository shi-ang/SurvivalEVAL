<p align="center">
  <a href="https://github.com/shi-ang/SurvivalEVAL/">
    <img
      alt="SurvivalEVAL logo"
      src="https://raw.githubusercontent.com/shi-ang/SurvivalEVAL/main/logo.png"
      width="755"
      height="200"></a>
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

## Metric Guide

SurvivalEVAL groups metrics by prediction target:

- Point prediction metrics compare predicted event times with observed
  event/censoring times.
- Single-time probability metrics evaluate survival or event probability at one
  target time.
- Survival distribution metrics evaluate the full predicted survival curve.
- Interval-censored metrics evaluate survival curves against observed event
  intervals.

<p align="center">
  <a href="https://github.com/shi-ang/SurvivalEVAL/blob/main/all_metrics.png">
    <img alt="Visualization of the evaluation metrics" src="https://github.com/shi-ang/SurvivalEVAL/blob/main/all_metrics.png"></a>
</p>

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
td_c_index, td_concordant, td_total = evl.concordance_time_dependent(method="Antolini")

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

### Point Prediction Discrimination

These metrics compare predicted survival times with observed event/censoring
times. For survival-curve evaluators, predicted times are derived using the
configured `predict_time_method`: `"Median"`, `"Mean"`, or `"RMST"`.

| Metric Name | Description | Code | Paper Link |
| --- | --- | --- | --- |
| Harrell's C-index | Uses observed-event comparable pairs and checks whether earlier observed events receive higher risk. | `evl.concordance(method="Harrell")` | [Harrell et al.](https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0258(19960229)15:4%3C361::AID-SIM168%3E3.0.CO;2-4) |
| Uno/IPCW C-index | Adds inverse-probability-of-censoring weights to comparable pairs. | `evl.concordance(method="Uno")` or `evl.concordance(method="IPCW")` | [Uno et al.](https://onlinelibrary.wiley.com/doi/10.1002/sim.4154) |
| Truncated C-index | Counts only pairs whose earlier or anchor time is strictly before `tau`. | `evl.concordance(method=..., tau=...)` | [Uno et al.](https://onlinelibrary.wiley.com/doi/10.1002/sim.4154) |
| Margin C-index | Replaces censored times with best-guess survival times before calculating C-index. | `evl.concordance(method="Margin")` | [Kumar et al.](https://www.nature.com/articles/s41598-022-08601-6) |

`"Uno"`, `"IPCW"`, and `"Margin"` require training event times and indicators.
When `tau` is omitted, no concordance truncation is applied.

### Point Prediction Errors And Reliability

MAE, MSE, and RMSE share the same censoring-handling methods. Use
`evl.mae(method=...)`, `evl.mse(method=...)`, or `evl.rmse(method=...)`.

| Metric Name | Description | Code | Paper Link |
| --- | --- | --- | --- |
| Uncensored error | Calculates error on observed-event samples only. | `evl.mae(method="Uncensored")` | N/A |
| Hinge error | Penalizes censored samples only when the prediction is earlier than the censoring time. | `evl.mae(method="Hinge")` | [Shivaswamy et al.](https://ieeexplore.ieee.org/document/4470306/) |
| Margin error | Replaces censored times with KM-based best guesses. | `evl.mae(method="Margin")` | [Haider et al.](https://jmlr.org/papers/v21/18-772.html) |
| IPCW-T error | Uses surrogate event times from later observed events with censoring weights. | `evl.mae(method="IPCW-T")` | [Qi et al.](https://proceedings.mlr.press/v202/qi23b/qi23b.pdf) |
| IPCW-D error | Uses IPCW weights directly on observed-event errors. | `evl.mae(method="IPCW-D")` | [Qi et al.](https://proceedings.mlr.press/v202/qi23b/qi23b.pdf) |
| Pseudo-observation error | De-censors censored observations with pseudo-observed event times. | `evl.mae(method="Pseudo_obs")` | [Qi et al.](https://proceedings.mlr.press/v202/qi23b/qi23b.pdf) |
| MSE | Uses the same methods as MAE with squared errors. | `evl.mse(method=...)` | Same as selected method |
| RMSE | Square root of MSE using the same methods as MAE. | `evl.rmse(method=...)` | Same as selected method |
| Log-rank test | Compares observed event times with predicted event times; weighted variants are available. | `evl.log_rank(weightings=...)` | [Mantel](https://pubmed.ncbi.nlm.nih.gov/5910392/) |

`"Margin"`, `"IPCW-T"`, `"IPCW-D"`, and `"Pseudo_obs"` require training event
times and indicators. Hinge uses training data only when `weighted=True`.

### Single-Time Probability Metrics

These metrics evaluate survival probabilities at a specified target time. Use
`SingleTimeEvaluator` when the model only outputs one survival probability per
patient at one target time.

| Metric Name | Description | Code | Paper Link |
| --- | --- | --- | --- |
| AUC/AUROC | Calculates ROC AUC after excluding samples censored before the target time. | `evl.auc(target_time=...)` or `evl.auroc(target_time=...)` | N/A |
| Plain Brier score | Mean squared error between survival status and predicted survival probability without IPCW. | `evl.brier_score(target_time=..., IPCW_weighted=False)` | [Brier](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOEPIT%3E2.0.CO;2) |
| IPCW Brier score | Brier score with inverse-probability-of-censoring weights. | `evl.brier_score(target_time=..., IPCW_weighted=True)` | [Graf et al.](https://pubmed.ncbi.nlm.nih.gov/10474158/) |
| Uncensored HL calibration | Hosmer-Lemeshow calibration test on uncensored samples. | `evl.one_calibration(target_time=..., method="Uncensored")` | [Hosmer and Lemeshow](https://www.tandfonline.com/doi/abs/10.1080/03610928008827941) |
| DN HL calibration | D'Agostino-Nam extension using Kaplan-Meier estimates for observed probabilities. | `evl.one_calibration(target_time=..., method="DN")` | [D'Agostino and Nam](https://www.sciencedirect.com/science/article/pii/S0169716103230017) |
| Integrated calibration index | Smooth calibration-curve summary at a target time. | `evl.integrated_calibration_index(target_time=...)` | [Austin et al.](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.8570) |

### Survival Distribution Metrics

These metrics evaluate the full predicted survival curve.

| Metric Name | Description | Code | Paper Link |
| --- | --- | --- | --- |
| Antolini's time-dependent C-index | Uses survival-curve risk scores at observed-event anchors. | `evl.concordance_time_dependent(method="Antolini", risks="Survival")` | [Antolini et al.](https://onlinelibrary.wiley.com/doi/10.1002/sim.2427) |
| Gandy-Matcham time-dependent C-index | Uses hazard-rate risk scores for crossing hazards. | `evl.concordance_time_dependent(method="IPCW", risks="Hazard", tau=...)` | [Gandy and Matcham](https://onlinelibrary.wiley.com/doi/full/10.1111/sjos.70000) |
| Plain integrated Brier score | Integrates unweighted Brier scores over a time grid. | `evl.integrated_brier_score(IPCW_weighted=False)` | [Graf et al.](https://pubmed.ncbi.nlm.nih.gov/10474158/) |
| IPCW integrated Brier score | Integrates IPCW Brier scores over a time grid. | `evl.integrated_brier_score(IPCW_weighted=True)` | [Graf et al.](https://pubmed.ncbi.nlm.nih.gov/10474158/) |
| Survival-AUPRC | Scores full survival distributions using area under the precision-recall curve. | `evl.auprc()` | [Avati et al.](https://proceedings.mlr.press/v115/avati20a.html) |
| D-calibration | Tests whether predicted survival probabilities at event times follow a uniform distribution. | `evl.d_calibration()` | [Haider et al.](https://jmlr.org/papers/volume21/18-772/18-772.pdf) |
| K-S D-calibration | Kolmogorov-Smirnov version of D-calibration. | `evl.ksd_calibration()` | [Qi et al.](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27713) |
| KM calibration | Compares the average predicted survival curve with the Kaplan-Meier curve. | `evl.km_calibration()` | [Chapfuwa et al.](https://ieeexplore.ieee.org/abstract/document/9244076/) |
| Cox-Snell residuals | Uses cumulative hazard at the observed time to check goodness of fit. | `evl.residuals(method="CoxSnell")` | [Cox and Snell](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1968.tb00724.x) |
| Modified Cox-Snell residuals | Adds an excess residual for censored subjects. | `evl.residuals(method="Modified CoxSnell-v1")` or `evl.residuals(method="Modified CoxSnell-v2")` | [Collett](https://www.taylorfrancis.com/books/mono/10.1201/b18041/modelling-survival-data-medical-research-david-collett) |
| Martingale residuals | Calculates event indicator minus cumulative hazard. | `evl.residuals(method="Martingale")` | [Barlow and Prentice](https://academic.oup.com/biomet/article-abstract/75/1/65/352141) |
| Deviance residuals | Transforms martingale residuals toward a normal residual scale. | `evl.residuals(method="Deviance")` | [Therneau et al.](https://academic.oup.com/biomet/article-abstract/77/1/147/271076) |

For time-dependent concordance, `risks="Survival"` uses `-S(t | z)` as the risk
score at each event anchor, while `risks="Hazard"` uses estimated hazard rates
directly. `method="IPCW"` requires training event times and indicators. `tau`
keeps event anchors whose observed time is strictly before `tau`; when omitted,
no truncation is applied.

## Interval-Censored Metrics

`IntervalCenEvaluator` evaluates predicted survival curves against interval
endpoints. It supports exact events, left-censored observations, interval
censoring, and right censoring within one interface.

### Interval-Censored Discrimination

| Metric Name | Description | Code | Paper Link |
| --- | --- | --- | --- |
| Comparable-pair interval C-index | Counts interval-censored pairs that are comparable from observed endpoints. | `evl.concordance(method="comparable")` | [Qi et al.](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27713) |
| Probability-weighted interval C-index | Uses Turnbull-estimated pair weights for interval-censored comparable ordering. | `evl.concordance(method="probability")` | [Turnbull](https://www.jstor.org/stable/2285518) |
| Midpoint-imputed C-index | Converts finite intervals to midpoint event times and right-censored intervals to censoring times. | `evl.concordance(method="midpoint")` | N/A |
| Interval Survival-AUPRC | Extends Survival-AUPRC scoring to interval-censored outcomes. | `evl.auprc()` | [Avati et al.](https://proceedings.mlr.press/v115/avati20a.html) |

### Interval-Censored Error And Scoring Metrics

| Metric Name | Description | Code | Paper Link |
| --- | --- | --- | --- |
| Uncensored interval Brier score | Excludes samples whose event status is ambiguous at the target time. | `evl.brier_score(target_time=..., method="uncensored")` | [Brier](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOEPIT%3E2.0.CO;2) |
| Tsouprou marginal Brier score | Uses marginal interval-censored survival estimates from the training intervals. | `evl.brier_score(target_time=..., method="Tsouprou-marginal")` | [Tsouprou](https://studenttheses.universiteitleiden.nl/access/item%3A3597164/view) |
| Tsouprou conditional Brier score | Uses a conditional Weibull AFT model with `x` and `x_train` covariates. | `evl.brier_score(target_time=..., method="Tsouprou-conditional")` | [Tsouprou](https://studenttheses.universiteitleiden.nl/access/item%3A3597164/view) |
| Multiple-time interval Brier score | Evaluates interval Brier scores on a supplied time grid. | `evl.brier_score_multiple_points(target_times=..., method=...)` | Same as selected Brier method |
| Integrated interval Brier score | Integrates interval Brier scores over a time grid. | `evl.integrated_brier_score(target_times=..., method=...)` | Same as selected Brier method |
| CRPS | Integrated unweighted interval Brier score, using survival CRPS terminology. | `evl.crps(...)` | [Avati et al.](https://proceedings.mlr.press/v115/avati20a.html) |
| Interval MAE | One-sided absolute error that penalizes predictions outside the observed interval. | `evl.mae()` | [Shivaswamy et al.](https://ieeexplore.ieee.org/document/4470306/) |
| Interval MSE | One-sided squared error that penalizes predictions outside the observed interval. | `evl.mse()` | [Shivaswamy et al.](https://ieeexplore.ieee.org/document/4470306/) |
| Interval RMSE | Square root of interval MSE. | `evl.rmse()` | [Shivaswamy et al.](https://ieeexplore.ieee.org/document/4470306/) |
| Inclusion rate | Fraction of point predictions that fall inside observed event intervals. | `evl.inclusion_rate()` | [Avati et al.](https://proceedings.mlr.press/v115/avati20a.html) |
| Interval prediction coverage | Fractional coverage of observed intervals by predicted intervals. | `evl.coverage(cov_level=...)` | [Qi et al.](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27713) |

The `"Tsouprou-conditional"` method requires test and train covariates through
`x` and `x_train`.

### Interval-Censored Calibration

| Metric Name | Description | Code | Paper Link |
| --- | --- | --- | --- |
| Turnbull one-calibration | One-time calibration using Turnbull interval estimates in prediction bins. | `evl.one_calibration(target_time=..., method="Turnbull")` | [Turnbull](https://www.jstor.org/stable/2285518) |
| Midpoint one-calibration | One-time calibration after midpoint imputation of observed intervals. | `evl.one_calibration(target_time=..., method="MidPoint")` | N/A |
| Interval D-calibration | D-calibration using probability intervals instead of exact event probabilities. | `evl.d_calibration()` | [Haider et al.](https://jmlr.org/papers/volume21/18-772/18-772.pdf) |
| Interval K-S D-calibration | Kolmogorov-Smirnov D-calibration for interval-censored outcomes. | `evl.ksd_calibration()` | [Qi et al.](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27713) |

## Other Evaluators And Helper APIs

| API | Description | Typical Code |
| --- | --- | --- |
| `PointEvaluator` | Evaluates already-computed point survival-time predictions with concordance, MAE, MSE, RMSE, and log-rank tests. | `PointEvaluator(predicted_times, event_times, event_indicators, ...)` |
| `SingleTimeEvaluator` | Evaluates already-computed survival probabilities at one target time with AUC, Brier score, one-calibration, and ICI. | `SingleTimeEvaluator(predicted_probs, event_times, event_indicators, ...)` |
| `QuantileRegEvaluator` | Converts event-time quantile predictions into survival curves and reuses right-censored survival-curve metrics. | `QuantileRegEvaluator(predicted_quantiles, quantile_levels, ...)` |
| Lower-level functions | Most evaluator methods are also available under `SurvivalEVAL.Evaluations` for advanced use cases. | `SurvivalEVAL.Evaluations.concordance(...)` |

`SurvivalEVAL.Evaluations.OtherMetrics` includes research helpers such as
calibration slope and coefficient of variation.

## Nonparametric Estimators

The `SurvivalEVAL.NonparametricEstimator.SingleEvent` module includes:

| Method | Description | Code | Paper Link |
| --- | --- | --- | --- |
| Kaplan-Meier | Nonparametric estimator of the marginal survival function. | `SingleEvent.KaplanMeier(...)` | [Kaplan and Meier](https://www.jstor.org/stable/2281868) |
| Kaplan-Meier area | Kaplan-Meier estimator with area/best-guess utilities for censored survival times. | `SingleEvent.KaplanMeierArea(...)` | [Kaplan and Meier](https://www.jstor.org/stable/2281868) |
| Nelson-Aalen | Nonparametric estimator of the cumulative hazard function. | `SingleEvent.NelsonAalen(...)` | [Nelson](https://www.jstor.org/stable/2958850) |
| Copula Graphic | Estimates survival under dependent censoring with a specified copula. | `SingleEvent.CopulaGraphic(...)` | [Emura and Chen](https://link.springer.com/book/10.1007/978-981-10-7164-5) |
| Turnbull | EM estimator for interval-censored survival data. | `SingleEvent.TurnbullEstimator(...)` | [Turnbull](https://www.jstor.org/stable/2285518) |
| Turnbull lifelines adapter | Lifelines-backed Turnbull estimator wrapper. | `SingleEvent.TurnbullEstimatorLifelines(...)` | [Turnbull](https://www.jstor.org/stable/2285518) |
| Fiducial interval-censoring fitter | Fiducial estimator for interval-censored CDF samples and summaries. | `SurvivalEVAL.NonparametricEstimator.SingleEvent.Fiducial.fit_fiducial_interval_censor(...)` | N/A |

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
