<h1 align="center">SurvivalEVAL</h1>

<p align="center">
  <a href="https://github.com/shi-ang/SurvivalEVAL/">
        <img alt="PyPI" src="https://github.com/shi-ang/SurvivalEVAL/blob/main/logo.png" width="300" height="300"></a>
</p>


-----------------

<p align="center">
    <a href="https://pypi.org/project/SurvivalEVAL/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/SurvivalEVAL"></a>
    <a href="https://pypi.org/project/SurvivalEVAL/">
        <img alt="PyPI - Python Version" src="https://img.shields.io/badge/python-3.8+-blue.svg"></a>
    <a href="https://opensource.org/license/gpl-3-0">
        <img alt="License" src="https://img.shields.io/badge/license-GPLv3-blue.svg"></a>
    <a href="https://github.com/shi-ang/SurvivalEVAL/issues">
        <img alt="Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg"></a>
</p>


This python-based package contains the most completeness evaluation methods for Survival Algorithms (see [paper](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27713)). 
These evaluation metrics can be divided into 3 categories:
- For point prediction (first row in the figure)
    - [Discrimination](#discrimination-for-point-prediction)
    - [Errors](#mean-absolute-error-mean-squared-error-and-root-mean-squared-error) 
    - [Reliability](#reliability) (missing for now)
- For single time probability prediction (second row in the figure below)
    - [Discrimination](#discrimination-for-single-time-probability-prediction)
    - [Errors](#errors-between-the-predicted-probability-and-survival-status)
    - [Calibration](#calibration-for-single-time-probability-prediction)
- For survival distribution prediction (third row in the figure below) 
    - [Discrimination](#discrimination-for-survival-distribution-prediction)
    - [Errors](#errors-for-individual-survival-function-and-the-heaviside-step-function)
    - [Calibration](#calibration-for-survival-distribution-prediction)

[//]: # (![Visualization of the evaluation metrics]&#40;all_metrics.png&#41;)

<p align="center">
    <a href="https://github.com/shi-ang/SurvivalEVAL/blob/main/all_metrics.png">
        <img alt="Visualization of the evaluation metrics" src="https://github.com/shi-ang/SurvivalEVAL/blob/main/all_metrics.png"></a>
</p>

## Installation
You can install the package via pip.
```bash
pip install SurvivalEVAL
```

Or if you want to do some modification by yourself. 
Clone the repo, cd into it and install it in editable mode (`-e` option).
That way, these are no more need to re-install the package after modification.
```bash
git clone https://github.com/shi-ang/SurvivalEVAL.git
cd SurvivalEVAL
pip install -r requirements.txt
pip install -e . 
```

## Quickstart Example



Install a survival analysis package, such as `lifelines`, and load the data.
Then, you can use the following code to evaluate the model.

```python
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

from SurvivalEVAL.Evaluator import LifelinesEvaluator

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
evl = LifelinesEvaluator(survival_curves, test_event_times, test_event_indicators,
                          train_event_times, train_event_indicators)

cindex, _, _ = evl.concordance()

mae_score = evl.mae(method="Pseudo_obs")

mse_score = evl.mse(method="Hinge")

# The largest event time is 52. So we use 53 time points (0, 1, ..., 52) to calculate the IBS
ibs = evl.integrated_brier_score(num_points=53, draw_figure=True)

d_cal = evl.d_calibration()

# The target time for the single time probability prediction is set to 25
auc_score = evl.auc(target_time=25)
bs_score = evl.brier_score(target_time=25)
one_cal = evl.one_calibration(target_time=25)

```
See the [Examples](examples) for more usage examples.



## Point Prediction

### Discrimination for point prediction
Concordance index (CI) identifies the “comparable” pairs of patients and calculates the percentage of correctly ranked pairs to assess a survival model’s performance. 
Given two predicted survival curves of a paired patients, it compares the predicted median/mean survival times and marks that pair as correct if the model's prediction about who died first matches with the reality. 

| Metric Name  | Description                                                                                                                                                | Code                                | Paper Link                                                         |
|--------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------|--------------------------------------------------------------------|
| Harrell's CI | Original Harrell's CI, using comparable pair for concordant checking.                                                                                      | `evl.concordance(method="Harrell")` | [Harrell et al.](https://pubmed.ncbi.nlm.nih.gov/8668867/)         |
| Uno's CI     | Uno's CI adds a (squared) IPCW weighting to each comparable pair to the overall calculation.                                                               | To be implemented                   | [Uno et al.](https://pubmed.ncbi.nlm.nih.gov/21484848/)            |
| Margin CI    | Calculate the marginal survival time (and proxy time) to substitute the censoring time for censored patients, then calculate CI as everyone is uncensored. | `eval.concordance(method="Margin")` | [Kumar et al.](https://www.nature.com/articles/s41598-022-08601-6) |


### Mean Absolute Error, Mean Squared Error and Root Mean Squared Error
One straightforward metric would be “MAE” –  the absolute difference between the actual and predicted survival times (e.g. median of a curve).
This requires using the “actual survival time”, which is trivial for uncensored instances, but problematic for censored individuals. 
This python package implemented MAE loss metrics using different ways of handling censored instances. Here we list three of them:
1. `Uncensored` simply discards all the censored individuals and compute the MAE for all the uncensored instances.
2. `Hinge` calculates the early prediction error. For a censored instance, if the predicted survival time is smaller than the censored time, then `MAE = censor_time - predict_time`. If the predicted survival time is equal or larger than the censored time, then `MAE = 0`. Note that the standard `Hinge` method requires the `Weighted` parameter to be set to `False`.
3. `Pseudo_obs` “de-censors” the censored patients, using pseudo-observation method (by estimating the contribution of a censored subject to the whole Kaplan-Meier distribution). Then it calculates the MAE between de-censoring time and the predicted survival time, just like the normal way. Note that the standard `Pseudo_obs` method requires the `Weighted` parameter to be set to `True`.

Mean squared error (MSE) is another metric to measure the difference between the actual and predicted survival times.
Similar to MAE, mean squared error (MSE) also has multiple ways to handle censored instances, similar to MAE.
We also have root mean squared error (RMSE) for each of the different ways.

| Metric Name    | Description                                                                                                                                                                                                                                                                                                                                                                                      | Code                           | Paper Link                                                         |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|--------------------------------------------------------------------|
| MAE-Uncensored | Calculate MAE for uncensored instances only.                                                                                                                                                                                                                                                                                                                                                     | `evl.mae(method="Uncensored")` | N/A                                                                |
| MAE-Hinge      | For censored instances, Hinge calculates the early prediction error. For a censored instance, if the predicted survival time is smaller than the censored time, then `MAE = censor_time - predict_time`. If the predicted survival time is equal or larger than the censored time, then `MAE = 0`. Note that the standard `Hinge` method requires the `Weighted` parameter to be set to `False`. | `evl.mae(method="Hinge")`      | [Shivaswamy et al.](https://ieeexplore.ieee.org/document/4470306/) |
| MAE-Margin     | It "de-censors” the censored patients, using the margin time (mean of the conditional KM curve given the censoring time).                                                                                                                                                                                                                                                                        | `evl.mae(method="Margin")`     | [Haider et al.](https://jmlr.org/papers/v21/18-772.html)           |
| MAE-PO         | It "de-censors” the censored patients, using pseudo-observation method.                                                                                                                                                                                                                                                                                                                          | `evl.mae(method="Pseudo_obs")` | [Qi et al.](https://proceedings.mlr.press/v202/qi23b.html)         |
| MSE            | MSE has the same user-interface with MAE.                                                                                                                                                                                                                                                                                                                                                        | `evl.mse(method=method)`       | N/A                                                                |
| RMSE           | RMSE has the same user-interface with MAE.                                                                                                                                                                                                                                                                                                                                                       | `evl.rmse(method=method)`      | N/A                                                                |


### Reliability
Metrics to be implemented. Including (weighted) log-rank test, etc.


## Single Time Probability Prediction
### Discrimination for single time probability prediction
Area Under the Receiver Operating Characteristic (AUROC) is a metric to measure the performance of a single time probability prediction. 
It is the area under the receiver operating characteristic curve, which is the plot of the true positive rate against the false positive rate at various threshold settings.
In the survival analysis, the single time probability prediction is the prediction of the survival probability at a specific time point.
And the true label is whether the patient died at that time point.
AUROC excludes the censored instances whose censoring time is earlier than the target time point.

| Metric Name | Description                                                              | Code              | Paper Link                                                                   |
|-------------|--------------------------------------------------------------------------|-------------------|------------------------------------------------------------------------------|
| AUC         | Calculate AUC/AUROC for uncensored instances only.                       | `evl.auc()`       | N/A                                                                          |
| Uno AUC     | Uno's CI adds a IPCW weighting to each pairs to the overall calculation. | To be implemented | [Uno et al.](https://www.tandfonline.com/doi/abs/10.1198/016214507000000149) |


### Errors between the predicted probability and survival status
The Brier score (BS), at a specific time-point, is computed as the mean squared error between the observed event (binary indicator variable) and the predicted event probability at that time-point. 
It is meaningful in the sense that the square root of the Brier score is the distance between the observed and predicted event on the probability scale. 

| Metric Name | Description                                                                  | Code                                   | Paper Link                                               |
|-------------|------------------------------------------------------------------------------|----------------------------------------|----------------------------------------------------------|
| Plain BS    | Calculate BS for uncensored instances only.                                  | `evl.brier_score(IPCW_weighted=False)` | N/A                                                      |
| IPCW BS     | Adding a IPCW weighting to uncensored instances after the target time point. | `evl.brier_score(IPCW_weighted=True)`  | [Graf et al.](https://pubmed.ncbi.nlm.nih.gov/10474158/) |


### Calibration for single time probability prediction
Calibration measures the alignment between the predicted survival probability and the observed survival status at a specific time point.
The most common Hosmer-Lemeshow (HL) goodness-of-fit test is used to check the calibration of a model at a specific time point.
Models with p-value higher than 0.05 can be considered as well-calibrated model at that time.

| Metric Name                  | Description                                                                                       | Code                                       | Paper Link                                                                            |
|------------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------|---------------------------------------------------------------------------------------|
| Uncensored HL test           | Standard Hosmer-Lemeshow test for uncensored part of the dataset.                                 | `evl.one_calibration(method="Uncensored")` | [Hosmer&Lemeshow](https://www.tandfonline.com/doi/abs/10.1080/03610928008827941)      |
| DN HL test                   | DN's extension of HL test, using Kaplan-Meier estimation to approximate the observed probability. | `evl.one_calibration(method="DN")`         | [D'Agostino&Nam](https://www.sciencedirect.com/science/article/pii/S0169716103230017) |
| Integrated Calibration Index | Use non-linear model to fit a smooth calibration curve and calculate the integrated error.        | `eval.integrated_calibration_index()`      | [Austin et al.](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.8570)             |


## Survival Distribution Prediction
### Discrimination for survival distribution prediction
Time-dependent concordance index (td-CI) is a generalization of the concordance index to the survival distribution prediction.
It is calculated by integrating the concordance index over times.

| Metric Name | Description                                    | Code              | Paper Link                                                   |
|-------------|------------------------------------------------|-------------------|--------------------------------------------------------------|
| td-CI       | Calculate td-CI for uncensored instances only. | To be implemented | [Antolini et al.](https://pubmed.ncbi.nlm.nih.gov/16320281/) |

### Errors for individual survival function and the heaviside step function
Integrated Brier Score (IBS) measures the squared difference between the predicted survival curve with the Heaviside step function of the observed event.
IBS can be viewed as the integration of the (single-time) Brier score across all the time points. 
A smaller IBS value is preferred over the larger value. 
This python implementation uses IPCW weighting to handle the censored instances. Please refer to [Assessment and Comparison of Prognostic Classification Schemes for Survival Data](https://pubmed.ncbi.nlm.nih.gov/10474158/) for the detail of IPCW weighting.
Please also note that IBS is also similar to the [Continuous Ranked Probability Score (CRPS)](https://arxiv.org/abs/1806.08324), except (1) the IPCW weighting, and (2) squared error instead of absolute error.

| Metric Name | Description                                                                           | Code                                              | Paper Link                                               |
|-------------|---------------------------------------------------------------------------------------|---------------------------------------------------|----------------------------------------------------------|
| Plain IBS   | Calculate IBS for up to the censoring time, for uncensored instances.                 | `evl.integrated_brier_score(IPCW_weighted=False)` | N/A                                                      |
| CRPS        | Calculate integrated absolute error between survival function and heaviside function. | To be implemented                                 | [Avati et al.](https://arxiv.org/abs/1806.08324)         |
| IPCW IBS    | Adding a IPCW weighting to uncensored instances after the target time point.          | `evl.integrated_brier_score(IPCW_weighted=True)`  | [Graf et al.](https://pubmed.ncbi.nlm.nih.gov/10474158/) |


### Calibration for survival distribution prediction
Calibration for the entire survival distribution is a more complex task than the single time calibration.
Some residuals are proposed to check the calibration of the entire survival distribution.
And [Haider et al.](https://jmlr.org/papers/volume21/18-772/18-772.pdf) proposed distribution calibration (D-calibration) test for determining if a model that produces ISDs is meaningful. 
D-calibration splits the time-axis into a fixed number of intervals and compares the actual number of events with the predicted number of events within each interval. 
A well D-calibrated model is the one where the predicted number of events within each time interval is statistically similar to the observed number.
Models with p-value higher than 0.05 can be considered as well-calibrated model across the survival distribution.

| Metric Name                    | Description                                                                                                                        | Code                                           | Paper Link                                                                                                                           |
|--------------------------------|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Cox-Snell Residual             | Calculate the cumulative hazard function at the observed time, and then fit a Nelson-Aalen estimator to check the goodness-of-fit. | `evl.residuals(method="CoxSnell")`             | [Cox&Snell](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1968.tb00724.x)                                          |
| Modified Cox-Snell Residual v1 | Calculate the cumulative hazard function at the observed time, add 1 for censored subjects.                                        | `evl.residuals(method="Modified CoxSnell-v1")` | [Collett, Chapter 4](https://www.taylorfrancis.com/books/mono/10.1201/b18041/modelling-survival-data-medical-research-david-collett) |
| Modified Cox-Snell Residual v2 | Calculate the cumulative hazard function at the observed time, add ln(2) for censored subjects.                                    | `evl.residuals(method="Modified CoxSnell-v2")` | [Crowley&Hu](https://www.tandfonline.com/doi/abs/10.1080/01621459.1977.10479903)                                                     |
| Martingale Residual            | Calculate the Martingale residual, which should have a mean of 0                                                                   | `evl.residuals(method="Martingale")`           | [Barlow&Prentice](https://academic.oup.com/biomet/article-abstract/75/1/65/352141)                                                   |
| Deviance Residual              | Calculate the Deviance residual, which should follow a normal distribution N(0, 1).                                                | `evl.residuals(method="Deviance")`             | [Therneau et al.](https://academic.oup.com/biomet/article-abstract/77/1/147/271076)                                                  |
| Distribution Calibration       | Check whether the survival probability at the event time follows the Uniform distribution betwen 0 and 1.                          | `eval.d_calibration()`                         | [Haider et al.](https://jmlr.org/papers/volume21/18-772/18-772.pdf)                                                                  |
| KM Calibration                 | Check whether the averaged predicted survival distribution is aligned with KM estimator.                                           | `eval.km_calibration()`                        | [Chapfuwa et al.](https://ieeexplore.ieee.org/abstract/document/9244076/)                                                                                                                  |


## Non-Parametric Methods
This package also provides some non-parametric methods to estimate the marginal survival function, 
without considering the covariates.

The non-parametric methods are included in the `SurvivalEVAL.NonparametricEstimator` module.

| Method         | Description                                                                                                                                         | Code                             | Paper Link                                                             |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|------------------------------------------------------------------------|
| Kaplan Meier   | The well known Kaplan Meier (KM) estimator that directly estimate the survival function.                                                            | `SingleEvent.KaplanMeier()`      | [Kaplan&Meier](https://www.jstor.org/stable/2281868)                   |
| Nelson Aalan   | Nelson Aalan (NA) estimate the cumulative hazard function directly, then transform to survival function.                                            | `SingleEvent.NelsonAalen()`      | [Aalan](https://www.jstor.org/stable/2958850)                          |
| Copula Graphic | The Copula Graphic (CG) estimator estimate the survival function under the dependent censoring assumption, with known type of copula and parameter. | `SingleEvent.CopulaGraphic()`    | [Emura&Chen](https://link.springer.com/book/10.1007/978-981-10-7164-5) |
| Turnbull       | The Turnbull estimater estimate the survival function for interval censoring dataset.                                                               | To be implemented                | [Turnbull](https://www.jstor.org/stable/2285518)                       |
| Aalan Johansen | The Aalan Johansen (AJ) estimator the cumulative incidence function (CIF) for competing risk dataset.                                               | To be implemented                | [Aalan&Johansen](https://www.jstor.org/stable/4615704)                                                     |


## Citing this work

We recommend you use the following to cite `SurvivalEVAL` in your publications:

```
@article{qi2024survivaleval,
year = {2024},
month = {01},
pages = {453-457},
title = {{SurvivalEVAL}: A Comprehensive Open-Source Python Package for Evaluating Individual Survival Distributions},
author={Qi, Shi-ang and Sun, Weijie and Greiner, Russell},
volume = {2},
journal = {Proceedings of the AAAI Symposium Series},
doi = {10.1609/aaaiss.v2i1.27713}
}
```