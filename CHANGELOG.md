# CHANGELOG

## 2025-07-26: Version 0.4.5
1. Bug fixed for Deviance residual
2. Bug fixed for Pchip interpolator 
3. Add log-rank test

## 2025-07-12: Version 0.4.4
1. Drop the use of `numpy.trapz` and `numpy.trapzoid` in the package, use `scipy.integrate.trapzoid` instead.

## 2025-06-24: Version 0.4.3
1. Clean up the deprecated code and files.
2. Improve code efficiency for the evaluation.
3. Add residuals (Cox-Snell, modified Cox-Snell, Martingale, Deviance) for the evaluation.
4. Add NA estimator for the evaluation.
5. Add Copula Graphic estimators
6. Update readme

## 2025-06-23: Version 0.4.2
1. Add graphical calibration plot and integrated calibration index (ICI) for the evaluation.
2. Minor improvements for calculating the median survival time

## 2025-03-05: Version 0.4.1
Add copyright disclaimer for two functions in the `SurvivalEVAL/Evaluation/Concordance.py` file.

## 2025-02-24: Version 0.4.0
1. Add restricted mean survival time (RMST) for the evaluation.
2. Improve computational cost for mean survival time
3. Change the default parameter/behavior for MSE/MAE/RMSE
4. Change the warning message for the IPCW related evaluation

## 2024-09-26: Version 0.3.0
Add the truncated MAE/MSE score. 
The details can be found in this [paper](https://ojs.aaai.org/index.php/AAAI-SS/article/view/27716).

## 2024-07-16: Version 0.2.7
1. Bug fix for the QuantileEvaluator.

## 2024-07-02: Version 0.2.6
1. Add the sanity check for the input survival curve. 
Now the Evaluator will automatically add time zero if the survival curve does not start from time zero.

## 2024-06-20: Version 0.2.5
1. Downgrade many dependencies to improve the compatibility with the older versions of the other packages.
2. Remove the `torchvision` and `torchaudio` dependencies.

## 2024-06-13: Version 0.2.4
Fix the no module found error in the package.

## 2024-06-13: Version 0.2.3
Fix the dependency issues. Remove the `python` version and `jupyter` in the requirements.txt file.

## 2024-06-13: Version 0.2.2
Improve the readme files in the distribution package

## 2024-06-13: Version 0.2.1
Format the package to prepare for the official PyPI submission

## 2024-06-12: Version 0.2.0
1. Add the single time evaluator class
2. minor revision for AUC
3. minor changes for Brier score if the training information is not available 
4. remove unnecessary dependencies, keep the main package as light as possible

## 2024-04-28: 
Add the non-IPCW-weighted Brier score and IBS.

## 2024-03-13: 
Remove the `rpy2` dependency. Now the package is pure python.


