# CHANGELOG

## 2026-06-17: Version 0.8.0

1. Add Antolini-style time-dependent concordance through the lower-level
   `concordance_time_dependent` function and `SurvivalEvaluator.concordance_time_dependent`.
2. Add survival-probability and hazard-rate risk modes for time-dependent concordance, including the
   Gandy-Matcham hazard-rate IPCW usage for crossing-hazards models.
3. Add IPCW weighting, strict before-`tau` anchor filtering, training-data validation, and tie handling for
   time-dependent concordance.
4. Add hazard prediction support and consistent target-time validation for evaluator probability and hazard lookups.
5. Add regression tests for time-dependent pair counting, IPCW weighting, tie policies, `tau` filtering,
   input validation, and evaluator end-to-end behavior.
6. Reconstruct the README metric reference with paper-linked tables covering the documented right-censored,
   interval-censored, helper, and estimator APIs.

## 2026-06-16: Version 0.7.0

1. Add Uno's right-censored concordance index through the `"Uno"` method, with `"IPCW"` kept as an alias.
2. Add `tau` truncation support for right-censored concordance methods, using strict before-`tau` anchor filtering.
3. Improve right-censored concordance pair accounting for Harrell/Naive, Uno/IPCW, and Margin methods, including
   tie handling and final-block IPCW edge cases.
4. Update evaluator concordance wrappers and docstrings to expose the new right-censored concordance options.
5. Add regression tests for Uno/IPCW weighting, `tau` truncation, tie policies, Margin behavior, and evaluator
   forwarding.

## 2026-06-11: Version 0.6.3

1. Fix zero-padding and prediction-input update paths so 1-D survival curves, raw replacements after padding, sample-specific time grids, and row-count mismatches are handled consistently in right- and interval-censored evaluators.
2. Make string option handling case-insensitive across evaluator methods, interpolation choices, Brier score, concordance, calibration, residual, mean-error, and nonparametric estimator settings.
3. Correct boundary handling for censored observations at target times in AUC and right-censored Brier score calculations, and for open-left/closed-right interval-censored Brier score cases.
4. Improve utility behavior for repeated infinite monotonic values, degenerate all-one survival-to-quantile curves, zero-padded probability prediction, and vectorized tail extrapolation for multiple target times.
5. Clean up evaluator prediction input APIs by adding shared `set_prediction_inputs`, refreshing dimension metadata and cached predictions on updates, and accepting list quantile levels.
6. Correct setup license metadata to GPLv3 and align documentation/comments for evaluator, interval-censored, Brier score, and utility behavior.
7. Add regression tests covering the new validation, boundary, zero-padding, quantile, utility, and metric behavior.

## 2026-06-08: Version 0.6.2

1. Improve survival-curve and time-coordinate validation, broadcasting, zero-padding, and monotonicity correction, including isotonic regression support.
2. Fix calibration edge cases for probability-one predictions, uncensored bin sizes, interval-specific limits, and arbitrary D-calibration bin counts.
3. Fix concordance, Brier score, RMST, coefficient of variation, and survival AUPRC behavior for interval-censored data and boundary cases.
4. Correct nonparametric estimator baselines and exact-event handling for Kaplan-Meier, Nelson-Aalen, Copula Graphic, Turnbull, and Fiducial estimators.
5. Improve evaluator input validation and defaults for prediction intervals, missing training data, duplicated time coordinates, and scikit-survival probability curves.
6. Clarify event-indicator terminology, API documentation, and code comments throughout the package.
7. Add regression tests for the corrected metrics, evaluators, utility functions, and nonparametric estimators.

## 2026-01-22: Version 0.6.1

1. Add a version of Concordance index for interval censoring based on comparable pairs only, just like Harrell's C for right censoring.
2. Added Fiducial estimator for interval censored data. This haven't been integrated into the Evaluator class yet.
3. Bug fixes for Distribution Calibration (for interval censoring), and coverage (for interval censoring).

## 2025-11-04: Version 0.6.0

1. Add IntervalCenEvaluator.py for interval censoring evaluation. The new features include:
   - Concordance index for interval censoring
   - Brier score (three versions) for interval censoring
   - Integrated Brier score (three versions) for interval censoring
   - Calibration metrics (1-calibration, d-calibration) for interval censoring
   - MAE/MSE/RMSE for interval censoring
   - Coverage for interval censoring
   - Inclusion rate for interval censoring
   - Newly proposed KSD-calibration for both right and interval censoring
   - Survival AUPRC for both right and interval censoring
   - CRPS (degenerated version of interval Brier score) for both right and interval censoring
2. Implement other metrics (from literature) that are currently not integrated in the Evaluator classes, including:
   - Coefficient of variation (CoV)
   - Calibration slope (for both right and interval censoring)
3. Add converter to use the mid-point imputation for interval censored data to convert to right censored data for evaluation purpose.
4. Add plot support for calibration metrics, integrated Brier score, for both right and interval censoring.
5. Remove the reference of old Turnbull estimator, implement a new class TurnbullEstimatorLifelines based on the lifelines package.
6. Update test scripts for the new features.

## 2025-09-09: Version 0.5.1

1. Fix bugs for quantile regression evaluator

## 2025-09-08: Version 0.5.0

1. Add interval censoring support for the evaluation.
2. Add log rank test for point evaluation.

## 2025-09-08: Version 0.4.8

1. Downgrade the dependencies in setup.py, requirements.txt to improve compatibility with older versions of other packages.

## 2025-08-30: Version 0.4.7

1. Update dependencies in setup.py, requirements.txt
2. Generate pyproject.toml for the package
3. Minor function rename
4. Adding more interval censoring metrics, and testing them

## 2025-08-13: Version 0.4.6

1. Add warning message for AUROC (when all positive or all negative labels).
2. Add Turnbull estimator.
3. Format the comment and docstring for the package.
4. Add d-calibration and 1-calibration for interval censoring

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

## 2024-04-28

Add the non-IPCW-weighted Brier score and IBS.

## 2024-03-13

Remove the `rpy2` dependency. Now the package is pure python.
