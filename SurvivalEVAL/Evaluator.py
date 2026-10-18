from __future__ import annotations

import warnings
from abc import ABC
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test

from SurvivalEVAL._BaseSurvivalEvaluator import _BaseSurvivalEvaluator
from SurvivalEVAL.Evaluations.AreaUnderPRCurve import auprc_right_censor
from SurvivalEVAL.Evaluations.BrierScore import (
    _brier_time_grid,
    _integrate_brier_scores,
    brier_multiple_points,
)
from SurvivalEVAL.Evaluations.Calibration import (
    d_calibration,
    integrated_calibration_index,
    km_calibration,
    ksd_calibration,
    one_calibration,
    residuals,
)
from SurvivalEVAL.Evaluations.Calibration.SingleTime.util import (
    _calibration_curve_details,
    _maximum_local_deviation,
)
from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.Discrimination import auc, concordance
from SurvivalEVAL.Evaluations.Discrimination._concordance_utils import _normalize_ties
from SurvivalEVAL.Evaluations.Discrimination.Distributional.TimeDependentConcordance import (
    _concordance_time_dependent,
    _normalize_time_dependent_method,
)
from SurvivalEVAL.Evaluations.MeanError import mean_error
from SurvivalEVAL.Evaluations.util import (
    check_and_convert,
    check_and_convert_event_data,
    fit_least_squares,
    quantile_to_survival,
    validate_time_points,
)
from SurvivalEVAL.Evaluations.util_plots import pp_plot


class SurvivalEvaluator(_BaseSurvivalEvaluator):
    def __init__(
        self,
        pred_survs: NumericArrayLike,
        time_coordinates: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ):
        """
        Initialize the Evaluator

        Parameters
        ----------
        pred_survs: NumericArrayLike,
            Accept shapes: (n_time_points,) or (n_samples, n_time_points).
            Predicted survival curves for the testing samples.
            At least one of `pred_survs` or `time_coordinates` must be a 2D array.
        time_coordinates: NumericArrayLike,
            Accept shapes: (n_time_points,) or (n_samples, n_time_points).
            Time coordinates corresponding to the survival curves.
            At least one of `pred_survs` or `time_coordinates` must be a 2D array.
        event_times: NumericArrayLike, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        event_indicators: NumericArrayLike, shape = (n_samples, )
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        train_event_times: NumericArrayLike | None, shape = (n_train_samples, ), default: None
            Actual event/censor time for the training samples.
        train_event_indicators: NumericArrayLike | None, shape = (n_train_samples, ), default: None
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        predict_time_method: str, default = "Median"
            Method for calculating predicted survival time. Available options are "Median", "Mean" and "RMST".
        interpolation: str, default = "Linear"
            Method for interpolation. Available options are ['Linear', 'Pchip'].
        """
        event_times, event_indicators = check_and_convert_event_data(
            event_times, event_indicators
        )
        self.event_times = event_times
        self.event_indicators = event_indicators
        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert_event_data(
                train_event_times, train_event_indicators
            )
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        self._NO_CENSOR = np.all(self.event_indicators == 1)
        super().__init__(
            pred_survs,
            time_coordinates,
            n_samples=self.event_times.shape[0],
            predict_time_method=predict_time_method,
            interpolation=interpolation,
        )

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError(
                "Train set information is missing. "
                f"Evaluator cannot perform {method_name} evaluation."
            )

    @property
    def default_target_time(self) -> float:
        """Median observed time across the available training and test outcomes."""
        times = self.event_times
        if self.train_event_times is not None:
            times = np.concatenate((times, self.train_event_times))
        return float(np.quantile(times, 0.5))

    def concordance(
        self,
        ties: str = "Risk",
        method: str = "Harrell",
        tau: Numeric | None = None,
    ) -> tuple[float, float, float]:
        """
        Calculate the concordance index between the predicted survival times and the true survival times.

        Parameters
        ----------
        ties: str, default = "Risk"
            A string indicating the way ties should be handled.
            Options: "None", "Time", "Risk" (default), or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        method: str, default = "Harrell"
            A string indicating the method for constructing the pairs of samples.
            Options are "Harrell" (default), "Naive", "Uno", "IPCW", or "Margin".
            "Harrell": comparable pairs are anchored by samples with observed
            events. If sample i has an observed event at time t_i, it is compared
            with samples whose observed event/censoring time is greater than t_i.
            "Naive": alias of "Harrell".
            "Uno": Harrell-style comparable pairs weighted by inverse probability
            of censoring weights from the training data.
            "IPCW": alias of "Uno".
            "Margin": the pairs are constructed between all samples. A best-guess
            time for the censored samples will be calculated and used to construct
            the pairs. "Uno", "IPCW", and "Margin" require training set
            information.
        tau: float, optional (default=None)
            Truncation time. If provided, only pairs whose effective earlier or
            anchor time is strictly before ``tau`` are counted. If None, no
            truncation is applied.

        Returns
        -------
        concordance_index: float
            The concordance index.
        num_concordant_pairs: float
            The number of concordant pairs.
        num_total_pairs: float
            The number of total pairs.
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "harrell"

        if method in {"margin", "uno", "ipcw"}:
            self._error_trainset(f"{method} concordance")

        return concordance(
            predicted_times=self.predicted_event_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties,
            tau=tau,
        )

    def auc(self, target_time: Numeric | None = None, method: str = "Naive") -> float:
        """Calculate cumulative/dynamic AUROC from predicted survival curves.

        Cases are observed events at or before the target. Controls are
        observations after the target and censorings exactly at the target.
        Earlier censorings are excluded.

        Args:
            target_time: Evaluation time. If None, use the median observed time
                of the combined training and test sets, or just the test set
                when training outcomes are unavailable.
            method: "Naive" (default), "IPCW", or its alias "Uno", case-insensitive. IPCW requires
                training outcomes, assumes independent censoring, and weights
                cases by 1/G(event_time-) from reverse Kaplan-Meier.

        Returns:
            AUROC at the target time.

        Raises:
            TypeError: If IPCW is requested without training outcomes.
            ValueError: If inputs or the method are invalid, either class is
                absent, or IPCW cannot be estimated at the target.
        """
        if method.lower() in {"ipcw", "uno"}:
            self._error_trainset("IPCW AUROC")

        if target_time is None:
            target_time = self.default_target_time

        predict_probs = self.predict_probability_from_curve(target_time)

        return auc(
            predict_probs=predict_probs,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            target_time=target_time,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
        )

    def auroc(self, target_time: Numeric | None = None, method: str = "Naive") -> float:
        """Alias for :meth:`auc`, including its boundary and IPCW conventions.

        Args:
            target_time: Evaluation time, defaulting to the median observed
                time of the available training and test outcomes.
            method: "Naive" (default), "IPCW", or its alias "Uno", case-insensitive.

        Returns:
            AUROC at the target time.
        """
        return self.auc(target_time, method=method)

    def concordance_time_dependent(
        self,
        ties: str = "Risk",
        method: str = "Antolini",
        risks: str = "Survival",
        tau: Numeric | None = None,
    ) -> tuple[float, float, float]:
        """
        Calculate the time-dependent concordance index.

        Risk scores are evaluated at observed-event anchor times, processing
        one sample's curve at a time. Each contributing time is predicted at
        most once per sample; observed events tied on time remain separate
        anchors for pair counting.

        Parameters
        ----------
        ties: str, default = "Risk"
            A string indicating the way ties should be handled.
            Options: "None", "Time", "Risk" (default), or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        method: str, default = "Antolini"
            A string indicating the method for constructing the pairs of samples.
            Options are "Antolini" (default), "Naive", "IPCW".
            "Antolini": comparable pairs are anchored by samples with observed
            events. If sample i has an observed event at time t_i, it is compared
            with samples whose observed event/censoring time is greater than t_i.
            "Naive": alias of "Antolini".
            "IPCW": Antolini-style comparable pairs weighted by inverse probability
            of censoring weights from the training data.
        risks: str, default = "Survival"
            A string indicating the type of risk scores to be used for concordance calculation.
            Options are "Survival" (default) and "Hazard".
            "Survival": predicted survival probabilities at anchor event times are converted to risk scores
            using -S(t | z).
            "Hazard": predicted hazard rates at anchor event times are used directly as risk scores.
        tau: float, optional (default=None)
            Truncation time. If provided, only pairs whose effective earlier or
            anchor time is strictly before ``tau`` are counted. If None, no
            truncation is applied.

        Returns
        -------
        concordance_index: float
            The time-dependent concordance index.
        num_concordant_pairs: float
            The number of concordant pairs.
        num_total_pairs: float
            The number of total pairs.

        Notes
        -----
        Event anchors with the same time are sorted by risk, allowing later
        samples to count concordant and tied pairs with rank queries rather
        than visiting each anchor. With U contributing distinct event times
        and at most M anchors at one time, counting uses O(n) working memory
        and O(n log n + n U log(M + 1)) time. Prediction adds O(k) temporary
        storage for one curve with k grid points. These bounds exclude the
        stored input curves and IPCW training data; no dense sample-by-anchor
        risk matrix is constructed. If all event times are distinct, counting
        still has quadratic worst-case complexity.
        """
        risks = risks.lower()

        if self._NO_CENSOR:
            method = "antolini"

        method = _normalize_time_dependent_method(method)
        ties = _normalize_ties(ties)

        if risks not in {"survival", "hazard"}:
            error = "Risks must be 'Survival' or 'Hazard', " f"got '{risks}' instead"
            raise ValueError(error)

        if method == "ipcw":
            self._error_trainset("IPCW time-dependent concordance")

        return _concordance_time_dependent(
            risk_scores=partial(self._predict_risks_from_curve, risks=risks),
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties,
            tau=tau,
        )

    def brier_score(
        self, target_time: Numeric | None = None, IPCW_weighted: bool = True
    ) -> float:
        """
        Calculate the Brier score at a given time point from the predicted survival curve.

        Parameters
        ----------
        target_time: float, int, or None, default = None
            Time point at which the Brier score is to be calculated. If None, the Brier score is calculated at the
            median time of the test event/censor times and, when available, the training event/censor times.
        IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.
        :return: float
            The Brier score at the target time point.
        """
        if target_time is None:
            target_time = self.default_target_time

        return float(
            self.brier_score_multiple_points(np.asarray([target_time]), IPCW_weighted)[
                0
            ]
        )

    def brier_score_multiple_points(
        self,
        target_times: np.ndarray,
        IPCW_weighted: bool = True,
    ) -> np.ndarray:
        """
        Calculate multiple Brier scores at multiple specific times.

        Parameters
        ----------
        target_times: np.ndarray
            The specific time points for which to estimate the Brier scores.
        IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.

        Returns
        -------
        brier_scores: np.ndarray
            Values of multiple Brier scores, in the same order as `target_times`.
        """
        # Fully observed test outcomes use the plain Brier score.
        if self._NO_CENSOR:
            IPCW_weighted = False

        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Brier score (BS)")

        target_times = validate_time_points(target_times, input_name="target_times")

        pred_probs_mat = self.predict_multi_probabilities_from_curve(target_times)

        return brier_multiple_points(
            pred_mat=pred_probs_mat,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            target_times=target_times,
            ipcw=IPCW_weighted,
        )

    def integrated_brier_score(
        self,
        num_points: int | None = None,
        target_times: np.ndarray | None = None,
        IPCW_weighted: bool = True,
        integration_method: str = "trapz",
        draw_figure: bool = False,
    ) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
        """
        Calculate the integrated Brier score (IBS) from the predicted survival curve.

        Parameters
        ----------
        num_points : int, optional (default=None)
            Number of evaluation time points to generate automatically.
            If provided, `target_times` must be None.
            We generate `num_points` linearly spaced times between 0 and max_target_time,
            where max_target_time is the maximum observed (event or censoring) time
            across train + test.

            If both `num_points` and `target_times` are None:
            - Infer `target_times` from the unique censoring times in the test set.

        target_times : np.ndarray, shape = (m,), optional (default=None)
            Explicit time grid at which to evaluate the Brier score.
            If provided, `num_points` must be None. We'll integrate over exactly these times.
            If unsorted, values are sorted ascending with a warning. You are
            responsible for making sure they lie within the support of the model.

        IPCW_weighted: bool, default = True
            Whether to use IPCW weighting.

        integration_method: str, default: "trapz"
            Numerical integration method. Options: "trapz" (trapezoidal), "simpson".

        draw_figure: bool, default = False
            Whether to draw the figure of the IBS.

        Returns
        -------
        ibs: float
            The integrated Brier score when `draw_figure` is False.
        result: tuple[float, tuple[plt.Figure, plt.Axes]]
            When `draw_figure` is True, returns `(ibs, (fig, ax))`.
        """
        # Fully observed test outcomes use the plain Brier score.
        if self._NO_CENSOR:
            IPCW_weighted = False

        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Integrated Brier Score (IBS)")

        target_times = _brier_time_grid(
            self.event_times,
            self.event_indicators,
            self.train_event_times,
            num_points,
            target_times,
        )
        b_scores = self.brier_score_multiple_points(target_times, IPCW_weighted)
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(target_times, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are:\n", bs_dict)

        ibs_score = _integrate_brier_scores(b_scores, target_times, integration_method)

        # Draw the Brier score graph
        if draw_figure:
            fig, ax = plt.subplots()
            ax.plot(target_times, b_scores, "bo-")
            score_text = rf"IBS$= {ibs_score:.3f}$"
            ax.plot([], [], " ", label=score_text)
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Brier Score")
            ax.set_title("Integrated Brier Score")
            plt.show()
            return ibs_score, (fig, ax)
        return ibs_score

    def mae(
        self,
        method: str = "Hinge",
        weighted: bool | None = None,
        log_scale: bool = False,
        verbose: bool = False,
        truncated_time=None,
    ) -> float:
        """
        Calculate the MAE score for the test set.

        Parameters
        ----------
        method: string, default: "Hinge"
            The method used to calculate the MAE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted: bool, default: None
            Whether to use weighting scheme for MAE.
            If None, the default value is False for "Uncensored" and "Hinge" methods, and True for the rest.
        log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        verbose: boolean, default: False
            Whether to show the progress bar.
        truncated_time: float, default: None
            Truncated time.

        Returns
        -------
        mae_score: float
            The MAE score for the test set.
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "uncensored"

        if weighted is None:
            weighted = method not in ("uncensored", "hinge")

        return mean_error(
            predicted_times=self.predicted_event_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="absolute",
            method=method,
            weighted=weighted,
            log_scale=log_scale,
            verbose=verbose,
            truncation_time=truncated_time,
        )

    def mse(
        self,
        method: str = "Hinge",
        weighted: bool | None = None,
        log_scale: bool = False,
        verbose: bool = False,
        truncated_time=None,
    ) -> float:
        """
        Calculate the MSE score for the test set.

        Parameters
        ----------
        method: string, default: "Hinge"
            The method used to calculate the MSE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted: bool, default: None
            Whether to use weighting scheme for MSE.
            If None, the default value is False for "Uncensored" and "Hinge" methods, and True for the rest.
        log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        verbose: boolean, default: False
            Whether to show the progress bar.
        truncated_time: float, default: None
            Truncated time.

        Returns
        -------
        mse_score: float
            The MSE score for the test set.
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "uncensored"

        if weighted is None:
            weighted = method not in ("uncensored", "hinge")

        return mean_error(
            predicted_times=self.predicted_event_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="squared",
            method=method,
            weighted=weighted,
            log_scale=log_scale,
            verbose=verbose,
            truncation_time=truncated_time,
        )

    def rmse(
        self,
        method: str = "Hinge",
        weighted: bool | None = None,
        log_scale: bool = False,
        verbose: bool = False,
        truncated_time=None,
    ) -> float:
        """
        Calculate the root mean squared error (RMSE) score for the test set.

        Parameters
        ----------
        method: string, default: "Hinge"
            The method used to calculate the RMSE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted: bool, default: None
            Whether to use weighting scheme for RMSE.
            If None, the default value is False for "Uncensored" and "Hinge" methods, and True for the rest.
        log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        verbose: boolean, default: False
            Whether to show the progress bar.
        truncated_time: float, default: None
            Truncated time.

        Returns
        -------
        rmse_score: float
            The RMSE score for the test set.
        """
        return self.mse(method, weighted, log_scale, verbose, truncated_time) ** 0.5

    def one_calibration(
        self,
        target_time: Numeric | None = None,
        num_bins: int = 10,
        binning_strategy: str = "C",
        method: str = "DN",
        return_details: bool = False,
        adjust_variance: bool = True,
    ) -> tuple[float, list, list] | tuple[float, dict]:
        """
        Calculate the one calibration score at a given time point from the predicted survival curve.
        Parameters
        ----------
        target_time: Numeric | None, default: None
            Evaluation time. None uses default_target_time, the median observed
            time across available training and test outcomes.
        num_bins: int, default: 10
            Number of bins used to calculate the one calibration score.
        binning_strategy: str
            The strategy to bin the predictions. The options are: "C" (default), and "H".
            C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
            H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
        method: string, default: "DN"
            The method used to calculate the one calibration score.
            Options: "Uncensored", or "DN" (default)
        return_details: bool, default: False
            Whether to return detailed calibration information, including plots and statistics.

        adjust_variance: bool, default: True
            Use censoring-aware Greenwood variance for DN, or observed binomial
            variance for Uncensored, with G degrees of freedom. False uses
            the original expected binomial variance and HL/DN convention.
            Adjusted p-values are undefined if any group variance is zero.

        Returns
        -------
        p_value: float
            The approximate calibration p-value; NaN for an undefined adjusted test.
        observed_probabilities: list
            The observed probabilities in each bin.
        expected_probabilities: list
            The expected probabilities in each bin.
        details: dict, optional
            A dictionary containing detailed calibration information, including:
            - p_value: The approximate calibration p-value; NaN for an undefined adjusted test.
            - statistics: The grouped calibration statistic.
            - variances, bin_sizes, degrees_of_freedom: Test variance and group information.
            - observed_probabilities: The observed probabilities in each bin.
            - expected_probabilities: The expected probabilities in each bin.
            - slope: The slope of the calibration curve.
            - intercept: The intercept of the calibration curve.
            - max_local_deviation: The maximum local deviation between observed and expected probabilities.
            - histogram_plot: A tuple containing the figure and axes of the histogram plot.
            - pp_plot: A tuple containing the figure and axes of the P-P plot.
            Only returned when 'return_details' is set to True.
        """
        if target_time is None:
            target_time = self.default_target_time

        if self._NO_CENSOR:
            method = "Uncensored"

        predict_probs = self.predict_probability_from_curve(target_time)
        p_value, details = one_calibration(
            preds=1 - predict_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=target_time,
            num_bins=num_bins,
            binning_strategy=binning_strategy,
            method=method,
            adjust_variance=adjust_variance,
            return_details=True,
        )
        if return_details:
            details.update(
                _calibration_curve_details(
                    details["observed_probabilities"], details["expected_probabilities"]
                )
            )
            return p_value, details

        return (
            p_value,
            details["observed_probabilities"],
            details["expected_probabilities"],
        )

    def integrated_calibration_index(
        self,
        target_time: Numeric | None = None,
        knots: int = 3,
        draw_figure: bool | None = True,
        figure_range: tuple | None = None,
    ) -> tuple[dict, plt.Figure]:
        """
        Calculate the integrated one calibration index (ICI) for a given set of predictions and true event times.

        Parameters
        ----------
        target_time: Numeric | None, default: None
            Evaluation time. None uses default_target_time, the median observed
            time across available training and test outcomes.
        knots: int, default: 3
            The number of knots to use for the spline fit in the ICI calculation.
        draw_figure: bool, default: True
            Whether to draw the figure of the ICI.
        figure_range: tuple, default: None
            The range of the figure to be drawn. If None, the range will be automatically determined based on the
            predicted probabilities and the target time.

        Returns
        -------
        summary: dict
            The summary of ICI for the target time point,
            including the ICI value, the E50, E90, and the E_max values.
        fig: plt.figure
            A figure showing the graphical calibration curve.
        """
        if target_time is None:
            target_time = self.default_target_time

        predict_probs = self.predict_probability_from_curve(target_time)
        return integrated_calibration_index(
            preds=1 - predict_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=target_time,
            knots=knots,
            draw_figure=draw_figure,
            figure_range=figure_range,
        )

    def d_calibration(
        self, num_bins: int = 10, return_details: bool = False
    ) -> tuple[float, np.ndarray] | tuple[float, dict]:
        """
        Calculate the D calibration score from the predicted survival curve.
        Parameters
        ----------
        num_bins: int, default: 10
            Number of bins used to calculate the D calibration score.
        return_details: bool, default: False
            Whether to return detailed calibration information, including plots and statistics.

        Returns
        -------
        p_value: float
            The p-value of the calibration test.
        hist: np.ndarray
            The histogram of the predicted probabilities in each bin.
        details: dict, optional
            A dictionary containing detailed calibration information, including:
            - statistics: The D calibration statistics.
            - p_value: The p-value of the calibration test.
            - histogram: The histogram of the predicted probabilities in each bin.
            - x_calibration: The x-calibration score.
            - linear_slope_intercept: A dictionary containing the slope and intercept of the linear fit.
            - max_local_slope: The maximum local slope deviation.
            - histogram_plot: A tuple containing the figure and axes of the histogram plot.
            - pp_plot: A tuple containing the figure and axes of the P-P plot.
        """
        predict_probs = self.predict_probability_from_curve(self.event_times)
        statistics, p_value, hist = d_calibration(
            pred_probs=predict_probs,
            event_indicators=self.event_indicators,
            num_bins=num_bins,
        )
        if return_details:
            # normalize the histogram
            N = hist.sum()
            d_cal_pdf = hist / N
            # compute the x-calibration score
            optimal = np.ones_like(d_cal_pdf) / num_bins
            x_cal = np.sum(np.square(d_cal_pdf - optimal))

            d_cal_cdf = np.cumsum(
                d_cal_pdf[::-1]
            )  # so that the first number belongs to the lowest prob bin
            d_cal_cdf = np.insert(d_cal_cdf, 0, 0)
            optimal_cdf = np.linspace(0, 1, num_bins + 1)

            # LS fit of observed CDF vs expected CDF
            slope, intercept = fit_least_squares(
                optimal_cdf, d_cal_cdf, left_anchor=False, right_anchor=False
            )

            # Local slope deviation, max slope with the highest ratio difference from 1
            max_slope = _maximum_local_deviation(d_cal_cdf, optimal_cdf)

            # horizontal histograms
            fig1, ax1 = plt.subplots()
            widths = hist
            gap_fraction = 0.95

            # label the y positions
            y_labels = [
                f"[{i/num_bins:.2f}, {(i+1)/num_bins:.2f}{')' if i < num_bins - 1 else ']'}"
                for i in range(num_bins)
            ]
            y_positions = (optimal_cdf[:-1] + optimal_cdf[1:]) / 2  # midpoints of bins

            ax1.barh(y_positions, widths, height=gap_fraction / num_bins, alpha=0.7)
            ax1.axvline(N / num_bins, ls="dashed", c="grey", label="Ideal Calibration")
            ax1.set_xlabel("Frequency")
            ax1.set_ylabel("Probability Bins")
            ax1.set_yticks(y_positions)
            ax1.set_yticklabels(y_labels)
            ax1.legend()
            fig1.tight_layout()

            # P-P plot
            fig2, ax2 = pp_plot(d_cal_cdf, optimal_cdf)

            details = {
                "statistics": statistics,
                "p_value": p_value,
                "histogram": hist,
                "x_calibration": x_cal,
                "linear_slope_intercept": {"slope": slope, "intercept": intercept},
                "max_local_slope": max_slope,
                "histogram_plot": (fig1, ax1),
                "pp_plot": (fig2, ax2),
            }
            return p_value, details

        return p_value, hist

    def ksd_calibration(
        self, return_details: bool = False
    ) -> tuple[float, float] | tuple[float, dict]:
        """
        Calculate the KSD calibration score from the predicted survival curve.
        Parameters
        ----------
        return_details: bool, default: False
            Whether to return detailed calibration information, including plots and statistics.
        Returns
        -------
        p_value: float
            The p-value of the calibration test.
        ksd_statistic: float
            The KSD calibration statistic.
        details: dict, optional
            A dictionary containing detailed calibration information, including:
            - p_value: The p-value of the calibration test.
            - ksd_statistic: The KSD calibration statistic.
            - histogram_plot: A tuple containing the figure and axes of the histogram plot.
            - pp_plot: A tuple containing the figure and axes of the P-P plot.
        """
        predict_probs = self.predict_probability_from_curve(self.event_times)
        return ksd_calibration(
            pred_probs=predict_probs,
            event_indicators=self.event_indicators,
            return_details=return_details,
        )

    def residuals(
        self,
        method: str = "CoxSnell",
        draw_figure: bool = False,
    ) -> np.ndarray:
        """
        Calculate the residuals from the predicted survival (cumulative hazard) function.

        Parameters
        ----------
        method: str, default: "CoxSnell"
            The method used to calculate the residuals.
            Options: "CoxSnell" (default), "Modified CoxSnell-v1", "Modified CoxSnell-v2",
            "Martingale", "Deviance".
        draw_figure: bool, default: False
            Whether to draw the figure of the residuals.

        Returns
        -------
        residuals: np.ndarray
            The residuals calculated from the predicted survival curve.
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = (
                "coxsnell"
                if method
                in ["coxsnell", "modified coxsnell-v1", "modified coxsnell-v2"]
                else method
            )

        predict_probs = self.predict_probability_from_curve(self.event_times)
        return residuals(
            pred_probs=predict_probs,
            event_indicators=self.event_indicators,
            method=method,
            draw_figure=draw_figure,
        )

    def km_calibration(
        self, draw_figure: bool = False
    ) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
        """
        Calculate the KM calibration score from the predicted survival curve.
        Parameters
        ----------
        draw_figure: bool, default: False
            Whether to draw the figure of the KM calibration curve.
        Returns
        -------
        km_cal: float
            Normalized integrated squared error between the average predicted
            survival curve and the Kaplan-Meier estimate. Lower is better.
        """
        if self.ndim_time == 1:
            average_survival_curve = np.mean(self._pred_survs, axis=0, dtype=float)
            calibration_times = self._time_coordinates
        else:
            calibration_times = np.unique(self.event_times[self.event_indicators == 1])
            sample_survival_curves = self.predict_multi_probabilities_from_curve(
                calibration_times
            )
            average_survival_curve = np.mean(
                sample_survival_curves, axis=0, dtype=float
            )

        return km_calibration(
            average_survival_curve=average_survival_curve,
            time_coordinates=calibration_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            interpolation_method=self.interpolation,
            draw_figure=draw_figure,
        )

    def log_rank(
        self,
        weightings: str | None = None,
        p: float | None = 0,
        q: float | None = 0,
    ) -> tuple[float, float]:
        """
        Calculate the log-rank test statistic and p-value for the predicted survival curve.

        Parameters
        ----------
        weightings: str, optional
           The weighting method is for weighted log-rank test.
           Options: "None" (default), "wilcoxon", "tarone-ware", "peto", "fleming-harrington".
           None means unweighted log-rank test.
           Wilcoxon uses the number of at-risk population at each time point as the weight.
           Tarone-Ware uses the square root of the number of at-risk population at each time point as the weight.
           Peto uses the estimated survival probability as the weight.
           Fleming-Harrington uses
               w_i = S(t_i) ** p * (1 - S(t_i)) ** q
        p: float, default: 0
            The p parameter for the Fleming-Harrington weighting method.
        q: float, default: 0
            The q parameter for the Fleming-Harrington weighting method.

        Returns
        -------
        p_value: float
            The p-value of the log-rank test.
        test_statistic: float
            The test statistic of the log-rank test.
        """
        results = logrank_test(
            durations_A=self.event_times,
            durations_B=self.predicted_event_times,
            event_observed_A=self.event_indicators,
            event_observed_B=np.ones_like(self.event_indicators, dtype=bool),
            weightings=weightings,
            p=p,
            q=q,
        )
        return results.p_value, results.test_statistic

    def auprc(self, n_quad: int = 256) -> float:
        """
        Calculate the survival AUPRC for right-censored data.

        Parameters
        ----------
        n_quad: int, default: 256
            Number of quadrature intervals used for numerical integration.

        Returns
        -------
        auprc: float
            The survival AUPRC.
        """
        predictions_cdf = 1 - self._pred_survs
        return auprc_right_censor(
            pred_cdf=predictions_cdf,
            time_grid=self._time_coordinates,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            n_quad=n_quad,
            return_details=False,
        )


class PycoxEvaluator(SurvivalEvaluator, ABC):
    def __init__(
        self,
        surv: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ):
        """
        Evaluator for survival models in PyCox packages.

        Parameters
        ----------
        surv: pd.DataFrame, shape = (n_time_points, n_samples)
            Predicted survival curves for the testing samples
            DataFrame index represents the time coordinates for the given curves.
            DataFrame value represents transpose of the survival probabilities.
        event_times: NumericArrayLike, shape = (n_samples,)
            Event times for the testing samples.
        event_indicators: NumericArrayLike, shape = (n_samples,)
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        train_event_times: NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        train_event_indicators: NumericArrayLike, shape = (n_samples,), optional
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        predict_time_method: string, default: "Median"
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean", and "RMST".
        interpolation: string, default: "Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        time_coordinates = surv.index.values
        predicted_survival_curves = surv.to_numpy(copy=False).T
        # Do not mutate the caller's DataFrame when correcting numerical noise.
        if np.any(predicted_survival_curves < 0):
            predicted_survival_curves = np.maximum(predicted_survival_curves, 0)
        super().__init__(
            predicted_survival_curves,
            time_coordinates,
            event_times,
            event_indicators,
            train_event_times,
            train_event_indicators,
            predict_time_method,
            interpolation,
        )


class LifelinesEvaluator(PycoxEvaluator, ABC):
    def __init__(
        self,
        surv: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ):
        """
        Evaluator for survival models in Lifelines packages.

        Parameters
        ----------
        surv: pd.DataFrame, shape = (n_time_points, n_samples)
            Predicted survival curves for the testing samples
        event_times: NumericArrayLike, shape = (n_samples,)
            Event times for the testing samples.
        event_indicators: NumericArrayLike, shape = (n_samples,)
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        train_event_times: NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        train_event_indicators: NumericArrayLike, shape = (n_samples,), optional
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        predict_time_method: string, default: "Median"
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean" and "RMST".
        interpolation: string, default: "Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        super().__init__(
            surv,
            event_times,
            event_indicators,
            train_event_times,
            train_event_indicators,
            predict_time_method,
            interpolation,
        )


class ScikitSurvivalEvaluator(SurvivalEvaluator, ABC):
    def __init__(
        self,
        surv,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ):
        """
        Evaluator for survival models in scikit-survival packages.

        Parameters
        ----------
        surv: shape = (n_samples,)
            Predicted survival curves for the testing samples from scikit-survival model.
            Each element is a scikit-survival customized object.
            '.x' attribute is the time coordinates for the given curve. '.y' attribute is the survival probabilities.
        event_times: NumericArrayLike, shape = (n_samples,)
            Event times for the testing samples.
        event_indicators: NumericArrayLike, shape = (n_samples,)
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        train_event_times: NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        train_event_indicators: NumericArrayLike, shape = (n_samples,), optional
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        predict_time_method: string, default: "Median"
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean", and "RMST".
        interpolation: string, default: "Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        time_coordinates = surv[0].x
        predict_curves = []
        for i in range(len(surv)):
            predict_curve = surv[i].y
            if False in (time_coordinates == surv[i].x):
                raise KeyError(
                    f"{i}-th survival curve does not have same time coordinates"
                )
            predict_curves.append(predict_curve)
        predicted_curves = np.array(predict_curves)
        # Ensure flat all-one curves have a finite extrapolated event time.
        end_probabilities = predicted_curves[:, -1]
        idx_need_fix = end_probabilities == 1
        if np.any(idx_need_fix):
            reference_end_probabilities = end_probabilities[~idx_need_fix]
            if reference_end_probabilities.size == 0:
                replacement_probability = 0.99
            else:
                max_prob_at_end = np.max(reference_end_probabilities)
                replacement_probability = max(0.1 * max_prob_at_end + 0.9, 0.99)
            predicted_curves[idx_need_fix, -1] = replacement_probability
        super().__init__(
            predicted_curves,
            time_coordinates,
            event_times,
            event_indicators,
            train_event_times,
            train_event_indicators,
            predict_time_method,
            interpolation,
        )


DistributionEvaluator = SurvivalEvaluator  # Alias for the SurvivalEvaluator


class PointEvaluator:
    def __init__(
        self,
        pred_times: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
    ):
        """
        Initialize the Evaluator

        Parameters
        ----------
        pred_times: structured array, shape = (n_samples, )
            Predicted survival times for the testing samples.
        event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        event_indicators: structured array, shape = (n_samples, )
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        train_event_indicators: structured array, shape = (n_train_samples, )
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        """
        self.event_times, self.event_indicators = check_and_convert_event_data(
            event_times, event_indicators
        )
        self.pred_times = pred_times

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert_event_data(
                train_event_times, train_event_indicators
            )
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        self._NO_CENSOR = np.all(self.event_indicators == 1)

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError(
                "Train set information is missing. "
                f"Evaluator cannot perform {method_name} evaluation."
            )

    @property
    def pred_times(self):
        return self._pred_times

    @pred_times.setter
    def pred_times(self, pred_times):
        pred_times = check_and_convert(pred_times)
        if pred_times.shape != self.event_times.shape:
            raise ValueError("pred_times and event_times must have the same shape.")
        self._pred_times = pred_times

    def concordance(
        self,
        ties: str = "Risk",
        method: str = "Harrell",
        tau: Numeric | None = None,
    ) -> tuple[float, float, float]:
        """
        Calculate the concordance index between the predicted survival times and the true survival times.

        Parameters
        ----------
        ties: str, default = "Risk"
            A string indicating the way ties should be handled.
            Options: "None", "Time", "Risk" (default), or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        method: str, default = "Harrell"
            A string indicating the method for constructing the pairs of samples.
            Options are "Harrell" (default), "Naive", "Uno", "IPCW", or "Margin".
            "Harrell": comparable pairs are anchored by samples with observed
            events. If sample i has an observed event at time t_i, it is compared
            with samples whose observed event/censoring time is greater than t_i.
            "Naive": alias of "Harrell".
            "Uno": Harrell-style comparable pairs weighted by inverse probability
            of censoring weights from the training data.
            "IPCW": alias of "Uno".
            "Margin": the pairs are constructed between all samples. A best-guess
            time for the censored samples will be calculated and used to construct
            the pairs. "Uno", "IPCW", and "Margin" require training set
            information.
        tau: float, optional (default=None)
            Truncation time. If provided, only pairs whose effective earlier or
            anchor time is strictly before ``tau`` are counted. If None, no
            truncation is applied.

        Returns
        -------
        concordance_index: float
            The concordance index, which is the proportion of concordant pairs among all pairs.
        concordant_pairs: float
            The number of concordant pairs.
        total_pairs: float
            The total number of comparable pairs considered in the concordance calculation.
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "harrell"

        if method in {"margin", "uno", "ipcw"}:
            self._error_trainset(f"{method} concordance")

        return concordance(
            predicted_times=self._pred_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties,
            tau=tau,
        )

    def mae(
        self,
        method: str = "Hinge",
        weighted: bool | None = None,
        log_scale: bool = False,
        verbose: bool = False,
        truncated_time=None,
    ) -> float:
        """
        Calculate the MAE score for the test set.

        Parameters
        ----------
        method: string, default: "Hinge"
            The method used to calculate the MAE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted: bool, default: None
            Whether to use weighting scheme for MAE.
            If None, the default value is False for "Uncensored" and "Hinge" methods, and True for the rest.
        log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        verbose: boolean, default: False
            Whether to show the progress bar.
        truncated_time: float, default: None
            Truncated time.

        Returns
        -------
        mae_score: float
            The MAE score for the test set.
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "uncensored"

        if weighted is None:
            weighted = method not in ("uncensored", "hinge")

        return mean_error(
            predicted_times=self._pred_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="absolute",
            method=method,
            weighted=weighted,
            log_scale=log_scale,
            verbose=verbose,
            truncation_time=truncated_time,
        )

    def mse(
        self,
        method: str = "Hinge",
        weighted: bool | None = None,
        log_scale: bool = False,
        verbose: bool = False,
        truncated_time=None,
    ) -> float:
        """
        Calculate the MSE score for the test set.

        Parameters
        ----------
        method: string, default: "Hinge"
            The method used to calculate the MSE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted: bool, default: None
            Whether to use weighting scheme for MSE.
            If None, the default value is False for "Uncensored" and "Hinge" methods, and True for the rest.
        log_scale: boolean, default: False
            Whether to use log scale for the time axis.
        verbose: boolean, default: False
            Whether to show the progress bar.
        truncated_time: float, default: None
            Truncated time.

        Returns
        -------
        mse_score: float
            The MSE score for the test set.
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "uncensored"

        if weighted is None:
            weighted = method not in ("uncensored", "hinge")

        return mean_error(
            predicted_times=self._pred_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="squared",
            method=method,
            weighted=weighted,
            log_scale=log_scale,
            verbose=verbose,
            truncation_time=truncated_time,
        )

    def rmse(
        self,
        method: str = "Hinge",
        weighted: bool | None = None,
        log_scale: bool = False,
        verbose: bool = False,
        truncated_time=None,
    ) -> float:
        """
        Calculate the root mean squared error (RMSE) score for the test set.

        Parameters
        ----------
        method: string, default: "Hinge"
            The method used to calculate the RMSE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted: bool, default: None
            Whether to use a weighting scheme for RMSE.
            If None, the default value is False for "Uncensored" and "Hinge" methods, and True for the rest.
        log_scale: boolean, default = False
            Whether to use log scale for the time axis.
        verbose: boolean, default = False
            Whether to show the progress bar.
        truncated_time: float, default: None
            Truncated time.

        Returns
        -------
        rmse_score: float
            The RMSE score for the test set.
        """
        return self.mse(method, weighted, log_scale, verbose, truncated_time) ** 0.5

    def log_rank(
        self,
        weightings: str | None = None,
        p: float | None = 0,
        q: float | None = 0,
    ) -> tuple[float, float]:
        """
        Calculate the log-rank test statistic and p-value for the predicted survival curve.

        Parameters
        ----------
        weightings: str, optional
           The weighting method is for weighted log-rank test.
           Options: "None" (default), "wilcoxon", "tarone-ware", "peto", "fleming-harrington".
           None means unweighted log-rank test.
           Wilcoxon uses the number of at-risk population at each time point as the weight.
           Tarone-Ware uses the square root of the number of at-risk population at each time point as the weight.
           Peto uses the estimated survival probability as the weight.
           Fleming-Harrington uses
               w_i = S(t_i) ** p * (1 - S(t_i)) ** q
        p: float, default: 0
            The p parameter for the Fleming-Harrington weighting method.
        q: float, default: 0
            The q parameter for the Fleming-Harrington weighting method.

        Returns
        -------
        p_value: float
            The p-value of the log-rank test.
        test_statistic: float
            The test statistic of the log-rank test.
        """
        results = logrank_test(
            durations_A=self.event_times,
            durations_B=self._pred_times,
            event_observed_A=self.event_indicators,
            event_observed_B=np.ones_like(self.event_indicators, dtype=bool),
            weightings=weightings,
            p=p,
            q=q,
        )
        return results.p_value, results.test_statistic


class SingleTimeEvaluator:
    def __init__(
        self,
        pred_probs: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        target_time: float | None = None,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
    ):
        """
        Initialize the Evaluator

        Parameters
        ----------
        pred_probs: structured array, shape = (n_samples, )
            Predicted survival probability at the target time for the testing samples.
        event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        event_indicators: structured array, shape = (n_samples, )
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        target_time: float, int, or None, default = None
            Time point at which the evaluation is to be performed. If None, the target time is set to the median time
        train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        train_event_indicators: structured array, shape = (n_train_samples, )
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        """
        self.event_times, self.event_indicators = check_and_convert_event_data(
            event_times, event_indicators
        )
        self.pred_probs = pred_probs

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert_event_data(
                train_event_times, train_event_indicators
            )
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        if target_time is None:
            # set to the median time of all the event/censor times from the training and test sets
            # if train set is not provided, use test set only
            event_times = (
                np.concatenate((self.event_times, self.train_event_times))
                if self.train_event_times is not None
                else self.event_times
            )
            target_time = np.quantile(event_times, 0.5)
        self.target_time = target_time
        self._NO_CENSOR = np.all(self.event_indicators == 1)

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError(
                "Train set information is missing. "
                f"Evaluator cannot perform {method_name} evaluation."
            )

    @property
    def pred_probs(self):
        return self._pred_probs

    @pred_probs.setter
    def pred_probs(self, pred_probs):
        pred_probs = check_and_convert(pred_probs)
        if pred_probs.ndim != 1:
            raise ValueError(
                "predicted_probs should be a 1D array-like object, "
                f"but got a {pred_probs.ndim}D array-like object"
            )
        if pred_probs.shape != self.event_times.shape:
            raise ValueError("pred_probs and event_times must have the same shape.")
        self._pred_probs = pred_probs

    def auc(self, method: str = "Naive") -> float:
        """Calculate cumulative/dynamic AUROC at this evaluator's target time.

        Cases are observed events at or before the target. Controls are
        observations after the target and censorings exactly at the target.
        Earlier censorings are excluded.

        Args:
            method: "Naive" (default), "IPCW", or its alias "Uno", case-insensitive. IPCW requires
                training outcomes, assumes independent censoring, and weights
                cases by 1/G(event_time-) from reverse Kaplan-Meier.

        Returns:
            AUROC at the target time.

        Raises:
            TypeError: If IPCW is requested without training outcomes.
            ValueError: If inputs or the method are invalid, either class is
                absent, or IPCW cannot be estimated at the target.
        """
        if method.lower() in {"ipcw", "uno"}:
            self._error_trainset("IPCW AUROC")

        return auc(
            predict_probs=self._pred_probs,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            target_time=self.target_time,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
        )

    def brier_score(self, IPCW_weighted: bool = True) -> float:
        """
        Calculate the Brier score at a given time point from the predicted survival curve.

        Parameters
        ----------
        IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.

        Returns
        -------
        brier_score: float
            The Brier score at the target time point.
        """
        # Fully observed test outcomes use the plain Brier score.
        if self._NO_CENSOR:
            IPCW_weighted = False

        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Brier score (BS)")
        return float(
            brier_multiple_points(
                pred_mat=self._pred_probs[:, None],
                event_times=self.event_times,
                event_indicators=self.event_indicators,
                train_event_times=self.train_event_times,
                train_event_indicators=self.train_event_indicators,
                target_times=np.asarray([self.target_time]),
                ipcw=IPCW_weighted,
            )[0]
        )

    def one_calibration(
        self,
        num_bins: int = 10,
        binning_strategy: str = "C",
        method: str = "DN",
        return_details: bool = False,
        adjust_variance: bool = True,
    ) -> tuple[float, list, list] | tuple[float, dict]:
        """
        Calculate the one calibration score at a given time point from the predicted survival curve.

        Parameters
        ----------
        num_bins: int, default: 10
            Number of bins used to calculate the one calibration score.
        binning_strategy: str, default: "C"
            The strategy to bin the predictions. The options are: "C" (default), and "H".
            C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
            H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
        method: string, default: "DN"
            The method used to calculate the one calibration score.
            Options: "Uncensored", or "DN" (default)
        return_details: bool, default: False
            Whether to return the detailed calibration information.

        adjust_variance: bool, default: True
            Use censoring-aware Greenwood variance for DN, or observed binomial
            variance for Uncensored, with G degrees of freedom. False uses
            the original expected binomial variance and HL/DN convention.
            Adjusted p-values are undefined if any group variance is zero.

        Returns
        -------
        p_value: float
            The approximate calibration p-value; NaN for an undefined adjusted test.
        observed_probabilities: list
            The observed probabilities in each bin.
        expected_probabilities: list
            The expected probabilities in each bin.
        details: dict, optional
            A dictionary containing detailed calibration information, including:
            - p_value: The approximate calibration p-value; NaN for an undefined adjusted test.
            - statistics: The grouped calibration statistic.
            - variances, bin_sizes, degrees_of_freedom: Test variance and group information.
            - observed_probabilities: The observed probabilities in each bin.
            - expected_probabilities: The expected probabilities in each bin.
            - slope: The slope of the calibration curve.
            - intercept: The intercept of the calibration curve.
            - max_local_deviation: The maximum local deviation between observed and expected probabilities.
            - histogram_plot: A tuple containing the figure and axes of the histogram plot.
            - pp_plot: A tuple containing the figure and axes of the P-P plot.
            Only returned when 'return_details' is set to True.
        """
        if self._NO_CENSOR:
            method = "Uncensored"

        p_value, details = one_calibration(
            preds=1 - self._pred_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=self.target_time,
            num_bins=num_bins,
            binning_strategy=binning_strategy,
            method=method,
            adjust_variance=adjust_variance,
            return_details=True,
        )
        if return_details:
            details.update(
                _calibration_curve_details(
                    details["observed_probabilities"], details["expected_probabilities"]
                )
            )
            return p_value, details

        return (
            p_value,
            details["observed_probabilities"],
            details["expected_probabilities"],
        )

    def integrated_calibration_index(
        self,
        knots: int = 3,
        draw_figure: bool | None = True,
        figure_range: tuple | None = None,
    ) -> dict | tuple[dict, tuple[plt.Figure, plt.Axes]]:
        """
        Calculate the integrated one calibration index (ICI) for a given set of predictions and true event times.

        Parameters
        ----------
        knots: int, default = 3
            The number of knots to use for the spline fit. If None, the number of knots is automatically determined.
        draw_figure: bool, default = True
            Whether to create a figure showing the graphical calibration curve.
        figure_range: tuple, optional
            The range of the figure to be plotted. If None, the range is automatically determined.

        Returns
        -------
        summary: dict
            A dictionary containing the summary of ICI for the target time point, including the ICI value,
            the E50, E90, and the E_max values.
        fig: tuple[plt.Figure, plt.Axes]
            The matplotlib figure and axes objects for the calibration curve plot. Returned only if draw_figure is True.
        """
        return integrated_calibration_index(
            preds=1 - self._pred_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=self.target_time,
            knots=knots,
            draw_figure=draw_figure,
            figure_range=figure_range,
        )


class QuantileRegEvaluator(SurvivalEvaluator):
    def __init__(
        self,
        pred_regs: NumericArrayLike,
        quantile_levels: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ):
        """
        Initialize the quantile regression evaluator.
        Parameters
        ----------
        pred_regs: structured array,
            Accept shapes: (n_quantiles,) or (n_samples, n_quantiles).
            Predicted event-time quantiles for the testing samples. Each column
            corresponds to the matching entry in `quantile_levels`.
            At least one of `pred_regs` or `quantile_levels` must be a 2D array.
        quantile_levels: structured array,
            Accept shapes: (n_quantiles,) or (n_samples, n_quantiles).
            Cumulative event-probability levels corresponding to `pred_regs`.
            At least one of `pred_regs` or `quantile_levels` must be a 2D array.
        event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        event_indicators: structured array, shape = (n_samples, )
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        train_event_indicators: structured array, shape = (n_train_samples, )
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        predict_time_method: str, default = "Median"
            Method for calculating predicted survival time. Available options are "Median", "Mean" or "RMST".
        interpolation: str, default = "Linear"
            Method for interpolation. Available options are ['Linear', 'Pchip'].
        """
        quantile_levels = check_and_convert(quantile_levels)
        survival_level = 1 - quantile_levels
        if predict_time_method == "RMST":
            raise NotImplementedError(
                "RMST prediction time method is not implemented for quantile curves."
            )
        super().__init__(
            survival_level,
            pred_regs,
            event_times,
            event_indicators,
            train_event_times,
            train_event_indicators,
            predict_time_method,
            interpolation,
        )

    def km_calibration(self, draw_figure: bool = False):
        """
        Calculate the KM calibration score from the predicted survival curve.

        Parameters
        ----------
        draw_figure: bool, default: False
            Whether to draw the figure of the KM calibration.
        Returns
        -------
        km_cal: float
            Normalized integrated squared error between the average predicted
            survival curve and the Kaplan-Meier estimate. Lower is better.
        """
        unique_times = np.unique(self.event_times[self.event_indicators == 1])
        survival_curves = quantile_to_survival(
            1 - self._pred_survs,
            self.time_coordinates,
            unique_times,
            interpolate=self.interpolation,
        )
        avg_surv = np.mean(survival_curves, axis=0, dtype=float)

        return km_calibration(
            average_survival_curve=avg_surv,
            time_coordinates=unique_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            interpolation_method=self.interpolation,
            draw_figure=draw_figure,
        )
