import warnings
from functools import cached_property
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson, trapezoid

from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.Evaluations.AreaUnderPRCurve import auprc_ic
from SurvivalEVAL.Evaluations.BrierScore import brier_multiple_points_ic, brier_score_ic
from SurvivalEVAL.Evaluations.Concordance import (
    concordance,
    concordance_ic,
    impute_times_midpoint,
)
from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.DistributionCalibration import (
    coverage_ic,
    d_cal_ic,
    ksd_cal_ic,
)
from SurvivalEVAL.Evaluations.MeanError import inclusion_rate, mean_error_ic
from SurvivalEVAL.Evaluations.SingleTimeCalibration import one_cal_ic
from SurvivalEVAL.Evaluations.util import (
    check_and_convert,
    fit_least_squares,
    predict_mean_st,
    predict_median_st,
    predict_rmst,
    zero_padding,
)
from SurvivalEVAL.Evaluations.util_plots import pp_plot


class IntervalCenEvaluator(SurvivalEvaluator):
    """
    Evaluator for interval-censored survival data.
    """

    def __init__(
        self,
        pred_survs: NumericArrayLike,
        time_coordinates: NumericArrayLike,
        left_limits: NumericArrayLike,
        right_limits: NumericArrayLike,
        train_left_limits: Optional[NumericArrayLike] = None,
        train_right_limits: Optional[NumericArrayLike] = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ):
        """
        Initialize the Evaluator

        Parameters
        ----------
        pred_survs: NumericArrayLike
            Accept shapes: (n_time_points,) or (n_samples, n_time_points).
            Predicted survival probabilities for the testing samples.
            At least one of `pred_survs` or `time_coordinates` must be a 2D array.
        time_coordinates: NumericArrayLike
            Accept shapes: (n_time_points,) or (n_samples, n_time_points).
            Time coordinates for the predicted survival probabilities.
            At least one of `pred_survs` or `time_coordinates` must be a 2D array.
        left_limits: NumericArrayLike, shape = (n_samples,)
            Left limits of the interval-censored testing data.
        right_limits: NumericArrayLike, shape = (n_samples,)
            Right limits of the interval-censored testing data.
        train_left_limits: Optional[NumericArrayLike], shape = (n_train_samples,), default: None
            Left limits of the interval-censored data for the training set.
        train_right_limits: Optional[NumericArrayLike], shape = (n_train_samples,), default: None
            Right limits of the interval-censored data for the training set.
        predict_time_method: str, default: "Median"
            Method to predict time from the survival curve. Options are "Median", "Mean", or "RMST".
        interpolation: str, default: "Linear"
            Interpolation method for the survival curve. Options are "Linear" or "Pchip".
        """
        pred_survs = check_and_convert(pred_survs)
        time_coordinates = check_and_convert(time_coordinates)

        self.ndim_time = time_coordinates.ndim
        self.ndim_surv = pred_survs.ndim
        if self.ndim_time == 1 and self.ndim_surv == 1:
            raise TypeError(
                "At least one of 'pred_survs' or 'time_coordinates' must be a 2D array."
            )

        self._pred_survs, self._time_coordinates = zero_padding(
            pred_survs, time_coordinates
        )

        left_limits, right_limits = check_and_convert(left_limits, right_limits)
        if np.any(left_limits > right_limits):
            raise ValueError("Found an interval with left > right in the testing data.")
        self.left_limits = left_limits
        self.right_limits = right_limits

        if (train_left_limits is not None) and (train_right_limits is not None):
            train_left_limits, train_right_limits = check_and_convert(
                train_left_limits, train_right_limits
            )
            if np.any(train_left_limits > train_right_limits):
                raise ValueError(
                    "Found an interval with left > right in the training data."
                )
        self.train_left_limits = train_left_limits
        self.train_right_limits = train_right_limits

        if predict_time_method == "Median":
            self.predict_time_method = predict_median_st
        elif predict_time_method == "Mean":
            self.predict_time_method = predict_mean_st
        elif predict_time_method == "RMST":
            self.predict_time_method = predict_rmst
        else:
            error = "Please enter one of 'Median', 'Mean', or 'RMST' for calculating predicted survival time."
            raise TypeError(error)

        self.interpolation = interpolation
        self._NO_CENSOR = np.all(left_limits == right_limits)

    @cached_property
    def predicted_event_times(self):
        return self.predict_time_from_curve(self.predict_time_method)

    def _clear_cache(self):
        # See how to clear cache in functools:
        # https://docs.python.org/3/library/functools.html#functools.cached_property
        # https://stackoverflow.com/questions/62662564/how-do-i-clear-the-cache-from-cached-property-decorator
        self.__dict__.pop("predicted_event_times", None)

    def _error_trainset(self, method_name: str):
        if (self.train_left_limits is None) or (self.train_right_limits is None):
            raise TypeError(
                "Train set information is missing. "
                "Evaluator cannot perform {} evaluation.".format(method_name)
            )

    def concordance(
        self, method: str = "probabilistic", ties: str = "skip", *args, **kwargs
    ) -> tuple[float, float, float]:
        """
        Calculate the concordance index from the predicted survival curve.

        Parameters
        ----------
        method: str, default: "probabilistic"
            Method to calculate concordance index. Options are "probabilistic" and "midpoint".
        tie_strategy: str, default: "skip"
            How to handle ties in eta:
              - "skip": pairs with eta_i == eta_j contribute 0 to the numerator.
              - "half":  ties contribute 0.5 * w_{i<j} to the numerator.

        Returns
        -------
        c_index: float
            The concordance index.
        num_matrix: np.ndarray of shape (n_sample, n_sample)
            per-pair contributions to numerator,
        den_matrix: np.ndarray of shape (n_sample, n_sample)
            per-pair weights (same as weights) in denominator.
        """
        pred_times = self.predict_time_method(
            self._pred_survs, self._time_coordinates, interpolation=self.interpolation
        )
        if method == "midpoint":
            imp_times, imp_indicators = impute_times_midpoint(
                self.left_limits, self.right_limits
            )
            ties = "None" if ties == "skip" else "Risk"
            c_index, num, den = concordance(
                predicted_times=pred_times,
                event_times=imp_times,
                event_indicators=imp_indicators,
                ties=ties,
            )
            return c_index, num, den
        elif method == "probabilistic":
            c_index, num, den = concordance_ic(
                eta=-pred_times,
                left=self.left_limits,
                right=self.right_limits,
                left_train=self.train_left_limits,
                right_train=self.train_right_limits,
                ties=ties,
            )
            return c_index, np.sum(num), np.sum(den)
        else:
            raise ValueError(
                "Please enter one of 'probabilistic' or 'midpoint' for concordance method."
            )

    def brier_score(
        self,
        target_time: Optional[Numeric] = None,
        method: str = "Tsouprou-marginal",
        x: Optional[np.ndarray] = None,
        x_train: Optional[np.ndarray] = None,
    ) -> float:
        """
        Calculate the Brier score at a given time point from the predicted survival curve.

        Parameters
        ----------
        target_time: Optional[Numeric], default: None
            The time point at which to calculate the Brier score. If None, the median of the unique observed times is used.
        method: str, default: "Tsouprou-marginal"
            The method to use for calculating the Brier score. Options are "uncensored", "Tsouprou-conditional", and "Tsouprou-marginal".
        x: Optional[np.ndarray], default: None
            Covariates for the test set. Required if method is "Tsouprou-conditional".
        x_train: Optional[np.ndarray], default: None
            Covariates for the training set. Required if method is "Tsouprou-conditional".

        Returns
        -------
        brier_score: float
            The Brier score at the given time point.
        """
        # Check if there is no censored instance, if so, naive Brier score is applied
        if self._NO_CENSOR:
            method = "uncensored"

        if method in ["Tsouprou-conditional", "Tsouprou-marginal"]:
            self._error_trainset("Tsouprou Brier score")
            if method == "Tsouprou-conditional":
                if x is None or x_train is None:
                    raise TypeError(
                        "x and x_train must be provided for Tsouprou-conditional method."
                    )

        if target_time is None:
            tau_vals = np.concatenate(
                [
                    self.left_limits,
                    self.right_limits[np.isfinite(self.right_limits)],
                    self.train_left_limits,
                    self.train_right_limits[np.isfinite(self.train_right_limits)],
                ]
            )
            tau = np.unique(np.sort(tau_vals))
            target_time = np.quantile(tau, 0.5)

        predict_probs = self.predict_probability_from_curve(target_time)

        return brier_score_ic(
            preds=predict_probs,
            left_limits=self.left_limits,
            right_limits=self.right_limits,
            train_left_limits=self.train_left_limits,
            train_right_limits=self.train_right_limits,
            x=x,
            x_train=x_train,
            target_time=target_time,
            method=method,
        )

    def brier_score_multiple_points(
        self,
        target_times: np.ndarray,
        method: str = "Tsouprou-marginal",
        x: Optional[np.ndarray] = None,
        x_train: Optional[np.ndarray] = None,
    ):
        # Check if there is no censored instance, if so, naive Brier score is applied
        if self._NO_CENSOR:
            method = "uncensored"

        if method in ["Tsouprou-conditional", "Tsouprou-marginal"]:
            self._error_trainset("Tsouprou Brier score")
            if method == "Tsouprou-conditional":
                if x is None or x_train is None:
                    raise TypeError(
                        "x and x_train must be provided for Tsouprou-conditional method."
                    )

        pred_probs_mat = self.predict_multi_probabilities_from_curve(target_times)

        return brier_multiple_points_ic(
            pred_mat=pred_probs_mat,
            left_limits=self.left_limits,
            right_limits=self.right_limits,
            train_left_limits=self.train_left_limits,
            train_right_limits=self.train_right_limits,
            x=x,
            x_train=x_train,
            target_times=target_times,
            method=method,
        )

    def integrated_brier_score(
        self,
        num_points: int | None = None,
        target_times: np.ndarray | None = None,
        method: str = "uncensored",
        x: Optional[np.ndarray] = None,
        x_train: Optional[np.ndarray] = None,
        integration_method: str = "trapz",
        draw_figure: bool = False,
    ) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
        """
        Calculate the Integrated Brier Score (IBS) from the predicted survival curve.

        Parameters
        ----------
        num_points : int, optional (default=None)
            Number of evaluation time points to generate automatically.
            If provided, `target_times` must be None.
            We generate `num_points` linearly spaced times between 0 and max_target_time,
            where max_target_time is the maximum observed (event or censoring) time
            across train + test.

            If both `num_points` and `target_times` are None:
            - We try to infer `target_times` from the unique censoring times in the *test* set.
            (This matches your old behavior.)

        target_times : np.ndarray, shape = (m,), optional (default=None)
            Explicit time grid at which to evaluate the Brier score.
            If provided, `num_points` must be None. We'll integrate over exactly these times.

            NOTE: You are responsible for making sure these are sorted ascending and lie within the support of the model.

        method: str, default: "uncensored"
            The method to use for calculating the Brier score. Options are "uncensored", "Tsouprou-conditional", and "Tsouprou-marginal".
            Note: "uncensored" method automatically ignores uncertain areas.

        x: Optional[np.ndarray], default: None
            Covariates for the test set. Required if method is "Tsouprou-conditional".

        x_train: Optional[np.ndarray], default: None
            Covariates for the training set. Required if method is "Tsouprou-conditional".

        integration_method: str, default: "trapz"
            Numerical integration method. Options: "trapz" (trapezoidal), "simpson".

        draw_figure: bool, default: False
            Whether to draw the Brier score curve.

        Returns
        -------
        ibs: float
            The Integrated Brier Score.
        figure: plt.Figure
            The figure object containing the Brier score curve.
        axes: plt.Axes
            The axes object containing the Brier score curve.
        """
        # Check if there is no censored instance, if so, naive method is applied
        if self._NO_CENSOR:
            method = "uncensored"

        if method in ["Tsouprou-conditional", "Tsouprou-marginal"]:
            self._error_trainset("Tsouprou IBS")
            if method == "Tsouprou-conditional":
                if x is None or x_train is None:
                    raise TypeError(
                        "x and x_train must be provided for Tsouprou-conditional method."
                    )

                # Sanity check: cannot pass both num_points and target_times
        if (num_points is not None) and (target_times is not None):
            raise ValueError(
                "Please provide either `num_points` OR `target_times`, not both."
            )

        # Compute max_target_time from test set (and train set if available)
        if (self.train_left_limits is not None) and (
            self.train_right_limits is not None
        ):
            # max over the lefts and rights of both test and train, ignoring infs
            max_target_time = np.max(
                np.concatenate(
                    [
                        self.left_limits,
                        self.right_limits[np.isfinite(self.right_limits)],
                        self.train_left_limits,
                        self.train_right_limits[np.isfinite(self.train_right_limits)],
                    ]
                )
            )
        else:
            max_target_time = np.max(
                np.concatenate(
                    [
                        self.left_limits,
                        self.right_limits[np.isfinite(self.right_limits)],
                    ]
                )
            )

        # Case 1: user provided explicit target_times
        if target_times is not None:
            # ensure numpy array
            target_times = np.asarray(target_times, dtype=float)

            if target_times.ndim != 1:
                raise ValueError("`target_times` must be a 1D array of times.")

            if len(target_times) < 2:
                raise ValueError(
                    "`target_times` must contain at least 2 time values "
                    "to perform numerical integration."
                )

            # We assume caller gave sorted points. If not, sort them.
            if not np.all(np.diff(target_times) >= 0):
                warnings.warn("`target_times` is not sorted; sorting it now.")
                target_times = np.sort(target_times)

            # time_range is the range on which we integrate
            time_range = target_times[-1] - target_times[0]
            if time_range <= 0:
                raise ValueError(
                    "target_times must span a positive range for IBS integration."
                )

        # Case 2: user provided num_points
        elif num_points is not None:
            if num_points < 2:
                raise ValueError("`num_points` must be >= 2 to perform integration.")

            # Uniform grid from 0 to max_target_time
            target_times = np.linspace(0.0, max_target_time, num_points)
            time_range = max_target_time  # because we started at 0

        # Case 3: neither provided → infer from the left/right limits of censored test samples, excluding infs
        else:
            target_times = np.unique(
                np.concatenate(
                    [
                        self.left_limits,
                        self.right_limits[np.isfinite(self.right_limits)],
                    ]
                )
            )

            if target_times.size < 2:
                # (old behavior raised if no censor data at all)
                raise ValueError(
                    "Could not infer `target_times` from testing samples "
                    "(e.g., no/too-few test points). "
                    "Please provide `num_points` or `target_times`."
                )

            # Sort just in case (np.unique already sorts but let's be explicit)
            target_times = np.sort(target_times)

            time_range = target_times[-1] - target_times[0]

        b_scores = self.brier_score_multiple_points(
            target_times=target_times, method=method, x=x, x_train=x_train
        )
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(target_times, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are:\n", bs_dict)

        if integration_method == "trapz":
            integral_value = trapezoid(b_scores, target_times)
        elif integration_method == "simpson":
            if len(b_scores) < 3:
                # Fall back to trapezoidal rule if not enough points for Simpson's rule
                integral_value = trapezoid(b_scores, target_times)
            else:
                integral_value = simpson(y=b_scores, x=target_times)
        else:
            raise ValueError(
                f"Integration method '{integration_method}' not supported. Use 'trapz' or 'simpson'."
            )

        ibs_score = integral_value / time_range

        if draw_figure:
            fig, ax = plt.subplots()
            ax.plot(target_times, b_scores, marker="o")
            ax.set_xlabel("Time")
            ax.set_ylabel("Brier Score")
            ax.set_title("Time-dependent Brier Score Curve")
            plt.grid()
            return ibs_score, (fig, ax)
        return ibs_score

    def crps(
        self,
        num_points: int | None = None,
        target_times: np.ndarray | None = None,
    ) -> float:
        """
        Calculate the Continuous Ranked Probability Score (CRPS) from the predicted survival curve.
        It is equivalent to the Integrated Brier Score (IBS) with uncensored method.
        It is named as CRPS following the terminology in meteorology and probabilistic forecasting field.
        The description of Survival CRPS can be found in [1].

        Parameters
        ----------
        num_points : int, optional (default=None)
            Number of evaluation time points to generate automatically.
            If provided, `target_times` must be None.
            We generate `num_points` linearly spaced times between 0 and max_target_time,
            where max_target_time is the maximum observed (event or censoring) time
            across train + test.

            If both `num_points` and `target_times` are None:
            - We try to infer `target_times` from the unique censoring times in the *test* set.
            (This matches your old behavior.)

        target_times : np.ndarray, shape = (m,), optional (default=None)
            Explicit time grid at which to evaluate the Brier score.
            If provided, `num_points` must be None. We'll integrate over exactly these times.

            NOTE: You are responsible for making sure these are sorted ascending and lie within the support of the model.

        Returns
        -------
        crps: float
            The Continuous Ranked Probability Score (CRPS) for the predicted survival curve.

        References
        ----------
        [1] Avati et al., "Countdown Regression: Sharp and Calibrated Survival Predictions", UAI, 2020.
        """
        return self.integrated_brier_score(
            num_points=num_points,
            target_times=target_times,
            method="uncensored",
            x=None,
            x_train=None,
            integration_method="trapz",
            draw_figure=False,
        )

    def one_calibration(
        self,
        target_time: Numeric,
        num_bins: int = 10,
        binning_strategy: str = "C",
        method: str = "Turnbull",
        return_details: bool = False,
    ) -> tuple[float, list, list] | tuple[float, dict]:
        """
        Calculate the one calibration score at a given time point from the predicted survival curve.

        Parameters
        ----------
        target_time: Numeric
            The time point at which to calculate the one calibration score.
        num_bins: int, default: 10
            Number of bins used to calculate the one calibration score.
        binning_strategy: str
            The strategy to bin the predictions. The options are: "C" (default), and "H".
            C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
            H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
        method: str, default: "Turnbull"
            The method to handle censored patients. The options are: "Turnbull" (default), and "MidPoint".
            "MidPoint" method simply treats the midpoint of the interval as the event time, and
            uses the DN's method (Kaplan-Meier estimate of the survival function).
            "Turnbull" method uses the Turnbull estimator for the survival function
            to compute the average observed probabilities in each bin.

        Returns
        -------
        p_value: float
            The p-value of the calibration test.
        observed_probabilities: list
            The observed probabilities in each bin.
        expected_probabilities: list
            The expected probabilities in each bin.
        details: dict, optional
            A dictionary containing detailed calibration information, including:
            - p_value: The p-value of the calibration test.
            - statistics: The Hosmer-Lemeshow statistics.
            - observed_probabilities: The observed probabilities in each bin.
            - expected_probabilities: The expected probabilities in each bin.
            - slope: The slope of the calibration curve.
            - intercept: The intercept of the calibration curve.
            - max_local_deviation: The maximum local deviation between observed and expected probabilities.
            - histogram_plot: A tuple containing the figure and axes of the histogram plot.
            - pp_plot: A tuple containing the figure and axes of the P-P plot.
            Only returned when 'return_details' is set to True.

        """
        predict_probs = self.predict_probability_from_curve(target_time)
        p_value, hl_stats, obs, exp = one_cal_ic(
            preds=1 - predict_probs,
            left_limits=self.left_limits,
            right_limits=self.right_limits,
            target_time=target_time,
            num_bins=num_bins,
            binning_strategy=binning_strategy,
            method=method,
        )
        if return_details:
            # Fit least squares line
            slope, intercept = fit_least_squares(
                np.array(exp), np.array(obs), left_anchor=True, right_anchor=True
            )

            # Maximum local deviation, x is expected, y is observed
            local_devs = np.diff(obs) / np.diff(exp)
            max_local_dev = np.max(
                np.maximum(local_devs / (1 + 1e-8), (1 + 1e-8) / local_devs)
            )

            # Histogram plot
            fig1, ax1 = plt.subplots()
            bar_width = 0.4
            indices = np.arange(len(obs)) + 1
            ax1.bar(
                [x - 0.2 for x in indices],
                obs[::-1],
                width=bar_width,
                align="center",
                label="Observed",
                alpha=0.7,
            )
            ax1.bar(
                [x + 0.2 for x in indices],
                exp[::-1],
                width=bar_width,
                align="center",
                label="Expected",
                alpha=0.7,
            )
            ax1.set_xlabel("Groups (from lowest to highest predicted probability)")
            ax1.set_xticks(indices)
            ax1.set_ylabel("Probabilities")
            ax1.legend()
            fig1.tight_layout()

            fig2, ax2 = pp_plot(
                obs, exp, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), color="blue"
            )

            details = {
                "p_value": p_value,
                "statistics": hl_stats,
                "observed_probabilities": obs,
                "expected_probabilities": exp,
                "slope": slope,
                "intercept": intercept,
                "max_local_deviation": max_local_dev,
                "histogram_plot": (fig1, ax1),
                "pp_plot": (fig2, ax2),
            }
            return p_value, details
        return p_value, obs, exp

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
            Whether to return detailed calibration information.

        Returns
        -------
        p_value: float
            The p-value of the calibration test.
        hist: np.ndarray
            The histogram of the predicted probabilities in each bin.
        details: dict, optional
            A dictionary containing detailed calibration information, including:
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
        pred_probs_left = self.predict_probability_from_curve(self.left_limits)
        pred_probs_right = self.predict_probability_from_curve(self.right_limits)

        statistics, p_value, hist = d_cal_ic(
            pred_probs_left=pred_probs_left,
            pred_probs_right=pred_probs_right,
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
            slopes = d_cal_pdf[::-1] / np.diff(optimal_cdf)
            max_slope = np.max(np.maximum(slopes / (1 + 1e-8), (1 + 1e-8) / slopes))

            # horizontal histograms
            fig1, ax1 = plt.subplots()
            widths = hist
            y_positions = (optimal_cdf[:-1] + optimal_cdf[1:]) / 2  # midpoints of bins
            ax1.barh(
                y_positions,
                widths,
                height=np.diff(optimal_cdf),
                color="blue",
                alpha=0.7,
                label="Prediction",
            )
            ax1.axvline(N / num_bins, ls="dashed", c="grey", label="Ideal Calibration")
            ax1.set_xlabel("Probability Bins")
            ax1.set_ylabel("Frequency")
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
        Calculate the K-S-D calibration score from the predicted survival curve.

        Parameters
        ----------
        return_details: bool, default: False
            Whether to return detailed calibration information.

        Returns
        -------
        ks_statistic: float
            The K-S statistic of the calibration test.
        d_statistic: float
            The D statistic of the calibration test.
        details: dict, optional
            A dictionary containing detailed calibration information, including:
            - ks_statistic: The K-S statistic of the calibration test.
            - d_statistic: The D statistic of the calibration test.
            - ks_p_value: The p-value of the K-S test.
            - d_p_value: The p-value of the D test.

        """
        pred_probs_left = self.predict_probability_from_curve(self.left_limits)
        pred_probs_right = self.predict_probability_from_curve(self.right_limits)

        return ksd_cal_ic(
            pred_probs_left=pred_probs_left,
            pred_probs_right=pred_probs_right,
            return_details=return_details,
        )

    def auprc(self, n_quad: int = 256) -> float:
        """
        Calculate the Survival-AUPRC from the predicted survival curve for interval-censored data.

        Parameters
        ----------
        n_quad: int, default: 256
            Number of quadrature points to use for numerical integration.
        returns AUPRC acores for each sample.
        -------
        """
        return auprc_ic(
            pred_cdf=1 - self._pred_survs,
            time_grid=self._time_coordinates,
            left=self.left_limits,
            right=self.right_limits,
            n_quad=n_quad,
            return_details=False,
        )

    def inclusion_rate(self) -> float:
        """
        Calculate the inclusion rate of the predicted median survival times within the interval.

        Returns
        -------
        inclusion_rate: float
            The inclusion rate of the predicted median survival times within the interval.
        """
        return inclusion_rate(
            self.left_limits, self.right_limits, self.predicted_event_times
        )

    def mae(
        self,
        log_scale: bool = False,
    ) -> float:
        """
        Calculate the Mean Absolute Error (MAE) from the predicted median survival times.

        Returns
        -------
        mae: float
            The Mean Absolute Error (MAE) from the predicted median survival times.
        """
        return mean_error_ic(
            self.left_limits,
            self.right_limits,
            self.predicted_event_times,
            error_type="absolute",
            log_scale=log_scale,
        )

    def mse(
        self,
        log_scale: bool = False,
    ) -> float:
        """
        Calculate the Mean Squared Error (MSE) from the predicted median survival times.

        Returns
        -------
        mse: float
            The Mean Squared Error (MSE) from the predicted median survival times.
        """
        return mean_error_ic(
            self.left_limits,
            self.right_limits,
            self.predicted_event_times,
            error_type="squared",
            log_scale=log_scale,
        )

    def rmse(
        self,
        log_scale: bool = False,
    ) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE) from the predicted median survival times.

        Returns
        -------
        rmse: float
            The Root Mean Squared Error (RMSE) from the predicted median survival times.
        """
        return self.mse(log_scale=log_scale) ** 0.5

    def coverage(
        self,
        quantile_range: tuple[float, float] = None,
        cov_level: float = None,
        method: str = "Turnbull",
    ) -> tuple[float, float, float]:
        """
        Calculate the coverage of the predicted survival intervals.

        Parameters
        ----------
        quantile_range: tuple[float, float], default: None
            The lower and upper quantiles to define the prediction interval.
            If provided, `cov_level` must be None.
        cov_level: float, default: None
            The coverage level to define the prediction interval.
            If provided, `quantile_range` must be None.
        method: str, default: "Turnbull"
            The method to handle censored patients. Options are "Turnbull" and "linear".
        Returns
        -------
        observed_coverage : float
            The observed coverage of the prediction intervals.
        cov_gap : float
            Difference between observed coverage and the target level (observed_cov - cov_level).
        avg_width : float
            The average width of the prediction intervals.
        """
        interval_pred = self.predict_interval(
            quantile_range=quantile_range, cov_level=cov_level
        )

        return coverage_ic(
            interval_pred[:, 0],
            interval_pred[:, 1],
            self.left_limits,
            self.right_limits,
            self.train_left_limits,
            self.train_right_limits,
            cov_level=cov_level,
            method=method,
        )
