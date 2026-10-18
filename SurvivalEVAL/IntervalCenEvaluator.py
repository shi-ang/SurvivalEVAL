from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson, trapezoid

from SurvivalEVAL._BaseSurvivalEvaluator import _BaseSurvivalEvaluator
from SurvivalEVAL.Evaluations.AreaUnderPRCurve import auprc_ic
from SurvivalEVAL.Evaluations.Calibration import (
    coverage_ic,
    d_cal_ic,
    ksd_cal_ic,
    one_cal_ic,
)
from SurvivalEVAL.Evaluations.Calibration.SingleTime.util import (
    _calibration_curve_details,
    _maximum_local_deviation,
)
from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.Discrimination import concordance, concordance_ic
from SurvivalEVAL.Evaluations.Discrimination.Static.IntervalConcordance import (
    impute_times_midpoint,
)
from SurvivalEVAL.Evaluations.PredictionError import (
    brier_multiple_points_ic,
    brier_score_ic,
    inclusion_rate,
    mean_error_ic,
)
from SurvivalEVAL.Evaluations.util import (
    check_and_convert,
    fit_least_squares,
    validate_time_points,
)
from SurvivalEVAL.Evaluations.util_plots import pp_plot


class IntervalCenEvaluator(_BaseSurvivalEvaluator):
    """Evaluator for interval-censored survival data."""

    def __init__(
        self,
        pred_survs: NumericArrayLike,
        time_coordinates: NumericArrayLike,
        left_limits: NumericArrayLike,
        right_limits: NumericArrayLike,
        train_left_limits: NumericArrayLike | None = None,
        train_right_limits: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ) -> None:
        """
        Initialize the IntervalCenEvaluator with predictions and observed outcomes.

        Parameters
        ----------
        pred_survs : NumericArrayLike
            Accept shapes: (n_time_points,) or (n_samples, n_time_points).
            Predicted survival probabilities for the testing samples.
            At least one of `pred_survs` or `time_coordinates` must be a 2D array.
        time_coordinates : NumericArrayLike
            Accept shapes: (n_time_points,) or (n_samples, n_time_points).
            Time coordinates for the predicted survival probabilities.
            At least one of `pred_survs` or `time_coordinates` must be a 2D array.
        left_limits : NumericArrayLike, shape = (n_samples,)
            Finite, non-negative left limits of the interval-censored testing
            data. Use 0 for left-censored observations.
        right_limits : NumericArrayLike, shape = (n_samples,)
            Right limits of the interval-censored testing data.
        train_left_limits : NumericArrayLike | None, shape = (n_train_samples,), default=None
            Finite, non-negative left limits of the interval-censored training
            data. Use 0 for left-censored observations.
        train_right_limits : NumericArrayLike | None, shape = (n_train_samples,), default=None
            Right limits of the interval-censored data for the training set.
        predict_time_method : str, default="Median"
            Method to predict time from the survival curve. Options are "Median", "Mean", or "RMST".
        interpolation : str, default="Linear"
            Interpolation method for the survival curve. Options are "Linear" or "Pchip".
        """
        left_limits, right_limits = check_and_convert(left_limits, right_limits)
        if np.any(~np.isfinite(left_limits)) or np.any(left_limits < 0):
            raise ValueError("Testing left limits must be finite and non-negative.")
        if np.any(left_limits > right_limits):
            raise ValueError("Found an interval with left > right in the testing data.")
        self.left_limits = left_limits
        self.right_limits = right_limits
        if (train_left_limits is not None) and (train_right_limits is not None):
            train_left_limits, train_right_limits = check_and_convert(
                train_left_limits, train_right_limits
            )
            if np.any(~np.isfinite(train_left_limits)) or np.any(train_left_limits < 0):
                raise ValueError(
                    "Training left limits must be finite and non-negative."
                )
            if np.any(train_left_limits > train_right_limits):
                raise ValueError(
                    "Found an interval with left > right in the training data."
                )
        self.train_left_limits = train_left_limits
        self.train_right_limits = train_right_limits

        self._NO_CENSOR = np.all(left_limits == right_limits)
        super().__init__(
            pred_survs,
            time_coordinates,
            n_samples=self.left_limits.shape[0],
            predict_time_method=predict_time_method,
            interpolation=interpolation,
        )

    @property
    def default_target_time(self) -> float:
        """
        Median unique finite bound across available training and test intervals.

        Returns
        -------
        float
            Median of the unique finite bounds in the available training and test intervals.
        """
        bounds = [self.left_limits, self.right_limits]
        if self.train_left_limits is not None and self.train_right_limits is not None:
            bounds.extend((self.train_left_limits, self.train_right_limits))
        times = np.concatenate(bounds)
        return float(np.quantile(np.unique(times[np.isfinite(times)]), 0.5))

    def _error_trainset(self, method_name: str) -> None:
        """
        Require training outcomes for a metric that estimates censoring or event distributions.

        Parameters
        ----------
        method_name : str
            Metric name to include in the missing-training-data error.

        Raises
        ------
        TypeError
            If the training outcomes required by the metric are unavailable.
        """
        if (self.train_left_limits is None) or (self.train_right_limits is None):
            raise TypeError(
                "Train set information is missing. "
                f"Evaluator cannot perform {method_name} evaluation."
            )

    def concordance(
        self,
        method: str = "comparable",
        ties: str = "skip",
        *args: Any,
        **kwargs: Any,
    ) -> tuple[float, float, float]:
        """
        Calculate the concordance index from the predicted survival curve.

        The ``probability`` method adapts the pair-ordering approach in [1]_
        using Turnbull survival estimates. The ``midpoint`` method applies
        Harrell concordance [2]_ after midpoint imputation.

        Parameters
        ----------
        method : str, default="comparable"
            Method to calculate concordance index. Options are "comparable",
            "probability", and "midpoint".
        ties : str, default="skip"
            How to handle ties in eta:
              - "skip": pairs with eta_i == eta_j contribute 0 to the numerator.
              - "half":  ties contribute 0.5 * w_{i<j} to the numerator.
        *args : Any
            Unused positional arguments.
        **kwargs : Any
            Unused keyword arguments.

        Returns
        -------
        c_index : float
            The concordance index.
        concordant_pairs : float
            The total concordant contribution to the numerator.
        total_pairs : float
            The total comparable weight in the denominator.

        References
        ----------
        .. [1] Tsouprou, Sofia (2015). "Measures of discrimination and predictive
           accuracy for interval censored survival data". Master's thesis,
           Leiden University. https://math.leidenuniv.nl/scripties/MasterTsouprou.pdf
        .. [2] Harrell, F. E., Jr.; Lee, K. L.; Mark, D. B. (1996).
           "Multivariable prognostic models: issues in developing models, evaluating
           assumptions and adequacy, and measuring and reducing errors".
           Statistics in Medicine, 15(4), 361-387.
           https://doi.org/10.1002/(SICI)1097-0258(19960229)15:4%3C361::AID-SIM168%3E3.0.CO;2-4
        """
        pred_times = self.predicted_event_times
        method = method.lower()
        ties = ties.lower()

        if method == "midpoint":
            imp_times, imp_indicators = impute_times_midpoint(
                self.left_limits, self.right_limits
            )
            ties = "none" if ties == "skip" else "risk"
            c_index, num, den = concordance(
                predicted_times=pred_times,
                event_times=imp_times,
                event_indicators=imp_indicators,
                ties=ties,
            )
            return c_index, num, den
        elif method in ["comparable", "probability"]:
            c_index, num, den = concordance_ic(
                eta=-pred_times,
                left=self.left_limits,
                right=self.right_limits,
                left_train=self.train_left_limits,
                right_train=self.train_right_limits,
                method=method,
                ties=ties,
            )
            return c_index, np.sum(num), np.sum(den)
        else:
            raise ValueError(
                "Please enter one of 'comparable', 'probability', or 'midpoint' "
                "for concordance method."
            )

    def brier_score(
        self,
        target_time: Numeric | None = None,
        method: str = "Tsouprou-marginal",
        x: np.ndarray | None = None,
        x_train: np.ndarray | None = None,
    ) -> float:
        """
        Calculate the Brier score at a given time point from the predicted survival curve.

        The ``Tsouprou-marginal`` and ``Tsouprou-conditional`` methods adapt
        the interval-censoring approach in [1]_, using Turnbull and Weibull
        AFT survival estimates, respectively.

        Parameters
        ----------
        target_time : Numeric | None, default=None
            The time point at which to calculate the Brier score. If None, the
            median of the unique finite test interval bounds and, when
            available, training interval bounds is used.
        method : str, default="Tsouprou-marginal"
                The method to use for calculating the Brier score. Options are "uncensored", "Tsouprou-conditional", and
            "Tsouprou-marginal".
        x : ndarray | None, default=None
            Covariates for the test set. Required if method is "Tsouprou-conditional".
        x_train : ndarray | None, default=None
            Covariates for the training set. Required if method is "Tsouprou-conditional".

        Returns
        -------
        brier_score : float
            The Brier score at the given time point.

        References
        ----------
        .. [1] Tsouprou, Sofia (2015). "Measures of discrimination and predictive
           accuracy for interval censored survival data". Master's thesis,
           Leiden University. https://math.leidenuniv.nl/scripties/MasterTsouprou.pdf
        """
        # Check if there is no censored instance, if so, naive Brier score is applied
        method = method.lower()
        if self._NO_CENSOR:
            method = "uncensored"

        if method in ["tsouprou-conditional", "tsouprou-marginal"]:
            self._error_trainset("Tsouprou Brier score")
            if method == "tsouprou-conditional" and (x is None or x_train is None):
                raise TypeError(
                    "x and x_train must be provided for Tsouprou-conditional method."
                )

        if target_time is None:
            target_time = self.default_target_time

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
        x: np.ndarray | None = None,
        x_train: np.ndarray | None = None,
    ) -> np.ndarray:
        # Check if there is no censored instance, if so, naive Brier score is applied
        """
        Calculate interval-censored Brier scores at multiple target times.

        The ``Tsouprou-marginal`` and ``Tsouprou-conditional`` methods adapt
        the interval-censoring approach in [1]_, using Turnbull and Weibull
        AFT survival estimates, respectively.

        Parameters
        ----------
        target_times : ndarray
            One-dimensional array of non-negative evaluation times.
        method : str
            Brier estimator: "Uncensored", "Tsouprou-marginal", or "Tsouprou-conditional", case-insensitive.
        x : ndarray | None
            Test covariates for the "Tsouprou-conditional" method, with shape (n_samples, n_features).
        x_train : ndarray | None
            Training covariates required by the "Tsouprou-conditional" method.

        Returns
        -------
        ndarray
            One Brier score per target time, with shape (n_times,).

        References
        ----------
        .. [1] Tsouprou, Sofia (2015). "Measures of discrimination and predictive
           accuracy for interval censored survival data". Master's thesis,
           Leiden University. https://math.leidenuniv.nl/scripties/MasterTsouprou.pdf
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "uncensored"

        if method in ["tsouprou-conditional", "tsouprou-marginal"]:
            self._error_trainset("Tsouprou Brier score")
            if method == "tsouprou-conditional" and (x is None or x_train is None):
                raise TypeError(
                    "x and x_train must be provided for Tsouprou-conditional method."
                )

        target_times = validate_time_points(target_times, input_name="target_times")

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
        x: np.ndarray | None = None,
        x_train: np.ndarray | None = None,
        integration_method: str = "trapz",
        draw_figure: bool = False,
    ) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
        """
        Calculate the Integrated Brier Score (IBS) from the predicted survival curve.

        The ``Tsouprou-marginal`` and ``Tsouprou-conditional`` methods adapt
        the interval-censoring approach in [1]_, using Turnbull and Weibull
        AFT survival estimates, respectively.

        Parameters
        ----------
        num_points : int, optional, default=None
            Number of evaluation time points to generate automatically.
            If provided, `target_times` must be None.
            We generate `num_points` linearly spaced times between 0 and max_target_time,
            where max_target_time is the maximum observed (event or censoring) time
            across train + test.

            If both `num_points` and `target_times` are None:
            - We infer `target_times` from the unique finite left and right interval
              endpoints in the test set.

        target_times : ndarray, shape = (m,), optional, default=None
            Explicit time grid at which to evaluate the Brier score.
            If provided, `num_points` must be None. We'll integrate over exactly these times.
            If unsorted, values are sorted ascending with a warning. You are
            responsible for making sure they lie within the support of the model.

        method : str, default="uncensored"
                The method to use for calculating the Brier score. Options are "uncensored", "Tsouprou-conditional", and
            "Tsouprou-marginal".
            Note: "uncensored" method automatically ignores uncertain areas.

        x : ndarray | None, default=None
            Covariates for the test set. Required if method is "Tsouprou-conditional".

        x_train : ndarray | None, default=None
            Covariates for the training set. Required if method is "Tsouprou-conditional".

        integration_method : str, default="trapz"
            Numerical integration method. Options: "trapz" (trapezoidal), "simpson".

        draw_figure : bool, default=False
            Whether to draw the Brier score curve.

        Returns
        -------
        ibs : float
            The Integrated Brier Score when `draw_figure` is False.
        result : tuple[float, tuple[plt.Figure, plt.Axes]]
            When `draw_figure` is True, returns `(ibs, (fig, ax))`.

        References
        ----------
        .. [1] Tsouprou, Sofia (2015). "Measures of discrimination and predictive
           accuracy for interval censored survival data". Master's thesis,
           Leiden University. https://math.leidenuniv.nl/scripties/MasterTsouprou.pdf
        """
        # Check if there is no censored instance, if so, naive method is applied
        method = method.lower()
        if self._NO_CENSOR:
            method = "uncensored"

        if method in ["tsouprou-conditional", "tsouprou-marginal"]:
            self._error_trainset("Tsouprou IBS")
            if method == "tsouprou-conditional" and (x is None or x_train is None):
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

            # Sort explicit target times if needed, preserving the integration range.
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

        # Case 3: neither provided -> infer from finite test interval endpoints.
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
                raise ValueError(
                    "Could not infer `target_times` from testing samples "
                    "(e.g., no/too-few test points). "
                    "Please provide `num_points` or `target_times`."
                )

            # Sort just in case (np.unique already sorts but let's be explicit)
            target_times = np.sort(target_times)

            time_range = target_times[-1] - target_times[0]

        b_scores = self.brier_score_multiple_points(
            target_times=target_times,
            method=method,
            x=x,
            x_train=x_train,
        )
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(target_times, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are:\n", bs_dict)

        integration_method = integration_method.lower()
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
        The description of Survival CRPS can be found in [1]_.

        Parameters
        ----------
        num_points : int, optional, default=None
            Number of evaluation time points to generate automatically.
            If provided, `target_times` must be None.
            We generate `num_points` linearly spaced times between 0 and max_target_time,
            where max_target_time is the maximum observed (event or censoring) time
            across train + test.

            If both `num_points` and `target_times` are None:
            - We infer `target_times` from the unique finite left and right interval
              endpoints in the test set.

        target_times : ndarray, shape = (m,), optional, default=None
            Explicit time grid at which to evaluate the Brier score.
            If provided, `num_points` must be None. We'll integrate over exactly these times.
            If unsorted, values are sorted ascending with a warning. You are
            responsible for making sure they lie within the support of the model.

        Returns
        -------
        crps : float
            The Continuous Ranked Probability Score (CRPS) for the predicted survival curve.

        References
        ----------
        .. [1] Avati et al. (2020). "Countdown Regression: Sharp and Calibrated Survival Predictions".
           Proceedings of Machine Learning Research, 115, 145-155.
           https://proceedings.mlr.press/v115/avati20a.html
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
        target_time: Numeric | None = None,
        num_bins: int = 10,
        binning_strategy: str = "C",
        method: str = "Turnbull",
        return_details: bool = False,
    ) -> tuple[float, list, list] | tuple[float, dict]:
        """
        Calculate the one calibration score at a given time point from the predicted survival curve.

        The ``Turnbull`` method estimates within-group survival using [1]_.
        The ``MidPoint`` method applies the grouped DN approach [2]_ after
        midpoint imputation.

        Parameters
        ----------
        target_time : Numeric | None, default=None
            Evaluation time. None uses default_target_time, the median of unique
            finite bounds across available training and test intervals.
        num_bins : int, default=10
            Number of bins used to calculate the one calibration score.
        binning_strategy : str
            The strategy to bin the predictions. The options are: "C" (default), and "H".
            C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
            H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
        method : str, default="Turnbull"
            The method to handle censored patients. The options are: "Turnbull" (default), and "MidPoint".
            "MidPoint" method simply treats the midpoint of the interval as the event time, and
            uses the DN's method (Kaplan-Meier estimate of the survival function).
            "Turnbull" method uses the Turnbull estimator for the survival function
            to compute the average observed probabilities in each bin.
        return_details : bool, default=False
            Whether to return detailed calibration information and plots.

        Returns
        -------
        p_value : float
            The p-value of the calibration test.
        observed_probabilities : list
            The observed probabilities in each bin.
        expected_probabilities : list
            The expected probabilities in each bin.
        details : dict, optional
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

        References
        ----------
        .. [1] Turnbull, B. W. (1976). "The Empirical Distribution Function with Arbitrarily Grouped,
           Censored and Truncated Data". Journal of the Royal Statistical Society: Series B
           (Methodological), 38(3), 290-295. https://doi.org/10.1111/j.2517-6161.1976.tb01597.x
           https://academic.oup.com/jrsssb/article/38/3/290/7027379
        .. [2] D'Agostino, R. B.; Nam, B.-H. (2003). "Evaluation of the performance
           of survival analysis models: discrimination and calibration measures".
           Handbook of Statistics, 23, 1-25.
           https://doi.org/10.1016/S0169-7161(03)23001-7
        """
        if target_time is None:
            target_time = self.default_target_time

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
            details = {
                "p_value": p_value,
                "statistics": hl_stats,
                "observed_probabilities": obs,
                "expected_probabilities": exp,
            }
            details.update(_calibration_curve_details(obs, exp))
            return p_value, details
        return p_value, obs, exp

    def d_calibration(
        self,
        num_bins: int = 10,
        return_details: bool = False,
    ) -> tuple[float, np.ndarray] | tuple[float, dict]:
        """
        Calculate the D calibration score from the predicted survival curve.

        This adapts D-calibration [1]_ to survival probabilities at interval boundaries.

        Parameters
        ----------
        num_bins : int, default=10
            Number of bins used to calculate the D calibration score.
        return_details : bool, default=False
            Whether to return detailed calibration information.

        Returns
        -------
        p_value : float
            The p-value of the calibration test.
        hist : ndarray
            The histogram of the predicted probabilities in each bin.
        details : dict, optional
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

        References
        ----------
        .. [1] Haider, H.; Hoehn, B.; Davis, S.; Greiner, R. (2020).
           "Effective Ways to Build and Evaluate Individual Survival Distributions".
           Journal of Machine Learning Research, 21(85), 1-63.
           https://www.jmlr.org/papers/v21/18-772.html
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
            max_slope = _maximum_local_deviation(d_cal_cdf, optimal_cdf)

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
        self,
        return_details: bool = False,
    ) -> tuple[float, float] | tuple[float, dict]:
        """
        Calculate the K-S-D calibration score from the predicted survival curve.

        The KS calibration metric is motivated by [1]_. This interval-censored
        variant estimates the distribution of predicted probabilities with Turnbull.

        Parameters
        ----------
        return_details : bool, default=False
            Whether to return detailed calibration information.

        Returns
        -------
        p_value : float
            Asymptotic Kolmogorov-Smirnov calibration p-value.
        statistic_or_details : float or dict
            Kolmogorov-Smirnov statistic when return_details is False. Otherwise, a dictionary
            containing statistics, p_value, empirical_distribution, and figure.

        References
        ----------
        .. [1] Park, J.; Kim, D.; Kim, C.; Park, H.; Kang, S.; Kim, G. (2025).
           "KSP: Kolmogorov-Smirnov metric-based Post-Hoc Calibration for
           Survival Analysis". Advances in Neural Information Processing Systems, 38.
           https://doi.org/10.52202/085713-0827
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

        The interval-censored survival precision-recall measure follows [1]_.

        Parameters
        ----------
        n_quad : int, default=256
            Number of quadrature points to use for numerical integration.

        Returns
        -------
        auprc : float
            The mean AUPRC across all testing samples.

        References
        ----------
        .. [1] Avati et al. (2020). "Countdown Regression: Sharp and Calibrated Survival Predictions".
           Proceedings of Machine Learning Research, 115, 145-155.
           https://proceedings.mlr.press/v115/avati20a.html
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
        Calculate the inclusion rate of the configured point predictions within the observed intervals.

        Point predictions use the evaluator's `predict_time_method`, which may
        be "Median", "Mean", or "RMST".

        The fraction of predictions within the observed intervals is described in [1]_.

        Returns
        -------
        inclusion_rate : float
            Proportion of configured point predictions within the observed intervals.

        References
        ----------
        .. [1] Avati et al. (2020). "Countdown Regression: Sharp and Calibrated Survival Predictions".
           Proceedings of Machine Learning Research, 115, 145-155.
           https://proceedings.mlr.press/v115/avati20a.html
        """
        return inclusion_rate(
            self.left_limits, self.right_limits, self.predicted_event_times
        )

    def mae(
        self,
        log_scale: bool = False,
    ) -> float:
        """
        Calculate MAE from point predictions produced by the configured prediction-time method.

        This uses the interval-censoring hinge loss in [1]_; its absolute-error
        form is the ``d_out`` metric in [2]_.

        Parameters
        ----------
        log_scale : bool, default=False
            Whether to calculate errors after applying the natural logarithm to times.

        Returns
        -------
        mae : float
            The Mean Absolute Error (MAE) of the configured point predictions.

        References
        ----------
        .. [1] Shivaswamy, Chu and Jansche (2007). "A Support Vector Approach to Censored
           Targets". ICDM, 655-660.
           https://research.google/pubs/a-support-vector-approach-to-censored-targets/
        .. [2] Avati et al. (2020). "Countdown Regression: Sharp and Calibrated Survival Predictions".
           Proceedings of Machine Learning Research, 115, 145-155.
           https://proceedings.mlr.press/v115/avati20a.html
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
        Calculate MSE from point predictions produced by the configured prediction-time method.

        This uses the interval-censoring hinge loss in [1]_; its absolute-error
        form is the ``d_out`` metric in [2]_.

        Parameters
        ----------
        log_scale : bool, default=False
            Whether to calculate errors after applying the natural logarithm to times.

        Returns
        -------
        mse : float
            The Mean Squared Error (MSE) of the configured point predictions.

        References
        ----------
        .. [1] Shivaswamy, Chu and Jansche (2007). "A Support Vector Approach to Censored
           Targets". ICDM, 655-660.
           https://research.google/pubs/a-support-vector-approach-to-censored-targets/
        .. [2] Avati et al. (2020). "Countdown Regression: Sharp and Calibrated Survival Predictions".
           Proceedings of Machine Learning Research, 115, 145-155.
           https://proceedings.mlr.press/v115/avati20a.html
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
        Calculate RMSE from point predictions produced by the configured prediction-time method.

        This uses the interval-censoring hinge loss in [1]_; its absolute-error
        form is the ``d_out`` metric in [2]_.

        Parameters
        ----------
        log_scale : bool, default=False
            Whether to calculate errors after applying the natural logarithm to times.

        Returns
        -------
        rmse : float
            The Root Mean Squared Error (RMSE) of the configured point predictions.

        References
        ----------
        .. [1] Shivaswamy, Chu and Jansche (2007). "A Support Vector Approach to Censored
           Targets". ICDM, 655-660.
           https://research.google/pubs/a-support-vector-approach-to-censored-targets/
        .. [2] Avati et al. (2020). "Countdown Regression: Sharp and Calibrated Survival Predictions".
           Proceedings of Machine Learning Research, 115, 145-155.
           https://proceedings.mlr.press/v115/avati20a.html
        """
        return self.mse(log_scale=log_scale) ** 0.5

    def coverage(
        self,
        quantile_range: tuple[float, float] | None = None,
        cov_level: float | None = None,
        method: str = "Turnbull",
    ) -> tuple[float, float, float]:
        """
        Calculate the coverage of the predicted survival intervals.

        Parameters
        ----------
        quantile_range : tuple[float, float], default=None
            The lower and upper quantiles to define the prediction interval.
            If `cov_level` is also provided, it must equal the difference
            between the upper and lower quantiles.
        cov_level : float, default=None
            The coverage level to define the prediction interval.
            If `quantile_range` is also provided, it must equal the difference
            between the upper and lower quantiles.
        method : str, default="Turnbull"
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
            cov_level=(
                cov_level
                if cov_level is not None
                else (quantile_range[1] - quantile_range[0])
            ),
            method=method,
        )
