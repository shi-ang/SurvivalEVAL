import numpy as np
from matplotlib import pyplot as plt
from typing import Optional
from functools import cached_property
from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.util import check_and_convert, predict_rmst, predict_mean_st, predict_median_st, zero_padding, fit_least_squares
from SurvivalEVAL.Evaluations.util_plots import pp_plot
from SurvivalEVAL.Evaluations.Concordance import concordance_ic, concordance, impute_times_midpoint
from SurvivalEVAL.Evaluations.BrierScore import brier_score_ic
from SurvivalEVAL.Evaluations.MeanError import cover_and_dist_ic
from SurvivalEVAL.Evaluations.SingleTimeCalibration import one_cal_ic
from SurvivalEVAL.Evaluations.DistributionCalibration import d_cal_ic
from SurvivalEVAL.Evaluations.AreaUnderPRCurve import auprc_ic


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
            interpolation: str = "Linear"
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
        self._pred_survs, self._time_coordinates = zero_padding(pred_survs, time_coordinates)

        left_limits, right_limits = check_and_convert(left_limits, right_limits)
        if np.any(left_limits > right_limits):
            raise ValueError("Found an interval with left > right in the testing data.")
        self.left_limits = left_limits
        self.right_limits = right_limits

        if (train_left_limits is not None) and (train_right_limits is not None):
            train_left_limits, train_right_limits = check_and_convert(train_left_limits, train_right_limits)
            if np.any(train_left_limits > train_right_limits):
                raise ValueError("Found an interval with left > right in the training data.")
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
        self.__dict__.pop('predicted_event_times', None)

    def _error_trainset(self, method_name: str):
        if (self.train_left_limits is None) or (self.train_right_limits is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

    def concordance(
            self,
            method: str = "probabilistic",
            ties: str = "skip",
            *args,
            **kwargs
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
            self._pred_survs,
            self._time_coordinates,
            interpolation=self.interpolation
        )
        if method == "midpoint":
            imp_times, imp_indicators = impute_times_midpoint(self.left_limits, self.right_limits)
            ties = "None" if ties == "skip" else "Risk"
            c_index, num, den = concordance(
                eta=-pred_times,
                event_times=imp_times,
                event_indicators=imp_indicators,
                tie_strategy=ties
            )
            return c_index, num, den
        elif method == "probabilistic":
            c_index, num, den = concordance_ic(
                eta=-pred_times,
                left=self.left_limits,
                right=self.right_limits,
                left_train=self.train_left_limits,
                right_train=self.train_right_limits,
                tie_strategy=ties
            )
            return c_index, np.sum(num), np.sum(den)
        else:
            raise ValueError("Please enter one of 'probabilistic' or 'midpoint' for concordance method.")

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
                    raise TypeError("x and x_train must be provided for Tsouprou-conditional method.")

        if target_time is None:
            tau_vals = np.concatenate([
                self.left_limits,
                self.right_limits[np.isfinite(self.right_limits)],
                self.train_left_limits,
                self.train_right_limits[np.isfinite(self.train_right_limits)],
            ])
            tau = np.unique(np.sort(tau_vals))
            target_time = np.quantile(tau, 0.5)

        predict_probs = self.predict_probability_from_curve(target_time)

        return brier_score_ic(
            preds=predict_probs,
            left_limits=self.left_limits,
            right_limits=self.right_limits,
            train_left_limits=self.train_left_limits,
            train_right_limits=self.train_right_limits,
            x = x,
            x_train = x_train,
            target_time = target_time,
            method = method,
        )

    def one_calibration(
            self,
            target_time: Numeric,
            num_bins: int = 10,
            binning_strategy: str = "C",
            method: str = "Turnbull",
            return_details: bool = False
    ) -> tuple[float, list, list]:
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
            method=method
        )
        if return_details:
            # Fit least squares line
            slope, intercept = fit_least_squares(np.array(exp), np.array(obs), left_anchor=True, right_anchor=True)

            # Maximum local deviation, x is expected, y is observed
            local_devs = np.diff(obs) / np.diff(exp)
            max_local_dev = np.max(np.maximum(local_devs / (1 + 1e-8), (1 + 1e-8) / local_devs))

            # Histogram plot
            fig1, ax1 = plt.subplots()
            bar_width = 0.35
            indices = np.arange(len(obs))
            ax1.bar(indices, obs, width=bar_width, label='Observed', alpha=0.7)
            ax1.bar(indices + bar_width, exp, width=bar_width, label='Expected', alpha=0.7)
            ax1.set_xlabel('Bins')
            ax1.set_xticks(indices + bar_width / 2)
            ax1.set_ylabel('Probabilities')
            ax1.legend()
            fig1.tight_layout()

            fig2, ax2 = pp_plot(obs, exp, xlim=(-0.05, 1.05), ylim=(-0.05, 1.05), color='blue')
        
            details = {
                "p_value": p_value,
                "statistics": hl_stats,
                "observed_probabilities": obs,
                "expected_probabilities": exp,
                "slope": slope,
                "intercept": intercept,
                "max_local_deviation": max_local_dev,
                "histogram_plot": (fig1, ax1),
                "pp_plot": (fig2, ax2)
            }
            return p_value, details
        return p_value, obs, exp


    def d_calibration(
            self,
            num_bins: int = 10,
            return_details: bool = False
    ) -> tuple[float, np.ndarray]:
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
            num_bins=num_bins
        )
        if return_details:
            # normalize the histogram
            N = hist.sum()
            d_cal_pdf = hist / N
            # compute the x-calibration score
            optimal = np.ones_like(d_cal_pdf) / num_bins
            x_cal = np.sum(np.square(d_cal_pdf - optimal))

            d_cal_cdf = np.cumsum(d_cal_pdf[::-1]) # so that the first number belongs to the lowest prob bin
            d_cal_cdf = np.insert(d_cal_cdf, 0, 0)
            optimal_cdf = np.linspace(0, 1, num_bins + 1)

            # LS fit of observed CDF vs expected CDF 
            slope, intercept = fit_least_squares(optimal_cdf, d_cal_cdf, left_anchor=False, right_anchor=False)

            # Local slope deviation, max slope with the highest ratio difference from 1
            slopes = d_cal_pdf[::-1] / np.diff(optimal_cdf)
            max_slope = np.max(np.maximum(slopes / (1 + 1e-8), (1 + 1e-8) / slopes))
            

            # horizontal histograms
            fig1, ax1 = plt.subplots()
            widths = hist
            y_positions = (optimal_cdf[:-1] + optimal_cdf[1:]) / 2  # midpoints of bins
            ax1.barh(y_positions, widths, height=np.diff(optimal_cdf), color='blue', alpha=0.7, label='Prediction')
            ax1.axvline(N / num_bins, ls='dashed', c='grey', label='Ideal Calibration')
            ax1.set_xlabel('Probability Bins')
            ax1.set_ylabel('Frequency')
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
                "pp_plot": (fig2, ax2)
            }
            return p_value, details

        return p_value, hist


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
        return auprc_ic(pred_cdf=1 - self._pred_survs, time_grid=self._time_coordinates,
                        left=self.left_limits, right=self.right_limits, n_quad=n_quad,
                        return_details=False)

    def coverage_and_distance(self) -> tuple[float, float]:
        """
        Calculate the proportion of predicted median survival times that fall outside the interval
        and the average distance from the predicted median survival times to the nearest interval boundary.

        Returns
        -------
        p_out: float
            The proportion of predicted median survival times that fall outside the interval.
        d_out: float
            The average distance from the predicted median survival times to the nearest interval boundary.
        """
        return cover_and_dist_ic(self.left_limits, self.right_limits, self.predicted_event_times, return_details=False)