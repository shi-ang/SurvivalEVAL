import numpy as np
from typing import Optional

from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.util import check_and_convert, predict_rmst, predict_mean_st, predict_median_st, zero_padding
from SurvivalEVAL.Evaluations.BrierScore import brier_score_ic
from SurvivalEVAL.Evaluations.SingleTimeCalibration import one_cal_ic
from SurvivalEVAL.Evaluations.DistributionCalibration import d_cal_ic

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
        self.left_limits = left_limits
        self.right_limits = right_limits

        if (train_left_limits is not None) and (train_right_limits is not None):
            train_left_limits, train_right_limits = check_and_convert(train_left_limits, train_right_limits)
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

    def _error_trainset(self, method_name: str):
        if (self.train_left_limits is None) or (self.train_right_limits is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

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
            method: str = "Turnbull"
    ) -> (float, list, list):
        """
        Calculate the one calibration score at a given time point from the predicted survival curve.

        Parameters
        ----------
        target_time
        num_bins
        binning_strategy
        method

        Returns
        -------
        p_value: float
            The p-value of the calibration test.
        observed_probabilities: list
            The observed probabilities in each bin.
        expected_probabilities: list
            The expected probabilities in each bin.
        """
        predict_probs = self.predict_probability_from_curve(target_time)
        return one_cal_ic(
            preds=1 - predict_probs,
            left_limits=self.left_limits,
            right_limits=self.right_limits,
            target_time=target_time,
            num_bins=num_bins,
            binning_strategy=binning_strategy,
            method=method
        )


    def d_calibration(
            self,
            num_bins: int = 10
    ) -> (float, np.ndarray):
        """
        Calculate the D calibration score from the predicted survival curve.
        Parameters
        ----------
        num_bins: int, default: 10
            Number of bins used to calculate the D calibration score.

        Returns
        -------
        p_value: float
            The p-value of the calibration test.
        hist: np.ndarray
            The histogram of the predicted probabilities in each bin.
        """
        pred_probs_left = self.predict_probability_from_curve(self.left_limits)
        pred_probs_right = self.predict_probability_from_curve(self.right_limits)
        return d_cal_ic(
            pred_probs_left=pred_probs_left,
            pred_probs_right=pred_probs_right,
            num_bins=num_bins
        )


