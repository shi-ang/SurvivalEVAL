import numpy as np
import pandas as pd
import warnings
from typing import Union, Optional, Callable, Tuple
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from abc import ABC
from functools import cached_property
from lifelines.statistics import logrank_test

from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.util import (check_and_convert, predict_rmst, predict_mean_st, predict_median_st,
                                           predict_prob_from_curve, predict_multi_probs_from_curve,
                                           quantile_to_survival, zero_padding)

from SurvivalEVAL.Evaluations.Concordance import concordance
from SurvivalEVAL.Evaluations.AreaUnderROCurve import auc
from SurvivalEVAL.Evaluations.BrierScore import single_brier_score, brier_multiple_points
from SurvivalEVAL.Evaluations.MeanError import mean_error
from SurvivalEVAL.Evaluations.SingleTimeCalibration import one_calibration, integrated_calibration_index
from SurvivalEVAL.Evaluations.DistributionCalibration import d_calibration, km_calibration, residuals


class SurvivalEvaluator:
    def __init__(
            self,
            pred_survs: NumericArrayLike,
            time_coordinates: NumericArrayLike,
            event_times: NumericArrayLike,
            event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
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
            Binary indicators of censoring for the testing samples
        train_event_times: Optional[NumericArrayLike], shape = (n_train_samples, ), default: None
            Actual event/censor time for the training samples.
        train_event_indicators: Optional[NumericArrayLike], shape = (n_train_samples, ), default: None
            Binary indicators of censoring for the training samples
        predict_time_method: str, default = "Median"
            Method for calculating predicted survival time. Available options are "Median", "Mean" and "RMST".
        interpolation: str, default = "Linear"
            Method for interpolation. Available options are ['Linear', 'Pchip'].
        """
        pred_survs = check_and_convert(pred_survs)
        time_coordinates = check_and_convert(time_coordinates)

        self.ndim_time = time_coordinates.ndim
        self.ndim_surv = pred_survs.ndim
        self._pred_survs, self._time_coordinates = zero_padding(pred_survs, time_coordinates)

        event_times, event_indicators = check_and_convert(event_times, event_indicators)
        self.event_times = event_times
        self.event_indicators = event_indicators

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

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

        self._NO_CENSOR = np.all(self.event_indicators == 1)


    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

    @property
    def pred_survs(self):
        return self._pred_survs

    @pred_survs.setter
    def pred_survs(self, val: NumericArrayLike):
        print("Setter called. Resetting predicted curves for this evaluator.")
        self._pred_survs = check_and_convert(val)
        self._clear_cache()

    @property
    def time_coordinates(self):
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, val: NumericArrayLike):
        print("Setter called. Resetting time coordinates for this evaluator.")
        self._time_coordinates = check_and_convert(val)
        self._clear_cache()

    @cached_property
    def predicted_event_times(self):
        return self.predict_time_from_curve(self.predict_time_method)

    def _clear_cache(self):
        # See how to clear cache in functools:
        # https://docs.python.org/3/library/functools.html#functools.cached_property
        # https://stackoverflow.com/questions/62662564/how-do-i-clear-the-cache-from-cached-property-decorator
        self.__dict__.pop('predicted_event_times', None)

    def predict_time_from_curve(
            self,
            predict_method: Callable,
    ) -> np.ndarray:
        """
        Predict survival time from survival curves.

        Parameters
        ----------
        predict_method: Callable
            A function that takes in a survival curve and returns a predicted survival time.
            There are three build-in methods: 'predict_median_st', 'predict_mean_st', and 'predict_rmst'.
            'predict_median_st' uses the median of the survival curve as the predicted survival time.
            'predict_mean_st' uses the expected time of the survival curve as the predicted survival time.
            'predict_rmst' uses the restricted mean survival time of the survival curve as the predicted survival time.

        Returns
        -------
        predicted_times: np.ndarray, shape = (n_samples, )
            Predicted survival time for each sample.
        """
        if ((predict_method is not predict_mean_st) and (predict_method is not predict_median_st) and
                (predict_method is not predict_rmst)):
            error = "Prediction method must be 'predict_mean_st', 'predict_median_st', or 'predict_rmst'" \
                    "got '{}' instead".format(predict_method.__name__)
            raise TypeError(error)

        predicted_times = predict_method(self._pred_survs, self._time_coordinates, self.interpolation)
        return predicted_times

    def predict_probability_from_curve(
            self,
            target_time: Union[float, int, np.ndarray],
    ) -> np.ndarray:
        """
        Calculate the survival probability at a given time point from a predicted curve.

        Each predicted curve will only have one corresponding probability.
        Note that this method is different from the 'predict_multi_probabilities_from_curve' method,
        which predicts the multiple probabilities at multiple time points from a predicted curve.

        Parameters
        ----------
        target_time: Union[float, int, np.ndarray], shape = (n_samples, )
            Time point(s) at which the probability of event is to be predicted. If float or int, the same time point is
            used for all samples. If array-like, each sample will have it own target time. The length of the array must
            be the same as the number of samples.

        Returns
        -------
        predicted_probability: np.ndarray, shape = (n_samples, )
            Predicted probabilities of event at the target time point(s).
        """
        n_samples = self._pred_survs.shape[0]
        if isinstance(target_time, (float, int)):
            target_time = target_time * np.ones(n_samples, dtype=self._time_coordinates.dtype)
        elif isinstance(target_time, np.ndarray):
            assert target_time.ndim == 1, "Target time must be a 1D array"
            assert target_time.shape[0] == self._pred_survs.shape[0], "Target time must have the same length as " \
                                                                           "the number of samples"
        else:
            error = "Target time must be a float, int, or 1D array, got '{}' instead".format(type(target_time))
            raise TypeError(error)

        predict_probs = np.empty(n_samples, dtype=self._pred_survs.dtype)

        if self.ndim_surv == 2 and self.ndim_time == 1:
            for i, curve in enumerate(self._pred_survs):
                predict_probs[i] = predict_prob_from_curve(curve, self.time_coordinates, target_time[i],
                                                           self.interpolation)
        elif self.ndim_surv == 1 and self.ndim_time == 2:
            for i, times in enumerate(self._time_coordinates):
                predict_probs[i] = predict_prob_from_curve(self._pred_survs, times, target_time[i],
                                                           self.interpolation)
        elif self.ndim_surv == 2 and self.ndim_time == 2:
            for i in range(n_samples):
                predict_probs[i] = predict_prob_from_curve(self._pred_survs[i, :], self._time_coordinates[i, :],
                                                           target_time[i], self.interpolation)
        else:
            raise TypeError("Dimensional error")

        return predict_probs

    def predict_multi_probabilities_from_curve(
            self,
            target_times: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the survival probability at multiple time points from the predicted curve.

        Parameters
        ----------
        target_times: np.ndarray, shape = (n_target_times)
            Time points at which the probability of event is to be predicted.

        Returns
        -------
        prob_mat: np.ndarray, shape = (n_samples, n_target_times)
            Predicted probabilities of event at the target time points.
        """
        if self.ndim_surv == 2 and self.ndim_time == 1:
            n_samples = self._pred_survs.shape[0]
            prob_mat = np.empty((n_samples, len(target_times)), dtype=self._pred_survs.dtype)

            for i, curve in enumerate(self._pred_survs):
                prob_mat[i] = predict_multi_probs_from_curve(
                    curve, self.time_coordinates, target_times, self.interpolation
                )
        elif self.ndim_surv == 1 and self.ndim_time == 2:
            n_samples = self._time_coordinates.shape[0]
            prob_mat = np.empty((n_samples, len(target_times)), dtype=self._pred_survs.dtype)

            for i, times in enumerate(self._time_coordinates):
                prob_mat[i] = predict_multi_probs_from_curve(
                    self._pred_survs, times, target_times, self.interpolation
                )
        elif self.ndim_surv == 2 and self.ndim_time == 2:
            n_samples = self._pred_survs.shape[0]
            prob_mat = np.empty((n_samples, len(target_times)), dtype=self._pred_survs.dtype)

            for i in range(n_samples):
                prob_mat[i] = predict_multi_probs_from_curve(
                    self._pred_survs[i, :], self._time_coordinates[i, :], target_times, self.interpolation
                )
        else:
            raise TypeError("Dimensional error")

        return prob_mat

    def plot_survival_curves(
            self,
            curve_indices: Union[int, list[int]],
            color: Union[str, list[str], None] = None,
            x_lim: Union[float, tuple[float, float], None] = None,
            y_lim: Union[float, tuple[float, float], None] = None,
            x_label: str = 'Time',
            y_label: str = 'Survival probability',
            **kwargs
    ) -> (plt.Figure, plt.Axes):
        """
        Plot survival curves from the predicted survival curves.

        Parameters
        ----------
        curve_indices: Union[int, list[int]]
            Index(ces) of the curves to be plotted.
        color: Union[str, list[str], None]
            Color(s) of the curves to be plotted. If None, default color will be used.
        x_lim: Union[float, tuple[float, float], None]
            Limits for the x-axis. If None, default limits will be used.
        y_lim: Union[float, tuple[float, float], None]
            Limits for the y-axis. If None, default limits will be used.
        x_label: str
            Label for the x-axis. Default is 'Time'.
        y_label: str
            Label for the y-axis. Default is 'Survival probability'.
        kwargs: dict

        Returns
        -------
        fig, ax: tuple
            Figure and axis objects of the plot.
        """
        fig, ax = plt.subplots()
        if self.ndim_surv == 2 and self.ndim_time == 1:
            ax.plot(self._time_coordinates, self._pred_survs[curve_indices, :].T, color=color,
                    label=curve_indices, **kwargs)
        elif self.ndim_surv == 1 and self.ndim_time == 2:
            ax.plot(self._time_coordinates[curve_indices, :].T, self._pred_survs, color=color,
                    label=curve_indices, **kwargs)
        elif self.ndim_surv == 2 and self.ndim_time == 2:
            ax.plot(self._time_coordinates[curve_indices, :].T, self._pred_survs[curve_indices, :].T,
                    color=color, label=curve_indices, **kwargs)
        else:
            raise TypeError("Dimensional error")

        if y_lim is None:
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(y_lim)

        if x_lim is not None:
            ax.set_xlim(x_lim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        return fig, ax

    def plot_quantile_curves(
            self,
            curve_indices: Union[int, list[int]],
            color: Union[str, list[str], None] = None,
            x_lim: Union[float, tuple[float, float], None] = None,
            y_lim: Union[float, tuple[float, float], None] = None,
            x_label: str = 'Quantile',
            y_label: str = 'Time',
            **kwargs
    ) -> (plt.Figure, plt.Axes):
        """
        Plot quantile curves from the predicted survival curves.

        Parameters
        ----------
        curve_indices: Union[int, list[int]]
            Index(ces) of the curves to be plotted.
        color: Union[str, list[str], None]
            Color(s) of the curves to be plotted. If None, default color will be used.
        x_lim: Union[float, tuple[float, float], None]
            Limits for the x-axis. If None, default limits will be used.
        y_lim: Union[float, tuple[float, float], None]
            Limits for the y-axis. If None, default limits will be used.
        x_label: str
            Label for the x-axis. Default is 'Quantile'.
        y_label: str
            Label for the y-axis. Default is 'Time'.
        kwargs: dict

        Returns
        -------
        fig, ax: tuple
            Figure and axis objects of the plot.
        """
        pred_quan = 1 - self._pred_survs
        fig, ax = plt.subplots()
        if self.ndim_surv == 2 and self.ndim_time == 1:
            ax.plot(pred_quan[curve_indices, :].T, self._time_coordinates, color=color,
                    label=curve_indices, **kwargs)
        elif self.ndim_surv == 1 and self.ndim_time == 2:
            ax.plot(pred_quan, self._time_coordinates[curve_indices, :].T, color=color,
                    label=curve_indices, **kwargs)
        elif self.ndim_surv == 2 and self.ndim_time == 2:
            ax.plot(pred_quan[curve_indices, :].T, self._time_coordinates[curve_indices, :].T,
                    color=color, label=curve_indices, **kwargs)
        else:
            raise TypeError("Dimensional error")

        if x_lim is None:
            ax.set_xlim(0, 1)
        else:
            ax.set_xlim(x_lim)

        if y_lim is not None:
            ax.set_ylim(y_lim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        return fig, ax

    def concordance(
            self,
            ties: str = "None",
            method: str = "Harrell"
    ) -> (float, float, int):
        """
        Calculate the concordance index between the predicted survival times and the true survival times.

        Parameters
        ----------
        ties: str, default = "None"
            A string indicating the way ties should be handled.
            Options: "None" (default), "Time", "Risk", or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        method: str, default = "Harrell"
            A string indicating the method for constructing the pairs of samples.
            Options: "Harrell" (default) or "Margin"
            "Harrell": the pairs are constructed by comparing the predicted survival time of each sample with the
            event time of all other samples. The pairs are only constructed between samples with comparable
            event times. For example, if i-th sample has a censor time of 10, then the pairs are constructed by
            comparing the predicted survival time of sample i with the event time of all samples with event
            time of 10 or less.
            "Margin": the pairs are constructed between all samples. A best-guess time for the censored samples
            will be calculated and used to construct the pairs.
        :return: (float, float, int)
            The concordance index, the number of concordant pairs, and the number of total pairs.
        """
        # Choose prediction method based on the input argument
        # Check if there is no censored instance, if so, naive Brier score is applied
        if self._NO_CENSOR:
            method = "Harrell"

        if method == "Margin":
            self._error_trainset("margin concordance")

        return concordance(
            predicted_times=self.predicted_event_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties
        )

    def auc(
            self,
            target_time: Optional[Numeric] = None
    ) -> float:
        """
        Calculate the area under the ROC curve (AUC) score at a given time point from the predicted survival curve.

        Parameters
        ----------
        target_time: float, int, or None, default = None
            Time point at which the AUC score is to be calculated. If None, the AUC score is calculated at the
            median time of all the event/censor times from the training and test sets.

        Returns
        -------
        brier_score: float
            The Brier score at the target time point.
        """
        event_times = np.concatenate((self.event_times, self.train_event_times)) \
            if self.train_event_times is not None else self.event_times

        if target_time is None:
            target_time = np.quantile(event_times, 0.5)

        predict_probs = self.predict_probability_from_curve(target_time)

        return auc(
            predict_probs=predict_probs,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            target_time=target_time
        )

    def brier_score(
            self,
            target_time: Optional[Numeric] = None,
            IPCW_weighted: bool = True
    ) -> float:
        """
        Calculate the Brier score at a given time point from the predicted survival curve.

        Parameters
        ----------
        target_time: float, int, or None, default = None
            Time point at which the Brier score is to be calculated. If None, the Brier score is calculated at the
            median time of all the event/censor times from the training and test sets.
        IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.
        :return: float
            The Brier score at the target time point.
        """
        if self._NO_CENSOR:
            IPCW_weighted = False

        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Brier score (BS)")

        if target_time is None:
            target_time = np.quantile(np.concatenate((self.event_times, self.train_event_times)), 0.5)

        predict_probs = self.predict_probability_from_curve(target_time)

        return single_brier_score(
            preds=predict_probs,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            target_time=target_time,
            ipcw=IPCW_weighted
        )

    def brier_score_multiple_points(
            self,
            target_times: np.ndarray,
            IPCW_weighted: bool = True
    ) -> np.ndarray:
        """
        Calculate multiple Brier scores at multiple specific times.

        Parameters
        ----------
        target_times: np.ndarray, default: None
            The specific time points for which to estimate the Brier scores.
        IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.
        :return:
            Values of multiple Brier scores.
        """
        # Check if there is no censored instance, if so, naive Brier score is applied
        if self._NO_CENSOR:
            IPCW_weighted = False

        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Brier score (BS)")

        pred_probs_mat = self.predict_multi_probabilities_from_curve(target_times)

        return brier_multiple_points(
            pred_mat=pred_probs_mat,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            target_times=target_times,
            ipcw=IPCW_weighted
        )

    def integrated_brier_score(
            self,
            num_points: int = None,
            IPCW_weighted: bool = True,
            draw_figure: bool = False
    ) -> float:
        """
        Calculate the integrated Brier score (IBS) from the predicted survival curve.

        Parameters
        ----------
        num_points: int, default = None
            Number of points at which the Brier score is to be calculated. If None, the number of points is set to
            the number of event/censor times from the training and test sets.
        IPCW_weighted: bool, default = True
            Whether to use IPCW weighting for the Brier score.
        draw_figure: bool, default = False
            Whether to draw the figure of the IBS.
        :return: float
            The integrated Brier score.
        """
        # Check if there is no censored instance, if so, naive Brier score is applied
        if self._NO_CENSOR:
            IPCW_weighted = False

        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Integrated Brier Score (IBS)")

        max_target_time = np.max(np.concatenate((self.event_times, self.train_event_times))) if self.train_event_times \
            is not None else np.max(self.event_times)

        # If number of target time is not indicated, then we use the censored times obtained from test set
        if num_points is None:
            censored_times = self.event_times[self.event_indicators == 0]
            time_points = np.unique(censored_times)
            if time_points.size == 0:
                raise ValueError("You don't have censor data in the test set, "
                                 "please provide \"num_points\" for calculating IBS")
            else:
                time_range = np.max(time_points) - np.min(time_points)
        else:
            time_points = np.linspace(0, max_target_time, num_points)
            time_range = max_target_time

        # Get single brier score from multiple target times, and use trapezoidal integral to calculate ISB.
        #########################
        # Solution 1, implemented using metrics multiplication, this is geometrically faster than solution 2
        b_scores = self.brier_score_multiple_points(time_points, IPCW_weighted)
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(time_points, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are:\n", bs_dict)
        integral_value = trapezoid(b_scores, time_points)
        ibs_score = integral_value / time_range
        ##########################
        # (Deprecated)
        # Solution 2, implemented by iteratively calling self.brier_score(),
        # this solution is much slower than solution 1
        # b_scores = []
        # for i in range(len(time_points)):
        #     b_score = self.brier_score(time_points[i])
        #     b_scores.append(b_score)
        # b_scores = np.array(b_scores)
        # integral_value = trapezoid(b_scores, time_points)
        # ibs_score = integral_value / time_range

        # Draw the Brier score graph
        if draw_figure:
            plt.plot(time_points, b_scores, 'bo-')
            score_text = r'IBS$= {:.3f}$'.format(ibs_score)
            plt.plot([], [], ' ', label=score_text)
            plt.legend()
            plt.xlabel('Time')
            plt.ylabel('Brier Score')
            plt.show()
        return ibs_score

    def mae(
            self,
            method: str = "Hinge",
            weighted: bool = None,
            log_scale: bool = False,
            verbose: bool = False,
            truncated_time = None
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
        if self._NO_CENSOR:
            method = "Uncensored"

        if weighted is None:
            weighted = False if method == "Uncensored" or "Hinge" else True

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
            truncated_time=truncated_time
        )

    def mse(
            self,
            method: str = "Hinge",
            weighted: bool = None,
            log_scale: bool = False,
            verbose: bool = False,
            truncated_time = None
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
        if self._NO_CENSOR:
            method = "Uncensored"

        if weighted is None:
            weighted = False if method == "Uncensored" or "Hinge" else True

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
            truncated_time=truncated_time
        )

    def rmse(
            self,
            method: str = "Hinge",
            weighted: bool = None,
            log_scale: bool = False,
            verbose: bool = False,
            truncated_time = None
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
            target_time: Numeric,
            num_bins: int = 10,
            binning_strategy: str = "C",
            method: str = "DN"
    ) -> (float, list, list):
        """
        Calculate the one calibration score at a given time point from the predicted survival curve.
        Parameters
        ----------
        target_time: float, int
            Time point at which the one calibration score is to be calculated.
        num_bins: int, default: 10
            Number of bins used to calculate the one calibration score.
        binning_strategy: str
            The strategy to bin the predictions. The options are: "C" (default), and "H".
            C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
            H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
        method: string, default: "DN"
            The method used to calculate the one calibration score.
            Options: "Uncensored", or "DN" (default)

        Returns
        -------
        p_value: float
            The p-value of the calibration test.
        observed_probabilities: list
            The observed probabilities in each bin.
        expected_probabilities: list
            The expected probabilities in each bin.
        """
        if self._NO_CENSOR:
            method = "Uncensored"

        predict_probs = self.predict_probability_from_curve(target_time)
        return one_calibration(
            preds=1 - predict_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=target_time,
            num_bins=num_bins,
            binning_strategy=binning_strategy,
            method=method
        )

    def integrated_calibration_index(
            self,
            target_time: Numeric,
            knots: int = 3,
            draw_figure: Optional[bool] = True,
            figure_range: Optional[tuple] = None
    ) -> (dict, plt.figure):
        """
        Calculate the integrated one calibration index (ICI) for a given set of predictions and true event times.

        Parameters
        ----------
        target_time: Numeric
            The specific time points for which to estimate the one calibration scores.
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
        predict_probs = self.predict_probability_from_curve(target_time)
        return integrated_calibration_index(
            preds=1 - predict_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=target_time,
            knots=knots,
            draw_figure=draw_figure,
            figure_range=figure_range
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
        predict_probs = self.predict_probability_from_curve(self.event_times)
        return d_calibration(
            pred_probs=predict_probs,
            event_indicators=self.event_indicators,
            num_bins=num_bins
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
        if self._NO_CENSOR:
            method = "CoxSnell" if method in ["CoxSnell", "Modified CoxSnell-v1", "Modified CoxSnell-v2"] else method

        predict_probs = self.predict_probability_from_curve(self.event_times)
        return residuals(
            pred_probs=predict_probs,
            event_indicators=self.event_indicators,
            method=method,
            draw_figure=draw_figure
        )

    def x_calibration(
            self,
            num_bins: int = 10
    ) -> float:
        """
        Calculate the X calibration score from the predicted survival curve.
        Parameters
        ----------
        num_bins: int, default: 10
            Number of bins used to calculate the X calibration score.

        Returns
        -------
        x_cal: float
            The X calibration score, which is the sum of squared differences between the predicted and optimal
            probabilities in each bin.
        """
        _, bin_hist = self.d_calibration(num_bins)
        n_bins = bin_hist.shape[0]
        # normalize the histogram
        d_cal_pdf = bin_hist / bin_hist.sum()
        # compute the x-calibration score
        optimal = np.ones_like(d_cal_pdf) / n_bins
        x_cal = np.sum(np.square(d_cal_pdf - optimal))
        return x_cal

    def km_calibration(
            self,
            draw_figure: bool = False
    ) -> float:
        """
        Calculate the KM calibration score from the predicted survival curve.
        Parameters
        ----------
        draw_figure: bool, default: False
            Whether to draw the figure of the KM calibration curve.
        Returns
        -------
        km_cal: float
            The KM calibration score, which is the mean survival curve of the predicted survival curves.
            It is calculated by comparing the average survival curve with the Kaplan-Meier estimate of the survival
            function.
        """
        average_survival_curve = np.mean(self._pred_survs, axis=0)
        return km_calibration(
            average_survival_curve=average_survival_curve,
            time_coordinates=self.time_coordinates,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            interpolation_method=self.interpolation,
            draw_figure=draw_figure
        )

    def log_rank(
            self,
            weightings: Optional[str] = None,
            p: Optional[float] = 0,
            q: Optional[float] = 0,
    ) -> Tuple[float, float]:
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
            durations_A = self.event_times,
            durations_B = self.predicted_event_times,
            event_observed_A = self.event_indicators,
            event_observed_B = np.ones_like(self.event_indicators, dtype=bool),
            weightings=weightings,
            p=p,
            q=q
        )
        return results.p_value, results.test_statistic

class PycoxEvaluator(SurvivalEvaluator, ABC):
    def __init__(
            self,
            surv: pd.DataFrame,
            event_times: NumericArrayLike,
            event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
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
            Event indicators for the testing samples.
        train_event_times: NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        train_event_indicators: NumericArrayLike, shape = (n_samples,), optional
            Event indicators for the training samples.
        predict_time_method: string, default: "Median"
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean", and "RMST".
        interpolation: string, default: "Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        time_coordinates = surv.index.values
        predicted_survival_curves = surv.values.T
        # Pycox models can sometimes obtain -0 as survival probabilities. Need to convert that to 0.
        predicted_survival_curves[predicted_survival_curves < 0] = 0
        super(PycoxEvaluator, self).__init__(predicted_survival_curves, time_coordinates, event_times,
                                             event_indicators, train_event_times, train_event_indicators,
                                             predict_time_method, interpolation)


class LifelinesEvaluator(PycoxEvaluator, ABC):
    def __init__(
            self,
            surv: pd.DataFrame,
            event_times: NumericArrayLike,
            event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
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
            Event indicators for the testing samples.
        train_event_times: NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        train_event_indicators: NumericArrayLike, shape = (n_samples,), optional
            Event indicators for the training samples.
        predict_time_method: string, default: "Median"
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean" and "RMST".
        interpolation: string, default: "Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        super(LifelinesEvaluator, self).__init__(surv, event_times, event_indicators, train_event_times,
                                                 train_event_indicators, predict_time_method, interpolation)


class ScikitSurvivalEvaluator(SurvivalEvaluator, ABC):
    def __init__(
            self,
            surv,
            event_times: NumericArrayLike,
            event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
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
            Event indicators for the testing samples.
        train_event_times: NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        train_event_indicators: NumericArrayLike, shape = (n_samples,), optional
            Event indicators for the training samples.
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
                raise KeyError("{}-th survival curve does not have same time coordinates".format(i))
            predict_curves.append(predict_curve)
        predicted_curves = np.array(predict_curves)
        if time_coordinates[0] != 0:
            time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
            predicted_curves = np.concatenate([np.ones([len(predicted_curves), 1]), predicted_curves], 1)
        # If some survival curves are all ones, we should do something.
        if np.any(predicted_curves[:, len(time_coordinates) - 1] == 1):
            idx_need_fix = predicted_curves[:, len(time_coordinates) - 1] == 1
            max_prob_at_end = np.max(predicted_curves[~idx_need_fix,
                                                      len(time_coordinates) - 1])
            # max_prob_at_end + (1 - max_prob_at_end) * 0.9
            predicted_curves[idx_need_fix, len(time_coordinates) - 1] = max(0.1 * max_prob_at_end + 0.9, 0.99)
        super(ScikitSurvivalEvaluator, self).__init__(predicted_curves, time_coordinates, event_times,
                                                      event_indicators, train_event_times, train_event_indicators,
                                                      predict_time_method, interpolation)


DistributionEvaluator = SurvivalEvaluator   # Alias for the SurvivalEvaluator


class PointEvaluator:
    def __init__(
            self,
            pred_times: NumericArrayLike,
            event_times: NumericArrayLike,
            event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
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
            Binary indicators of censoring for the testing samples
        train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        train_event_indicators: structured array, shape = (n_train_samples, )
            Binary indicators of censoring for the training samples
        """
        self._pred_times = check_and_convert(pred_times)

        self.event_times, self.event_indicators = check_and_convert(event_times, event_indicators)

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        self._NO_CENSOR = np.all(self.event_indicators == 1)

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

    @property
    def pred_times(self):
        return self._pred_times

    @pred_times.setter
    def pred_times(self, pred_times):
        print("Setter called. Resetting pred_times.")
        self._pred_times = pred_times

    def concordance(
            self,
            ties: str = "None",
            method: str = "Harrell"
    ) -> (float, float, int):
        """
        Calculate the concordance index between the predicted survival times and the true survival times.

        Parameters
        ----------
        ties: str, default = "None"
            A string indicating the way ties should be handled.
            Options: "None" (default), "Time", "Risk", or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        method: str, default = "Harrell"
            A string indicating the method for constructing the pairs of samples.
            Options: "Harrell" (default) or "Margin"
            "Harrell": the pairs are constructed by comparing the predicted survival time of each sample with the
            event time of all other samples. The pairs are only constructed between samples with comparable
            event times. For example, if i-th sample has a censor time of 10, then the pairs are constructed by
            comparing the predicted survival time of sample i with the event time of all samples with event
            time of 10 or less.
            "Margin": the pairs are constructed between all samples. A best-guess time for the censored samples
            will be calculated and used to construct the pairs.

        Returns
        -------
        concordance_index: float
            The concordance index, which is the proportion of concordant pairs among all pairs.
        concordant_pairs: float
            The number of concordant pairs.
        total_pairs: int
            The total number of comparable pairs considered in the concordance calculation.
        """
        # Check if there is no censored instance, if so, naive Brier score is applied
        if self._NO_CENSOR:
            method = "Harrell"

        if method == "Margin":
            self._error_trainset("margin concordance")

        return concordance(
            predicted_times=self._pred_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties
        )

    def mae(
            self,
            method: str = "Hinge",
            weighted: bool = None,
            log_scale: bool = False,
            verbose: bool = False,
            truncated_time = None
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
        if self._NO_CENSOR:
            method = "Uncensored"

        if weighted is None:
            weighted = False if method == "Uncensored" or "Hinge" else True

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
            truncated_time=truncated_time
        )

    def mse(
            self,
            method: str = "Hinge",
            weighted: bool = None,
            log_scale: bool = False,
            verbose: bool = False,
            truncated_time = None
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
        if self._NO_CENSOR:
            method = "Uncensored"

        if weighted is None:
            weighted = False if method == "Uncensored" or "Hinge" else True

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
            truncated_time=truncated_time
        )

    def rmse(
            self,
            method: str = "Hinge",
            weighted: bool = None,
            log_scale: bool = False,
            verbose: bool = False,
            truncated_time = None
    ) -> float:
        """
        Calculate the root mean squared error (RMSE) score for the test set.

        Parameters
        ----------
        method: string, default: "Hinge"
            The method used to calculate the MAE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted: bool, default: None
            Whether to use weighting scheme for MAE.
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
            The MAE score for the test set.
        """
        return self.mse(method, weighted, log_scale, verbose, truncated_time) ** 0.5

    def log_rank(
            self,
            weightings: Optional[str] = None,
            p: Optional[float] = 0,
            q: Optional[float] = 0,
    ) -> Tuple[float, float]:
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
            durations_A = self.event_times,
            durations_B = self._pred_times,
            event_observed_A = self.event_indicators,
            event_observed_B = np.ones_like(self.event_indicators, dtype=bool),
            weightings=weightings,
            p=p,
            q=q
        )
        return results.p_value, results.test_statistic


class SingleTimeEvaluator:
    def __init__(
            self,
            pred_probs: NumericArrayLike,
            event_times: NumericArrayLike,
            event_indicators: NumericArrayLike,
            target_time: Union[float, int] = None,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
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
            Binary indicators of censoring for the testing samples
        target_time: float, int, or None, default = None
            Time point at which the evaluation is to be performed. If None, the target time is set to the median time
        train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        train_event_indicators: structured array, shape = (n_train_samples, )
            Binary indicators of censoring for the training samples
        """
        self._pred_probs = check_and_convert(pred_probs)
        if self._pred_probs.ndim != 1:
            raise ValueError("predicted_probs should be a 1D array-like object, "
                             "but got a {}D array-like object".format(self._pred_probs.ndim))


        self.event_times, self.event_indicators = check_and_convert(event_times, event_indicators)

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        if target_time is None:
            # set to the median time of all the event/censor times from the training and test sets
            # if train set is not provided, use test set only
            event_times = np.concatenate((self.event_times, self.train_event_times)) \
                if self.train_event_times is not None else self.event_times
            target_time = np.quantile(event_times, 0.5)
        self.target_time = target_time
        self._NO_CENSOR = np.all(self.event_indicators == 1)

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

    @property
    def pred_probs(self):
        return self._pred_probs

    @pred_probs.setter
    def pred_probs(self, pred_probs):
        print("Setter called. Resetting pred_probs.")
        self._pred_probs = pred_probs

    def auc(
        self,
    ) -> float:
        """
        Calculate the area under the ROC curve (AUC) score at a given time point from the predicted survival curve.

        Returns
        -------
        auc_score: float
            The AUC score at the target time point.
        """
        return auc(
            predict_probs=self._pred_probs,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            target_time=self.target_time
        )

    def brier_score(
        self,
        IPCW_weighted: bool = True
    ) -> float:
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
        # Check if there is no censored instance, if so, naive Brier score is applied
        if self._NO_CENSOR:
            IPCW_weighted = False

        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Brier score (BS)")
        return single_brier_score(
            preds=self._pred_probs,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            target_time=self.target_time,
            ipcw=IPCW_weighted)

    def one_calibration(
        self,
        num_bins: int = 10,
        binning_strategy: str = "C",
        method: str = "DN"
    ) -> (float, list, list):
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

        Returns
        -------
        p_value: float
            The p-value of the calibration test.
        observed_probabilities: list
            The observed probabilities in each bin.
        expected_probabilities: list
            The expected probabilities in each bin.
        """
        if self._NO_CENSOR:
            method = "Uncensored"

        return one_calibration(
            preds=1 - self._pred_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=self.target_time,
            num_bins=num_bins,
            binning_strategy=binning_strategy,
            method=method
        )

    def integrated_calibration_index(
            self,
            knots: int = 3,
            draw_figure: Optional[bool] = True,
            figure_range: Optional[tuple] = None
    ) -> (dict, plt.figure):
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
        fig: plt.figure
            A figure showing the graphical calibration curve.
        """
        return integrated_calibration_index(
            preds=1 - self._pred_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=self.target_time,
            knots=knots,
            draw_figure=draw_figure,
            figure_range=figure_range
        )


class QuantileRegEvaluator(SurvivalEvaluator):
    def __init__(
            self,
            pred_regs: NumericArrayLike,
            quantile_levels: NumericArrayLike,
            event_times: NumericArrayLike,
            event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
    ):
        """
        Initialize the quantile regression evaluator.
        Parameters
        ----------
        pred_regs: structured array,
            Accept shapes: (n_quantiles,) or (n_samples, n_quantiles).
            Predicted survival curves for the testing samples.
            At least one of `pred_regs` or `quantile_levels` must be a 2D array.
        quantile_levels: structured array,
            Accept shapes: (n_quantiles,) or (n_samples, n_quantiles).
            Time coordinates corresponding to the survival curves.
            At least one of `pred_regs` or `quantile_levels` must be a 2D array.
        event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        event_indicators: structured array, shape = (n_samples, )
            Binary indicators of censoring for the testing samples
        train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        train_event_indicators: structured array, shape = (n_train_samples, )
            Binary indicators of censoring for the training samples
        predict_time_method: str, default = "Median"
            Method for calculating predicted survival time. Available options are "Median", "Mean" or "RMST".
        interpolation: str, default = "Linear"
            Method for interpolation. Available options are ['Linear', 'Pchip'].
        """
        survival_level = 1 - quantile_levels
        super(QuantileRegEvaluator, self).__init__(survival_level, pred_regs, event_times,
                                                   event_indicators, train_event_times, train_event_indicators,
                                                   predict_time_method, interpolation)

    def km_calibration(
            self,
            draw_figure: bool = False
    ):
        """
        Calculate the KM calibration score from the predicted survival curve.

        Parameters
        ----------
        draw_figure: bool, default: False
            Whether to draw the figure of the KM calibration.
        Returns
        -------
        km_cal: float
            The KM calibration score, which is the mean survival curve of the predicted survival curves.
            It is calculated by comparing the average survival curve with the Kaplan-Meier estimate of the survival
            function.
        """
        unique_times = np.unique(self.event_times[self.event_indicators == 1])
        survival_curves = quantile_to_survival(1 - self._pred_survs, self.time_coordinates,
                                               unique_times, interpolate=self.interpolation)
        avg_surv = np.mean(survival_curves, axis=0)

        return km_calibration(
            average_survival_curve=avg_surv,
            time_coordinates=unique_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            interpolation_method=self.interpolation,
            draw_figure=draw_figure
        )
