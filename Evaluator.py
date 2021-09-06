import numpy as np
import pandas as pd
import warnings
from typing import Optional, Callable
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

from Evaluations.custom_types import Numeric, NumericArrayLike
from Evaluations.util import check_and_convert
from Evaluations.util import predict_mean_survival_time, predict_median_survival_time
from Evaluations.util import predict_prob_from_curve, predict_multi_probs_from_curve

from Evaluations.Concordance import concordance
from Evaluations.BrierScore import single_brier_score, brier_multiple_points
from Evaluations.L1 import l1_loss
from Evaluations.One_Calibration import one_calibration
from Evaluations.D_Calibration import d_calibration


class BaseEvaluator:
    def __init__(
            self,
            predicted_survival_curves: NumericArrayLike,
            time_coordinates: NumericArrayLike,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None
    ):
        """
        Initialize the Evaluator
        :param predicted_survival_curves: structured array, shape = (n_samples, n_time_points)
            Predicted survival curves for the testing samples.
        :param time_coordinates: structured array, shape = (n_time_points, )
            Time coordinates for the given curves.
        :param test_event_times: structured array, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        :param test_event_indicators: structured array, shape = (n_samples, )
            Binary indicators of censoring for the testing samples
        :param train_event_times: structured array, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        :param train_event_indicators: structured array, shape = (n_train_samples, )
            Binary indicators of censoring for the training samples
        """
        self._predicted_curves = check_and_convert(predicted_survival_curves)
        self._time_coordinates = check_and_convert(time_coordinates)

        test_event_times, test_event_indicators = check_and_convert(test_event_times, test_event_indicators)
        self.event_times = test_event_times
        self.event_indicators = test_event_indicators

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        else:
            warnings.warn("Train set information is missing. Evaluator cannot perform single time Brier score, "
                          "integrated Brier score, and L1-margin loss analysis")
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

    @property
    def predicted_curves(self):
        return self._predicted_curves

    @predicted_curves.setter
    def predicted_curves(self, val):
        print("Setter called. Resetting predicted curves for this evaluator.")
        self._predicted_curves = val

    @property
    def time_coordinates(self):
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, val):
        print("Setter called. Resetting time coordinates for this evaluator.")
        self._time_coordinates = val

    def predict_time_from_curve(
            self,
            predict_method: Callable,
    ) -> np.ndarray:
        """

        :param predict_method:
        :return:
        """
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            error = "Prediction method must be 'predict_mean_survival_time' or 'predict_median_survival_time', " \
                    "got '{}' instead".format(predict_method.__name__)
            raise TypeError(error)

        predicted_times = []
        for i in range(self.predicted_curves.shape[0]):
            predicted_time = predict_method(self.predicted_curves[i, :], self.time_coordinates)
            predicted_times.append(predicted_time)
        predicted_times = np.array(predicted_times)
        return predicted_times

    def predict_probability_from_curve(
            self,
            target_time: float
    ) -> np.ndarray:
        """

        :param target_time:
        :return:
        """
        predict_probs = []
        for i in range(self.predicted_curves.shape[0]):
            predict_prob = predict_prob_from_curve(self.predicted_curves[i, :], self.time_coordinates, target_time)
            predict_probs.append(predict_prob)
        predict_probs = np.array(predict_probs)
        return predict_probs

    def predict_multi_probabilities_from_curve(
            self,
            target_times: np.ndarray
    ) -> np.ndarray:
        """

        :param target_times:
        :return:
        """
        predict_probs_mat = []
        for i in range(self.predicted_curves.shape[0]):
            predict_probs = predict_multi_probs_from_curve(self.predicted_curves[i, :], self.time_coordinates,
                                                           target_times).tolist()
            predict_probs_mat.append(predict_probs)
        predict_probs_mat = np.array(predict_probs_mat)
        return predict_probs_mat

    def concordance(
            self,
            ties: str = "None",
            predicted_time_method: str = "Median"
    ) -> (float, float, int):
        """

        :param ties: str, default = "None"
            A string indicating the way ties should be handled. Options: "None" will throw out all ties in
            survival time and all ties from risk scores. "Time" includes ties in survival time but removes ties
            in risk scores. "Risk" includes ties in risk scores but not in survival time. "All" includes all
            ties (both in survival time and in risk scores). Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        :param predicted_time_method:
        :return:
        """
        # Choose prediction method based on the input argument
        if predicted_time_method == "Median":
            predict_method = predict_median_survival_time
        elif predicted_time_method == "Mean":
            predict_method = predict_mean_survival_time
        else:
            error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
            raise TypeError(error)

        predicted_times = self.predict_time_from_curve(predict_method)
        return concordance(predicted_times, self.event_times, self.event_indicators, ties)

    def brier_score(
            self,
            target_time: float = None
    ) -> float:
        """

        :param target_time:
        :return:
        """
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise ValueError("Don't have information for training set, cannot do IPCW weighting")

        if target_time is None:
            target_time = np.quantile(np.concatenate((self.event_times, self.train_event_times)), 0.5)

        predict_probs = self.predict_probability_from_curve(target_time)

        return single_brier_score(predict_probs, self.event_times, self.event_indicators, self.train_event_times,
                                  self.train_event_indicators, target_time)

    def brier_score_multiple_points(
            self,
            target_times: np.ndarray
    ) -> np.ndarray:
        """
        Calculate multiple Brier scores at multiple specific times.
        :param target_times: float, default: None
            The specific time points for which to estimate the Brier scores.
        :return:
            Values of multiple Brier scores.
        """
        predict_probs_mat = self.predict_multi_probabilities_from_curve(target_times)

        return brier_multiple_points(predict_probs_mat, self.event_times, self.event_indicators, self.train_event_times,
                                     self.train_event_indicators, target_times)

    def integrated_brier_score(
            self,
            num_points: int = None,
            draw_figure: bool = False
    ) -> float:
        """

        :param num_points:
        :param draw_figure:
        :return:
        """
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise ValueError("Don't have information for training set, cannot do IPCW weighting")

        max_target_time = np.amax(np.concatenate((self.event_times, self.train_event_times)))

        # If number of target time is not indicated, then we use the censored times obtained from test set
        if num_points is None:
            # test_censor_status = 1 - event_indicators
            censored_times = self.event_times[self.event_indicators == 0]
            sorted_censored_times = np.sort(censored_times)
            time_points = sorted_censored_times
            time_range = np.amax(time_points) - np.amin(time_points)
        else:
            time_points = np.linspace(0, max_target_time, num_points)
            time_range = max_target_time

        # Get single brier score from multiple target times, and use trapezoidal integral to calculate ISB.
        #########################
        # Solution 1, implemented using metrics multiplication, this is geometrically faster than solution 2
        b_scores = self.brier_score_multiple_points(time_points)
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(time_points, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are".format(bs_dict))
        integral_value = trapezoid(b_scores, time_points)
        ibs_score = integral_value / time_range
        ##########################
        # Solution 2, implemented by iteratively call single_brier_score_pycox(),
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
            plt.xlabel('Time')
            plt.ylabel('Brier Score')
            plt.show()
        return ibs_score

    def l1_loss(
            self,
            method: str = "Hinge",
            log_scale: bool = False,
            predicted_time_method: str = "Median"
    ) -> float:
        """
        Calculate the L1 loss for the test set.
        :param method: string, default: "Hinge"
        :param log_scale: boolean, default: False
        :param predicted_time_method: string, default: "Median"
        :return:
            Value for the calculated L1 loss.
        """
        if predicted_time_method == "Median":
            predict_method = predict_median_survival_time
        elif predicted_time_method == "Mean":
            predict_method = predict_mean_survival_time
        else:
            error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
            raise TypeError(error)

        # get median/mean survival time from the predicted curve
        predicted_times = self.predict_time_from_curve(predict_method)
        return l1_loss(predicted_times, self.event_times, self.event_indicators, self.train_event_times,
                       self.train_event_indicators, method, log_scale)

    def one_calibration(
            self,
            target_time: Numeric,
            num_bins: int = 10,
            method: str = "DN"
    ) -> (float, list, list):
        """
        :param target_time:
        :param num_bins:
        :param method:
        :return:
        """
        predict_probs = self.predict_probability_from_curve(target_time)
        return one_calibration(predict_probs, self.event_times, self.event_indicators, target_time, num_bins, method)

    def d_calibration(
            self,
            num_bins: int = 10
    ) -> (float, np.ndarray):
        """
        :param num_bins:
        :return:
            (p-value, counts in bins)
        """
        predict_probs = []
        for i in range(self.predicted_curves.shape[0]):
            predict_prob = predict_prob_from_curve(self.predicted_curves[i, :], self.time_coordinates,
                                                   self.event_times[i])
            predict_probs.append(predict_prob)
        predict_probs = np.array(predict_probs)
        return d_calibration(predict_probs, self.event_indicators, num_bins)


class PycoxEvaluator(BaseEvaluator):
    def __init__(
            self,
            surv: pd.DataFrame,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None
    ):
        """

        :param surv: pd.DataFrame, shape = (n_time_points, n_samples)
            Predicted survival curves for the testing samples
            DataFrame index represents the time coordinates for the given curves.
            DataFrame value represents transpose of the survival probabilities.
        :param test_event_times:
        :param test_event_indicators:
        :param train_event_times:
        :param train_event_indicators:
        """
        time_coordinates = surv.index.values
        predicted_survival_curves = surv.values.T
        # Pycox models can sometimes obtain -0 as survival probabilities. Need to convert that to 0.
        predicted_survival_curves[predicted_survival_curves < 0] = 0
        super(PycoxEvaluator, self).__init__(predicted_survival_curves, time_coordinates, test_event_times,
                                             test_event_indicators, train_event_times, train_event_indicators)


class LifelinesEvaluator(PycoxEvaluator):
    def __init__(
            self,
            surv: pd.DataFrame,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None
    ):
        super(LifelinesEvaluator, self).__init__(surv, test_event_times, test_event_indicators, train_event_times,
                                                 train_event_indicators)


class ScikitSurvivalEvaluator(BaseEvaluator):
    def __init__(
            self,
            surv: np.ndarray,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None
    ):
        """

        :param surv:
        :param test_event_times:
        :param test_event_indicators:
        :param train_event_times:
        :param train_event_indicators:
        """
        time_coordinates = surv[0].x
        predict_curves = []
        for i in range(len(surv)):
            predict_curve = surv[i].y
            if False in (time_coordinates == surv[i].x):
                raise KeyError("{}-th survival curve does not have same time coordinates".format(i))
            predict_curves.append(predict_curve)
        predicted_curves = np.array(predict_curves)
        super(ScikitSurvivalEvaluator, self).__init__(predicted_curves, time_coordinates, test_event_times,
                                                      test_event_indicators, train_event_times, train_event_indicators)
