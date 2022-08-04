import numpy as np
import pandas as pd
import torch
import warnings
import rpy2.robjects as robjects
import scipy.integrate as integrate
from dataclasses import InitVar, dataclass, field

from Evaluations.custom_types import NumericArrayLike


def check_and_convert(*args):
    """ Makes sure that the given inputs are numpy arrays, list,
        tuple, panda Series, pandas DataFrames, or torch Tensors.

        Also makes sure that the given inputs have the same shape.

        Then convert the inputs to numpy array.

        Parameters
        ----------
        * args : tuple of objects
                 Input object to check / convert.

        Returns
        -------
        * result : tuple of numpy arrays
                   The converted and validated arg.

        If the input isn't numpy arrays, list or pandas DataFrames, it will
        fail and ask to provide the valid format.
    """

    result = ()
    last_length = ()
    for i, arg in enumerate(args):

        if len(arg) == 0:
            error = " The input is empty. "
            error += "Please provide at least 1 element in the array."
            raise IndexError(error)

        else:

            if isinstance(arg, np.ndarray):
                x = (arg.astype(np.double),)
            elif isinstance(arg, list):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, tuple):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, pd.Series):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, pd.DataFrame):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, torch.Tensor):
                x = (arg.cpu().numpy().astype(np.double),)
            else:
                error = """{arg} is not a valid data format. Only use 'list', 'tuple', 'np.ndarray', 'torch.Tensor', 
                        'pd.Series', 'pd.DataFrame'""".format(arg=type(arg))
                raise TypeError(error)

            if np.sum(np.isnan(x)) > 0.:
                error = "The #{} argument contains null values"
                error = error.format(i + 1)
                raise ValueError(error)

            if len(args) > 1:
                if i > 0:
                    assert x[0].shape == last_length, """Shapes between {}-th input array and 
                    {}-th input array are not consistent""".format(i - 1, i)
                result += x
                last_length = x[0].shape
            else:
                result = x[0]

    return result


def predict_prob_from_curve(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        target_time: float
) -> float:
    """
    Quote from ISDEvaluation/Evaluations/EvaluationHelperFunction.R
    We need some type of predict function for survival curves - here we build a spline to fit the survival model curve.
    This spline is the monotonic spline using the hyman filtering of the cubic Hermite spline method,
    see https://en.wikipedia.org/wiki/Monotone_cubic_interpolation. Also see help(splinefun).

    Note that we make an alteration to the method because if the last two time points
    have the same probability (y value) then the spline is constant outside of the training data.
    We need this to be a decreasing function outside the training data so instead we take the linear fit of (0,1)
    and the last time point we have (p,t*) and then apply this linear function to all points outside of our fit.
    """
    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    # predicting boundary
    max_time = float(max(times_coordinate))

    # simply calculate the slope by using the [0, 1] - [max_time, S(t|x)]
    # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # If the true event time is out of predicting boundary, then use the linear fit mentioned above;
    # Else if the true event time is in the boundary, then use the spline
    if target_time > max_time:
        # func: y = slope * x + 1, the minimum prob should be 0
        predict_probability = max(slope * target_time + 1, 0)
    else:
        predict_probability = np.array(spline(float(target_time))).item()

    return predict_probability


def predict_multi_probs_from_curve(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        target_times: NumericArrayLike
) -> np.ndarray:
    """
    Quote from ISDEvaluation/Evaluations/EvaluationHelperFunction.R
    We need some type of predict function for survival curves - here we build a spline to fit the survival model curve.
    This spline is the monotonic spline using the hyman filtering of the cubic Hermite spline method,
    see https://en.wikipedia.org/wiki/Monotone_cubic_interpolation. Also see help(splinefun).

    Note that we make an alteration to the method because if the last two time points
    have the same probability (y value) then the spline is constant outside of the training data.
    We need this to be a decreasing function outside the training data so instead we take the linear fit of (0,1)
    and the last time point we have (p,t*) and then apply this linear function to all points outside of our fit.
    """
    target_times = check_and_convert(target_times).astype(float).tolist()

    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    # predicting boundary
    max_time = float(max(times_coordinate))

    # simply calculate the slope by using the [0, 1] - [maxtime, S(t|x)]
    # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # If the true event time is out of predicting boundary, then use the linear fit mentioned above;
    # Else if the true event time is in the boundary, then use the spline
    predict_probabilities = np.array(spline(target_times))
    for i, target_time in enumerate(target_times):
        if target_time > max_time:
            predict_probabilities[i] = max(slope * target_time + 1, 0)

    return predict_probabilities


def predict_mean_survival_time(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray
):
    # If all the predicted probabilities are 1 the integral will be infinite.
    if np.all(survival_curve == 1):
        warnings.warn("All the predicted probabilities are 1, the integral will be infinite.")
        return np.inf

    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    # predicting boundary
    max_time = max(times_coordinate.tolist())

    # simply calculate the slope by using the [0, 1] - [max_time, S(t|x)]
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # zero_probability_time = min(times_coordinate[np.where(survival_curve == 0)],
    #                             max_time + (0 - np.array(spline(max_time)).item()) / slope)
    if 0 in survival_curve:
        zero_probability_time = min(times_coordinate[np.where(survival_curve == 0)])
    else:
        zero_probability_time = max_time + (0 - np.array(spline(max_time)).item()) / slope

    def _func_to_integral(time, maximum_time, slope_rate):
        return np.array(spline(time)).item() if time < maximum_time else (1 + time * slope_rate)
    # _func_to_integral = lambda time: spline(time) if time < max_time else (1 + time * slope)
    # limit controls the subdivision intervals used in the adaptive algorithm.
    # Set it to 1000 is consistent with Haider's R code
    mean_survival_time, *rest = integrate.quad(_func_to_integral, 0, zero_probability_time,
                                               args=(max_time, slope), limit=1000)
    return mean_survival_time


def predict_median_survival_time(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray
):
    # If all the predicted probabilities are 1 the integral will be infinite.
    if np.all(survival_curve == 1):
        warnings.warn("All the predicted probabilities are 1, the median survival time will be infinite.")
        return np.inf

    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline = r_splinefun(x, y, method='hyman')

    min_prob = min(spline(times_coordinate.tolist()))

    if 0.5 in survival_curve:
        median_probability_time = times_coordinate[np.where(survival_curve == 0.5)[0][0]]
    elif min_prob < 0.5:
        min_time_before_median = times_coordinate[np.where(survival_curve > 0.5)[0][-1]]
        max_time_after_median = times_coordinate[np.where(survival_curve < 0.5)[0][0]]

        prob_range = robjects.FloatVector(
            spline(np.linspace(min_time_before_median, max_time_after_median, num=1000).tolist()))
        time_range = robjects.FloatVector(np.linspace(min_time_before_median, max_time_after_median, num=1000))
        inverse_spline = r_splinefun(prob_range, time_range, method='hyman')
        # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
        median_probability_time = np.array(inverse_spline(0.5)).item()
    else:
        max_time = max(times_coordinate.tolist())
        slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)
        median_probability_time = max_time + (0.5 - np.array(spline(max_time)).item()) / slope

    return median_probability_time


def stratified_folds_survival(
        dataset: pd.DataFrame,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        number_folds: int = 5
):
    event_times, event_indicators = event_times.tolist(), event_indicators.tolist()
    assert len(event_indicators) == len(event_times)

    indicators_and_times = list(zip(event_indicators, event_times))
    sorted_idx = [i[0] for i in sorted(enumerate(indicators_and_times), key=lambda v: (v[1][0], v[1][1]))]

    folds = [[sorted_idx[0]], [sorted_idx[1]], [sorted_idx[2]], [sorted_idx[3]], [sorted_idx[4]]]
    for i in range(5, len(sorted_idx)):
        fold_number = i % number_folds
        folds[fold_number].append(sorted_idx[i])

    training_sets = [dataset.drop(folds[i], axis=0) for i in range(number_folds)]
    testing_sets = [dataset.iloc[folds[i], :] for i in range(number_folds)]

    cross_validation_set = list(zip(training_sets, testing_sets))
    return cross_validation_set


@dataclass
class KaplanMeier:
    """
    This class is borrowed from survival_evaluation package.
    """
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)
    cumulative_dens: np.array = field(init=False)
    probability_dens: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.survival_probabilities = np.empty(population_count.size)
        survival_probability = 1
        counter = 0
        for population, event_num in zip(population_count, events):
            survival_probability *= 1 - event_num / population
            self.survival_probabilities[counter] = survival_probability
            counter += 1
        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(self.cumulative_dens)
        self.probability_dens = np.append(0, self.probability_dens)

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities


@dataclass
class KaplanMeierArea(KaplanMeier):
    area_times: np.array = field(init=False)
    area_probabilities: np.array = field(init=False)
    area: np.array = field(init=False)
    km_linear_zero: float = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        super().__post_init__(event_times, event_indicators)
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        if self.survival_probabilities[-1] != 0:
            slope = (area_probabilities[-1] - 1) / area_times[-1]
            zero_survival = -1 / slope
            area_times = np.append(area_times, zero_survival)
            area_probabilities = np.append(area_probabilities, 0)

        area_diff = np.diff(area_times, 1)
        area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)
        self.km_linear_zero = -1 / ((1 - min(self.survival_probabilities))/(0 - max(self.survival_times)))

    def best_guess(self, censor_times: np.array):
        surv_prob = self.predict(censor_times)
        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )
        censor_area = (self.area_times[censor_indexes] - censor_times
                      ) * self.area_probabilities[censor_indexes - 1]
        censor_area += self.area[censor_indexes]
        return censor_times + censor_area / surv_prob

    def _km_linear_predict(self, times):
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))

        predict_prob = np.empty_like(times)
        before_last_time_idx = times <= max(self.survival_times)
        after_last_time_idx = times > max(self.survival_times)
        predict_prob[before_last_time_idx] = self.predict(times[before_last_time_idx])
        predict_prob[after_last_time_idx] = np.clip(1 + times[after_last_time_idx] * slope, a_min=0, a_max=None)
        # if time <= max(self.survival_times):
        #     predict_prob = self.predict(time)
        # else:
        #     predict_prob = max(1 + time * slope, 0)
        return predict_prob

    def _compute_best_guess(self, time: float):
        """
        Given a censor time, compute the decensor event time based on the residual mean survival time on KM curves.
        :param time:
        :return:
        """
        # Using integrate.quad from Scipy should be more accurate, but also making the program unbearably slow.
        # The compromised method uses numpy.trapz to approximate the integral using composite trapezoidal rule.
        time_range = np.linspace(time, self.km_linear_zero, 2000)
        best_guess = time + np.trapz(self._km_linear_predict(time_range), time_range) / self.predict(time)
        # best_guess = time + integrate.quad(self._km_linear_predict, time, self.km_linear_zero,
        #                                    limit=2000)[0] / self.predict(time)
        return best_guess

    def best_guess_revise(self, censor_times: np.array):
        bg_times = np.zeros_like(censor_times)
        for i in range(len(censor_times)):
            bg_times[i] = self._compute_best_guess(censor_times[i])
        return bg_times

