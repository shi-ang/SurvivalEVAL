import numpy as np
import pandas as pd
import torch
import warnings
import rpy2.robjects as robjects
from dataclasses import InitVar, dataclass, field
import scipy.integrate as integrate
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator, interp1d

from Evaluations.custom_types import NumericArrayLike

r_splinefun = robjects.r['splinefun']  # extract splinefun method from R


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


def check_monotonicity(array: NumericArrayLike):
    array = check_and_convert(array)
    if array.ndim == 1:
        return (all(array[i] <= array[i + 1] for i in range(len(array) - 1)) or
                all(array[i] >= array[i + 1] for i in range(len(array) - 1)))
    elif array.ndim == 2:
        return (all(all(array[i] <= array[i + 1]) for i in range(len(array) - 1)) or
                all(all(array[i] >= array[i + 1]) for i in range(len(array) - 1)))
    else:
        raise ValueError("The input array must be 1-D or 2-D.")


def make_monotonic(
        survival_curves: np.ndarray,
        times_coordinate: np.ndarray,
        method: str = "ceil",
        seed: int = None,
        num_bs: int = None
):
    """
    Make the survival curves monotonic.
    Parameters
    ----------
    survival_curves: np.ndarray
        Survival curves. 2-D array of survival probabilities. The first dimension is the number of samples. The second
        dimension is the number of time points.
    times_coordinate: np.ndarray
        Time points corresponding to the survival curves. 1-D array of time points.
    method: str
        The method to make the survival curves monotonic. One of ['ceil', 'floor', 'bootstrap']. Default: 'ceil'.
    seed: int
        Random seed for bootstrapping. Default: None.
    num_bs: int
        Number of bootstrap samples. Default: None. If None, then num_bs = 10 * num_times.

    Returns
    -------
    survival_curves: np.ndarray
        Survival curves with monotonicity. 2-D array of survival probabilities.
    """
    if num_bs is None and method == "bootstrap":
        # 10 times the number of time points or 1000, whichever is larger
        num_bs = max(10 * len(times_coordinate), 1000)

    survival_curves = np.clip(survival_curves, 0, 1)
    for i in range(survival_curves.shape[0]):
        if not check_monotonicity(survival_curves[i]):
            # if not, then make it monotonic
            if method == "ceil":
                survival_curves[i] = np.maximum.accumulate(survival_curves[i][::-1])[::-1]
            elif method == "floor":
                survival_curves[i] = np.minimum.accumulate(survival_curves[i])
            elif method == "bootstrap":
                # This is called "bootstrapped rearrangement" method, based on
                # Chernozhukov et al. (2010) Quantile and Probability Curves Without Crossing, Econometrica
                # The original method is for quantile curves, but we make it adaptive to survival curves.
                if seed:
                    np.random.seed(seed)

                inter_lin = interp1d(survival_curves[i], times_coordinate, kind='linear', fill_value='extrapolate')
                # Bootstrap the quantile function
                bootstrap_qf = inter_lin(np.random.uniform(0, 1, num_bs))
                # Now compute the rearranged survival curve
                # The original method is to compute a value (time) given the fixed quantile (probability)
                # Here we compute the probability (quantile) given the fixed value (time)
                for j, time in enumerate(times_coordinate):
                    survival_curves[i, j] = np.mean(bootstrap_qf > time)
            else:
                raise ValueError("method must be one of ['ceil', 'floor', 'bootstrap']")
    return survival_curves


def interpolated_survival_curve(times_coordinate, survival_curve, interpolation):
    if interpolation == "Linear":
        spline = interp1d(times_coordinate, survival_curve, kind='linear', fill_value='extrapolate')
    elif interpolation == "Pchip":
        spline = PchipInterpolator(times_coordinate, survival_curve)
    elif interpolation == "Hyman":
        x = robjects.FloatVector(times_coordinate)
        y = robjects.FloatVector(survival_curve)
        spline = r_splinefun(x, y, method='hyman')
    else:
        raise ValueError("interpolation must be one of ['Linear', 'Pchip', 'Hyman']")
    return spline


def predict_prob_from_curve(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        target_time: float,
        interpolation: str = 'Hyman'
) -> float:
    """
    Predict the probability of survival at a given time point from the survival curve. The survival curve is
    interpolated using the specified interpolation method ('Pchip' or 'Hyman'). If the target time is outside the
    range of the survival curve, the probability is extrapolated by the linear function of (0, 1) and the last time
    point.

    Parameters
    ----------
    survival_curve: np.ndarray
        Survival curve. 1-D array of survival probabilities.
    times_coordinate: np.ndarray
        Time points corresponding to the survival curve. 1-D array of time points.
    target_time: float
        Time point at which to predict the probability of survival.
    interpolation: str
        The monotonic cubic interpolation method. One of ['Linear', 'Pchip', 'Hyman']. Default: 'Hyman'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.
        If 'Hyman', use the splinefun method from R with method='hyman'.

    Returns
    -------
    predict_probability: float
        Predicted probability of survival at the target time point.
    """
    spline = interpolated_survival_curve(times_coordinate, survival_curve, interpolation)

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
        target_times: NumericArrayLike,
        interpolation: str = 'Hyman'
) -> np.ndarray:
    """
    Predict the probability of survival at multiple time points from the survival curve. The survival curve is
    interpolated using the specified interpolation method ('Pchip' or 'Hyman'). If the target time is outside the
    range of the survival curve, the probability is extrapolated by the linear function of (0, 1) and the last time.

    Parameters
    ----------
    survival_curve: np.ndarray
        Survival curve. 1-D array of survival probabilities.
    times_coordinate: np.ndarray
        Time points corresponding to the survival curve. 1-D array of time points.
    target_times: NumericArrayLike
        Time points at which to predict the probability of survival.
    interpolation: str
        The monotonic cubic interpolation method. One of ['Linear', 'Pchip', 'Hyman']. Default: 'Hyman'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.
        If 'Hyman', use the splinefun method from R with method='hyman'.

    Returns
    -------
    predict_probabilities: np.ndarray
        Predicted probabilities of survival at the target time points.
    """
    target_times = check_and_convert(target_times).astype(float).tolist()

    spline = interpolated_survival_curve(times_coordinate, survival_curve, interpolation)

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
        times_coordinate: np.ndarray,
        interpolation: str = "Hyman"
) -> float:
    """
    Get the mean survival time from the survival curve. The mean survival time is defined as the area under the survival
    curve. The curve is first interpolated by the given monotonic cubic interpolation method (Pchip or Hyman). Then the
    curve gets extroplated by the linear function of (0, 1) and the last time point. The area is calculated by the
    trapezoidal rule.
    Parameters
    ----------
    survival_curve: np.ndarray
        The survival curve of the sample. 1-D array.
    times_coordinate: np.ndarray
        The time coordinate of the survival curve. 1-D array.
    interpolation: str
        The monotonic cubic interpolation method. One of ['Linear', 'Pchip', 'Hyman']. Default: 'Hyman'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.
        If 'Hyman', use the splinefun method from R with method='hyman'.
    Returns
    -------
    mean_survival_time: float
        The mean survival time.
    """
    # If all the predicted probabilities are 1 the integral will be infinite.
    if np.all(survival_curve == 1):
        warnings.warn("All the predicted probabilities are 1, the integral will be infinite.")
        return np.inf

    spline = interpolated_survival_curve(times_coordinate, survival_curve, interpolation)

    # predicting boundary
    max_time = float(max(times_coordinate))

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
        times_coordinate: np.ndarray,
        interpolation: str = "Hyman"
) -> float:
    """
    Get the median survival time from the survival curve. The median survival time is defined as the time point where
    the survival curve crosses 0.5. The curve is first interpolated by the given monotonic cubic interpolation method
    (Pchip or Hyman). Then the curve gets extroplated by the linear function of (0, 1) and the last time point. The
    median survival time is calculated by finding the time point where the survival curve crosses 0.5.
    Parameters
    ----------
    survival_curve: np.ndarray
        The survival curve of the sample. 1-D array.
    times_coordinate: np.ndarray
        The time coordinate of the survival curve. 1-D array.
    interpolation: str
        The monotonic cubic interpolation method. One of ['Linear', 'Pchip', 'Hyman']. Default: 'Hyman'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.
        If 'Hyman', use the splinefun method from R with method='hyman'.
    Returns
    -------
    median_survival_time: float
        The median survival time.
    """
    # If all the predicted probabilities are 1 the integral will be infinite.
    if np.all(survival_curve == 1):
        warnings.warn("All the predicted probabilities are 1, the median survival time will be infinite.")
        return np.inf

    min_prob = float(min(survival_curve))

    if 0.5 in survival_curve:
        median_probability_time = times_coordinate[np.where(survival_curve == 0.5)[0][0]]
    elif min_prob < 0.5:
        idx_before_median = np.where(survival_curve > 0.5)[0][-1]
        idx_after_median = np.where(survival_curve < 0.5)[0][0]
        min_time_before_median = times_coordinate[idx_before_median]
        max_time_after_median = times_coordinate[idx_after_median]

        if interpolation == "Linear":
            # given last time before median and first time after median, solve the linear equation
            slope = ((survival_curve[idx_after_median] - survival_curve[idx_before_median]) /
                     (max_time_after_median - min_time_before_median))
            intercept = survival_curve[idx_before_median] - slope * min_time_before_median
            median_probability_time = (0.5 - intercept) / slope
        elif interpolation == "Pchip":
            # reverse the array because the PchipInterpolator requires the x to be strictly increasing
            spline = interpolated_survival_curve(times_coordinate, survival_curve, interpolation)
            time_range = np.linspace(min_time_before_median, max_time_after_median, num=1000)
            prob_range = spline(time_range)
            inverse_spline = PchipInterpolator(prob_range[::-1], time_range[::-1])
            median_probability_time = np.array(inverse_spline(0.5)).item()
        elif interpolation == "Hyman":
            spline = interpolated_survival_curve(times_coordinate, survival_curve, interpolation)
            time_range = robjects.FloatVector(np.linspace(min_time_before_median, max_time_after_median, num=1000))
            prob_range = robjects.FloatVector(spline(time_range))
            inverse_spline = r_splinefun(prob_range, time_range, method='hyman')
            median_probability_time = np.array(inverse_spline(0.5)).item()
        else:
            raise ValueError("interpolation should be one of ['Pchip', 'Hyman']")
        # Need to convert the R floatvector to numpy array and use .item() to obtain the single value
    else:
        max_time = float(max(times_coordinate))
        min_prob = float(min(survival_curve))
        slope = (1 - min_prob) / (0 - max_time)
        median_probability_time = - 0.5 / slope
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


def km_mean(
        times: np.ndarray,
        survival_probabilities: np.ndarray
) -> float:
    """
    Calculate the mean of the Kaplan-Meier curve.

    Parameters
    ----------
    times: np.ndarray, shape = (n_samples, )
        Survival times for KM curve of the testing samples
    survival_probabilities: np.ndarray, shape = (n_samples, )
        Survival probabilities for KM curve of the testing samples

    Returns
    -------
    The mean of the Kaplan-Meier curve.
    """
    # calculate the area under the curve for each interval
    area_probabilities = np.append(1, survival_probabilities)
    area_times = np.append(0, times)
    km_linear_zero = -1 / ((area_probabilities[-1] - 1) / area_times[-1])
    if survival_probabilities[-1] != 0:
        area_times = np.append(area_times, km_linear_zero)
        area_probabilities = np.append(area_probabilities, 0)
    area_diff = np.diff(area_times, 1)
    # we are using trap rule
    average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
    area = np.flip(np.flip(area_diff * average_probabilities).cumsum())
    area = np.append(area, 0)

    # calculate the mean
    surv_prob = get_prob_at_zero(times, survival_probabilities)
    return area[0] / surv_prob


def get_prob_at_zero(
        times: np.ndarray,
        survival_probabilities: np.ndarray
) -> float:
    """
    Get the survival probability at time 0. Note that this function doesn't consider the interpolation.

    Parameters
    ----------
    times: np.ndarray, shape = (n_samples, )
        Survival times for KM curve of the testing samples
    survival_probabilities: np.ndarray, shape = (n_samples, )
        Survival probabilities for KM curve of the testing samples

    Returns
    -------
    The survival probability at time 0.
    """
    probability_index = np.digitize(0, times)
    probability = np.append(1, survival_probabilities)[probability_index]

    return probability


@dataclass
class KaplanMeier:
    """
    This class is borrowed from survival_evaluation package.
    """
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    population_count: np.array = field(init=False)
    events: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)
    cumulative_dens: np.array = field(init=False)
    probability_dens: np.array = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        self.population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        self.events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        event_ratios = 1 - self.events / self.population_count
        self.survival_probabilities = np.cumprod(event_ratios)
        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

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
        self.km_linear_zero = -1 / ((area_probabilities[-1] - 1) / area_times[-1])
        if self.survival_probabilities[-1] != 0:
            area_times = np.append(area_times, self.km_linear_zero)
            area_probabilities = np.append(area_probabilities, 0)

        # we are facing the choice of using the trapzoidal rule or directly using the area under the step function
        # we choose to use trapz because it is more accurate
        area_diff = np.diff(area_times, 1)
        average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
        area = np.flip(np.flip(area_diff * average_probabilities).cumsum())
        # area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)

    @property
    def mean(self):
        return self.best_guess(np.array([0])).item()

    def best_guess(self, censor_times: np.array):
        # calculate the slope by using the [0, 1] - [max_time, S(t|x)]
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))
        # if after the last time point, then the best guess is the linear function
        before_last_idx = censor_times <= max(self.survival_times)
        after_last_idx = censor_times > max(self.survival_times)
        surv_prob = np.empty_like(censor_times).astype(float)
        # do not use np.clip(min=0) here because we will use surv_prob as the denominator,
        # and we don't want to divide by 0. The nominator will be 0 anyway.
        surv_prob[after_last_idx] = 1 + censor_times[after_last_idx] * slope
        surv_prob[before_last_idx] = self.predict(censor_times[before_last_idx])

        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )

        # for those beyond the end point, censor_area = 0
        beyond_idx = censor_indexes > len(self.area_times) - 2
        censor_area = np.zeros_like(censor_times).astype(float)
        # trapzoidal rule:  (x1 - x0) * (f(x0) + f(x1)) * 0.5
        censor_area[~beyond_idx] = ((self.area_times[censor_indexes[~beyond_idx]] - censor_times[~beyond_idx]) *
                                    (self.area_probabilities[censor_indexes[~beyond_idx]] + surv_prob[~beyond_idx])
                                    * 0.5)
        censor_area[~beyond_idx] += self.area[censor_indexes[~beyond_idx]]
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

    def _compute_best_guess(self, time: float, restricted: bool = False):
        """
        Given a censor time, compute the decensor event time based on the residual mean survival time on KM curves.
        :param time:
        :return:
        """
        # Using integrate.quad from Scipy should be more accurate, but also making the program unbearably slow.
        # The compromised method uses numpy.trapz to approximate the integral using composite trapezoidal rule.
        warnings.warn("This method is deprecated. Use best_guess instead.", DeprecationWarning)
        if restricted:
            last_time = max(self.survival_times)
        else:
            last_time = self.km_linear_zero
        time_range = np.linspace(time, last_time, 2000)
        if self.predict(time) == 0:
            best_guess = time
        else:
            best_guess = time + np.trapz(self._km_linear_predict(time_range), time_range) / self.predict(time)

        return best_guess

    def best_guess_revise(self, censor_times: np.array, restricted: bool = False):
        warnings.warn("This method is deprecated. Use best_guess instead.", DeprecationWarning)
        bg_times = np.zeros_like(censor_times)
        for i in range(len(censor_times)):
            bg_times[i] = self._compute_best_guess(censor_times[i], restricted=restricted)
        return bg_times


if __name__ == "__main__":
    #%% Test the interpolation functions
    from scipy.interpolate import CubicSpline, interp1d
    import matplotlib.pyplot as plt

    times_coordinate = np.array([0, 5, 8, 10, 25, 30, 50])
    survival_curve = np.array([1, 0.9, 0.88, 0.85, 0.7, 0.6, 0.4])

    # print(predict_median_survival_time(survival_curve, times_coordinate, 'Pchip'))
    linear = interp1d(times_coordinate, survival_curve)
    cs = CubicSpline(times_coordinate, survival_curve)
    pchip = PchipInterpolator(times_coordinate, survival_curve)
    x = robjects.FloatVector(times_coordinate)
    y = robjects.FloatVector(survival_curve)
    r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
    spline_monoh = r_splinefun(x, y, method='monoH.FC')
    spline_hyman = r_splinefun(x, y, method='hyman')

    times = np.linspace(0, 50, 100)
    plt.plot(times, linear(times), label='Linear')
    plt.plot(times, cs(times), label='Py CubicSpline')
    plt.plot(times, pchip(times), label='Py Pchip')
    plt.plot(times, spline_monoh(robjects.FloatVector(times)), label='R monoh')
    plt.plot(times, spline_hyman(robjects.FloatVector(times)), label='R hyman')
    plt.plot(times_coordinate, survival_curve, 'o', label='Data')

    plt.legend()
    plt.show()

    #%% Test the speed of interpolation function
    import datetime
    import numpy as np
    from scipy.interpolate import interp1d

    times_coordinate = np.array([0, 1, 5, 8, 10, 25, 30, 50, 100])
    times = np.linspace(0, 100, 1000)
    # randomize the survival curves
    pdf_pre = np.random.rand(len(times_coordinate), 100)
    pdf = pdf_pre / pdf_pre.sum(axis=0)
    cdf = pdf.cumsum(axis=0)
    cdf = np.insert(cdf, 0, 0, axis=0)
    survival_curve = (1 - cdf).clip(min=0, max=1)
    survival_curve = survival_curve[:-1, ]

    # test the speed of linear interpolate
    start = datetime.datetime.now()
    linear = np.empty((100, 1000))
    for idx in range(survival_curve.shape[1]):
        intp = interp1d(times_coordinate, survival_curve[:, idx])
        linear[idx, :] = intp(times)
        # linear[i, :] = np.interp(times, times_coordinate, survival_curve[:, i])
    end = datetime.datetime.now()
    print('Linear interpolation takes {} seconds'.format((end - start).total_seconds()))

    # test the speed of pchip spline
    start = datetime.datetime.now()
    pchip = np.empty((100, 1000))
    for idx in range(survival_curve.shape[1]):
        intp = PchipInterpolator(times_coordinate, survival_curve[:, idx])
        pchip[idx, :] = intp(times)
    end = datetime.datetime.now()
    print('Pchip interpolation takes {} seconds'.format((end - start).total_seconds()))

    # test the speed of Hyman spline
    start = datetime.datetime.now()
    hyman = np.empty((100, 1000))
    for idx in range(survival_curve.shape[1]):
        x = robjects.FloatVector(times_coordinate)
        y = robjects.FloatVector(survival_curve[:, idx])
        r_splinefun = robjects.r['splinefun']  # extract splinefun method from R
        spline_hyman = r_splinefun(x, y, method='hyman')
        hyman[idx, :] = spline_hyman(robjects.FloatVector(times))
    end = datetime.datetime.now()
    print('Hyman interpolation takes {} seconds'.format((end - start).total_seconds()))
