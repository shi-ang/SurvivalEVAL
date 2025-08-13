import numpy as np
import pandas as pd
import torch
import warnings
from typing import Union
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator, interp1d

from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike


def check_and_convert(*args):
    """
    Makes sure that the given inputs are numpy arrays, list, tuple, panda Series, pandas DataFrames, or torch Tensors.

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
        return (all(all(array[:, i] <= array[:, i + 1]) for i in range(array.shape[1] - 1)) or
                all(all(array[:, i] >= array[:, i + 1]) for i in range(array.shape[1] - 1)))
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
    if np.all(np.sort(times_coordinate) != times_coordinate):
        raise ValueError("The time coordinates must be sorted in ascending order.")

    if num_bs is None:
        # 10 times the number of time points or 1000, whichever is larger
        num_bs = max(10 * len(times_coordinate), 1000)

    if seed is not None:
        np.random.seed(seed)

    survival_curves = np.clip(survival_curves, 0, 1)
    if not check_monotonicity(survival_curves):
        if method == "ceil":
            survival_curves = np.maximum.accumulate(survival_curves[:, ::-1], axis=1)[:, ::-1]
        elif method == "floor":
            survival_curves = np.minimum.accumulate(survival_curves, axis=1)
        elif method == "bootstrap":
            need_rearrange = np.where(np.any((np.sort(survival_curves, axis=1)[:, ::-1] != survival_curves), axis=1))[0]

            for i in need_rearrange:
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
    else:
        raise ValueError("interpolation must be one of ['Linear', 'Pchip']")
    return spline


def predict_prob_from_curve(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        target_time: float,
        interpolation: str = 'Linear'
) -> float:
    """
    Predict the probability of survival at a given time point from the survival curve. The survival curve is
    interpolated using the specified interpolation method ('Linear' or 'Pchip'). If the target time is outside the
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
        The monotonic cubic interpolation method. One of ['Linear', 'Pchip']. Default: 'Linear'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.

    Returns
    -------
    predict_probability: float
        Predicted probability of survival at the target time point.
    """
    spline = interpolated_survival_curve(times_coordinate, survival_curve, interpolation)

    # predicting boundary
    max_time = float(max(times_coordinate))

    # simply calculate the slope by using the [0, 1] - [max_time, S(t|x)]
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
        interpolation: str = 'Linear'
) -> np.ndarray:
    """
    Predict the probability of survival at multiple time points from the survival curve. The survival curve is
    interpolated using the specified interpolation method ('Linear' or 'Pchip'). If the target time is outside the
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
        The monotonic cubic interpolation method. One of ['Linear', 'Pchip']. Default: 'Linear'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.

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
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # If the true event time is out of predicting boundary, then use the linear fit mentioned above;
    # Else if the true event time is in the boundary, then use the spline
    predict_probabilities = np.array(spline(target_times))
    for i, target_time in enumerate(target_times):
        if target_time > max_time:
            predict_probabilities[i] = max(slope * target_time + 1, 0)

    return predict_probabilities


def _check_dim_align(
        survival_curves: np.ndarray,
        times_coordinates: np.ndarray
) -> None:
    # check dimension alignment
    ndim_surv = survival_curves.ndim
    ndim_time = times_coordinates.ndim

    if ndim_surv == 1 and ndim_time == 1:
        assert len(survival_curves) == len(times_coordinates), \
            "The length of survival_curves and times_coordinate must be the same."
    elif ndim_surv == 2 and ndim_time == 2:
        assert survival_curves.shape[0] == times_coordinates.shape[0], \
            "The number of samples in survival_curves and times_coordinate must be the same."
        assert survival_curves.shape[1] == times_coordinates.shape[1], \
            "The number of time points in survival_curves and times_coordinate must be the same."
    elif ndim_surv == 2 and ndim_time == 1:
        assert survival_curves.shape[1] == len(times_coordinates), \
            "The number of time points in survival_curves and times_coordinate must be the same."
    elif ndim_surv == 1 and ndim_time == 2:
        assert len(survival_curves) == times_coordinates.shape[1], \
            "The number of time points in survival_curves and times_coordinate must be the same."
    else:
        raise ValueError("The dimension of survival_curves and times_coordinate must be 1-D or 2-D.")


def predict_rmst(
        survival_curves: np.ndarray,
        times_coordinates: np.ndarray,
        interpolation: str = "Linear",
) -> Union[float, np.ndarray]:
    """
    Get the restricted mean survival time (RMST) from the survival curve.
    The restricted mean survival time is defined as the area under the survival curve up to a certain time point.

    Parameters
    ----------
    survival_curves: np.ndarray
        The survival curve of samples. It is a 2-D or 1-D array. If it is a 2-D array, the first dimension is the
        number of samples, and the second dimension is the number of time points. If it is a 1-D array, it is the
        survival curve of a single sample.
    times_coordinates: np.ndarray
        The time coordinate of the survival curve. It is a 2-D or 1-D array. If it is a 2-D array, the first dimension
        is the number of samples, and the second dimension is the number of time points. If it is a 1-D array, it is the
        time coordinate of the survival curve of a single sample.
    interpolation: str
        The monotonic cubic interpolation method. One of ['None', 'Linear', 'Pchip']. Default: 'Linear'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.

    Returns
    -------
    restricted_mean_survival_times: float
        The restricted mean survival time(s).
    """
    _check_dim_align(survival_curves, times_coordinates)

    ndim_surv = survival_curves.ndim
    ndim_time = times_coordinates.ndim

    if interpolation == "None":
        width = np.diff(times_coordinates, axis=1 if ndim_time == 2 else 0)
        areas = width * survival_curves[:, :-1] if ndim_surv == 2 else width * survival_curves[:-1]
        rmst = np.sum(areas, axis=1)
    elif interpolation == "Linear":
        rmst = trapezoid(survival_curves, times_coordinates, axis=-1)
    elif interpolation == "Pchip":
        if ndim_time == 1:
            spline = PchipInterpolator(times_coordinates, survival_curves, axis=1 if ndim_surv == 2 else 0)
            rmst = spline.integrate(0, max(times_coordinates))
        elif ndim_time == 2:
            rmst = np.empty(survival_curves.shape[0])
            if ndim_surv == 1:
                for i in range(times_coordinates.shape[0]):
                    spline = PchipInterpolator(times_coordinates[i], survival_curves)
                    rmst[i] = spline.integrate(0, max(times_coordinates[i]))
            elif ndim_surv == 2:
                for i in range(times_coordinates.shape[0]):
                    print(i)
                    spline = PchipInterpolator(times_coordinates[i], survival_curves[i])
                    rmst[i] = spline.integrate(0, max(times_coordinates[i]))
        else:
            raise ValueError("times_coordinate must be 1-D or 2-D")
    else:
        raise ValueError("interpolation should be one of ['None', 'Linear', 'Pchip']")

    return rmst


def predict_mean_st(
        survival_curves: np.ndarray,
        times_coordinates: np.ndarray,
        interpolation: str = "Linear"
) -> Union[float, np.ndarray]:
    """
    Get the mean survival time(s) from the survival curve for a group of samples.
    The mean survival time is calculated as the area under the survival curve, which is the RMST + residual area.

    Parameters
    ----------
    survival_curves: np.ndarray
        The survival curve of samples. It is a 2-D or 1-D array. If it is a 2-D array, the first dimension is the
        number of samples, and the second dimension is the number of time points. If it is a 1-D array, it is the
        survival curve of a single sample.
    times_coordinates: np.ndarray
        The time coordinate of the survival curve. It is a 2-D or 1-D array. If it is a 2-D array, the first dimension
        is the number of samples, and the second dimension is the number of time points. If it is a 1-D array, it is the
        time coordinate of the survival curve of a single sample.
    interpolation: str
        The monotonic cubic interpolation method. One of ['None', 'Linear', 'Pchip']. Default: 'Linear'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.

    Returns
    -------
    median_survival_time: float
        The median survival time(s).
    """
    _check_dim_align(survival_curves, times_coordinates)

    ndim_surv = survival_curves.ndim
    ndim_time = times_coordinates.ndim

    rmst = predict_rmst(survival_curves, times_coordinates, interpolation)

    last_prob = survival_curves[:, -1] if ndim_surv == 2 else survival_curves[-1]
    last_time = times_coordinates[:, -1] if ndim_time == 2 else times_coordinates[-1]
    # the residual area is calculated as the area of a triangle with height = last_prob
    # and base = extrapolation_time - last_time
    # extrapolation_time is the time point where the survival curve crosses 0 (using the linear function of [0, 1] - [last_time, last_prob])
    residual_area = 0.5 * last_prob**2 * last_time / (1 - last_prob)
    return rmst + residual_area


def predict_median_st(
        survival_curves: np.ndarray,
        times_coordinates: np.ndarray,
        interpolation: str = "Linear"
) -> Union[float, np.ndarray]:
    """
    Get the median survival time(s) from the survival curve for a group of samples.
    The median survival time is defined as the time point where the survival curve crosses 0.5.

    Parameters
    ----------
    survival_curves: np.ndarray
        The survival curve of samples. It is a 2-D or 1-D array. If it is a 2-D array, the first dimension is the
        number of samples, and the second dimension is the number of time points. If it is a 1-D array, it is the
        survival curve of a single sample.
    times_coordinates: np.ndarray
        The time coordinate of the survival curve. It is a 2-D or 1-D array. If it is a 2-D array, the first dimension
        is the number of samples, and the second dimension is the number of time points. If it is a 1-D array, it is the
        time coordinate of the survival curve of a single sample.
    interpolation: str
        The monotonic cubic interpolation method. One of ['None', 'Linear', 'Pchip']. Default: 'Linear'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.

    Returns
    -------
    median_survival_time: float
        The median survival time(s).
    """
    _check_dim_align(survival_curves, times_coordinates)

    ndim_surv = survival_curves.ndim
    ndim_time = times_coordinates.ndim

    if ndim_surv == 1 and ndim_time == 1:
        median_sts = predict_median_st_ind(survival_curves, times_coordinates, interpolation)
    elif ndim_surv == 2 and ndim_time == 1:
        median_sts = np.empty(survival_curves.shape[0])
        for i in range(survival_curves.shape[0]):
            median_sts[i] = predict_median_st_ind(survival_curves[i, :], times_coordinates, interpolation)
    elif ndim_surv == 1 and ndim_time == 2:
        median_sts = np.empty(times_coordinates.shape[0])
        for i in range(times_coordinates.shape[0]):
            median_sts[i] = predict_median_st_ind(survival_curves, times_coordinates[i, :], interpolation)
    elif ndim_surv == 2 and ndim_time == 2:
        median_sts = np.empty(survival_curves.shape[0])
        for i in range(survival_curves.shape[0]):
            median_sts[i] = predict_median_st_ind(survival_curves[i, :], times_coordinates[i, :], interpolation)
    else:
        raise ValueError("The dimension of survival_curves and times_coordinate must be 1-D or 2-D.")

    return median_sts


def predict_median_st_ind(
        survival_curve: np.ndarray,
        times_coordinate: np.ndarray,
        interpolation: str = "Linear",
        discretize_num: int = 1000
) -> float:
    """
    Get the median survival time from the survival curve for 1 individual.
    The median survival time is defined as the time point where the survival curve crosses 0.5.
    The curve is first interpolated by the given monotonic cubic interpolation method (Linear or Pchip).
    Then the curve gets extrapolated by the linear function of (0, 1) and the last time point. The
    median survival time is calculated by finding the time point where the survival curve crosses 0.5.

    Parameters
    ----------
    survival_curve: np.ndarray
        The survival curve of the sample. 1-D array.
    times_coordinate: np.ndarray
        The time coordinate of the survival curve. 1-D array.
    interpolation: str
        The monotonic cubic interpolation method. One of ['Linear', 'Pchip']. Default: 'Linear'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.
    discretize_num: int
        The number of points to discretize the time range for interpolation. Default: 1000.
        This is used only when interpolation is 'Pchip'.

    Returns
    -------
    median_st: float
        The median survival time.
    """
    # If all the predicted probabilities are 1 the integral will be infinite.
    if np.all(survival_curve == 1):
        warnings.warn("All the predicted probabilities are 1, the median survival time will be infinite.")
        return np.inf

    min_prob = min(survival_curve)

    if min_prob <= 0.5:
        idx_arr = np.where(survival_curve <=0.5)[0]

        idx_after_median = idx_arr[0]
        if idx_after_median == 0 or survival_curve[idx_after_median] == 0.5:
            median_st = times_coordinate[idx_after_median]
        else:
            t1, t2 = times_coordinate[idx_after_median - 1], times_coordinate[idx_after_median]
            if interpolation == "Linear":
                # linear interpolation to find the median time
                p1, p2 = survival_curve[idx_after_median - 1], survival_curve[idx_after_median]
                median_st = (t1 + (0.5 - p1) * (t2 - t1) / (p2 - p1))
            elif interpolation == "Pchip":
                # reverse the array because the PchipInterpolator requires the x to be strictly increasing
                spline = interpolated_survival_curve(times_coordinate, survival_curve, interpolation)
                time_range = np.linspace(t1, t2, num=discretize_num)
                prob_range = spline(time_range)
                inverse_spline = PchipInterpolator(prob_range[::-1], time_range[::-1])
                median_st = np.array(inverse_spline(0.5)).item()
            else:
                raise ValueError("interpolation should be one of ['Linear', 'Pchip']")
    else:
        max_time = max(times_coordinate)
        min_prob = min(survival_curve)
        slope = (1 - min_prob) / (0 - max_time)
        median_st = - 0.5 / slope
    return median_st


def quantile_to_survival(quantile_levels, quantile_predictions, time_coordinates, interpolate='Pchip'):
    survival_level = 1 - quantile_levels
    slope = - quantile_levels[-1] / quantile_predictions[:, -1]
    surv_pred = np.empty((quantile_predictions.shape[0], time_coordinates.shape[0]))
    for i in range(quantile_predictions.shape[0]):
        # fit an interpolation function to the cdf
        spline = interpolated_survival_curve(quantile_predictions[i, :], survival_level, interpolate)

        # if the quantile level is beyond last cdf, we extrapolate the
        beyond_prob_idx = np.where(time_coordinates > quantile_predictions[i, -1])[0]
        surv_pred[i] = spline(time_coordinates)
        surv_pred[i, beyond_prob_idx] = np.clip(time_coordinates[beyond_prob_idx] * slope[i] + 1,
                                                a_min=0, a_max=1)

    # sanity checks
    assert np.all(surv_pred >= 0), "Survival predictions contain negative."
    assert check_monotonicity(surv_pred), "Survival predictions are not monotonic."
    return surv_pred


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
