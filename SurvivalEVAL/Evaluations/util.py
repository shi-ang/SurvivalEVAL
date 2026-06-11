from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import pandas as pd
import torch
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator, interp1d
from sklearn.isotonic import isotonic_regression

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
                error = (
                    f"{type(arg)} is not a valid data format. Only use "
                    "'list', 'tuple', 'np.ndarray', 'torch.Tensor', "
                    "'pd.Series', 'pd.DataFrame'"
                )
                raise TypeError(error)

            if np.sum(np.isnan(x)) > 0.0:
                error = "The #{} argument contains null values"
                error = error.format(i + 1)
                raise ValueError(error)

            if len(args) > 1:
                if i > 0:
                    assert x[0].shape == last_length, (
                        "Shapes between {}-th input array and "
                        "{}-th input array are not consistent".format(i - 1, i)
                    )
                result += x
                last_length = x[0].shape
            else:
                result = x[0]

    return result


def check_monotonicity(
    array: NumericArrayLike,
    direction: str | None = None,
) -> bool:
    """
    Check whether values are monotonic along the last axis.

    Parameters
    ----------
    array : NumericArrayLike
        A one- or two-dimensional array.
    direction : {"increasing", "decreasing"}, optional
        Required monotonic direction. Both directions allow equal adjacent values.
        If omitted, either direction is accepted for backward compatibility.

    Returns
    -------
    bool
        Whether the array is monotonic in the requested direction.
    """
    array = check_and_convert(array)
    if array.ndim not in (1, 2):
        raise ValueError("The input array must be 1-D or 2-D.")

    # Direct adjacent comparisons handle repeated infinities; np.diff would
    # turn inf - inf into nan and falsely reject monotonic arrays.
    adjacent_left = array[..., :-1]
    adjacent_right = array[..., 1:]
    is_increasing = bool(np.all(adjacent_right >= adjacent_left))
    is_decreasing = bool(np.all(adjacent_right <= adjacent_left))

    if direction is None:
        return is_increasing or is_decreasing
    direction = direction.lower()
    if direction == "increasing":
        return is_increasing
    if direction == "decreasing":
        return is_decreasing
    raise ValueError("direction must be one of ['increasing', 'decreasing']")


def make_monotonic(
    survival_curves: np.ndarray,
    times_coordinate: np.ndarray,
    method: str = "ceil",
    seed: int = None,
    num_bs: int = None,
    direction: str = "decreasing",
):
    """
    Make curves monotonic along the time axis.

    Parameters
    ----------
    survival_curves: np.ndarray
        One- or two-dimensional probability curves. For a 2-D array, rows are
        samples and columns are time points.
    times_coordinate: np.ndarray
        Time points corresponding to the survival curves. 1-D array of time points.
    method: str
        Correction method. One of ``"ceil"``, ``"floor"``, ``"bootstrap"``,
        or ``"isotonic"``. Isotonic regression gives the L2-optimal monotonic
        approximation. Default: ``"ceil"``.
    seed: int
        Random seed for bootstrapping. Default: None.
    num_bs: int
        Number of bootstrap samples. Default: None. If None, then
        ``num_bs = max(10 * num_times, 1000)``.
    direction: {"increasing", "decreasing"}
        Required monotonic direction. Survival curves are decreasing, while
        cumulative distribution functions are increasing. Default: ``"decreasing"``.

    Returns
    -------
    survival_curves: np.ndarray
        Monotonic probability curves with the same dimensionality as the input.
    """
    method = method.lower()
    direction = direction.lower()

    valid_methods = {"ceil", "floor", "bootstrap", "isotonic"}
    if method not in valid_methods:
        raise ValueError(
            "method must be one of ['ceil', 'floor', 'bootstrap', 'isotonic']"
        )
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be one of ['increasing', 'decreasing']")

    survival_curves = np.asarray(survival_curves, dtype=float)
    times_coordinate = np.asarray(times_coordinate, dtype=float)
    if survival_curves.ndim not in (1, 2):
        raise ValueError("survival_curves must be a 1-D or 2-D array.")
    if times_coordinate.ndim != 1:
        raise ValueError("times_coordinate must be a 1-D array.")
    if survival_curves.shape[-1] != times_coordinate.size:
        raise ValueError(
            "survival_curves and times_coordinate must have the same number of time points."
        )
    if not check_monotonicity(times_coordinate, direction="increasing"):
        raise ValueError("The time coordinates must be sorted in ascending order.")

    input_was_1d = survival_curves.ndim == 1
    curves = np.atleast_2d(np.clip(survival_curves, 0.0, 1.0))
    if check_monotonicity(curves, direction=direction):
        return curves[0] if input_was_1d else curves

    if method == "isotonic":
        increasing = direction == "increasing"
        curves = np.vstack(
            [
                isotonic_regression(
                    curve,
                    y_min=0.0,
                    y_max=1.0,
                    increasing=increasing,
                )
                for curve in curves
            ]
        )
    elif method == "ceil":
        if direction == "decreasing":
            curves = np.maximum.accumulate(curves[:, ::-1], axis=1)[:, ::-1]
        else:
            curves = np.maximum.accumulate(curves, axis=1)
    elif method == "floor":
        if direction == "decreasing":
            curves = np.minimum.accumulate(curves, axis=1)
        else:
            curves = np.minimum.accumulate(curves[:, ::-1], axis=1)[:, ::-1]
    else:
        if num_bs is None:
            # 10 times the number of time points or 1000, whichever is larger
            num_bs = max(10 * len(times_coordinate), 1000)
        if num_bs <= 0:
            raise ValueError("num_bs must be positive.")
        if seed is not None:
            np.random.seed(seed)

        # The bootstrap rearrangement is defined for decreasing survival curves.
        bootstrap_curves = curves.copy()
        if direction == "increasing":
            bootstrap_curves = 1.0 - bootstrap_curves

        need_rearrange = np.where(
            np.any(np.diff(bootstrap_curves, axis=1) > 0, axis=1)
        )[0]
        for i in need_rearrange:
            inter_lin = interp1d(
                bootstrap_curves[i],
                times_coordinate,
                kind="linear",
                fill_value="extrapolate",
            )
            bootstrap_qf = inter_lin(np.random.uniform(0, 1, num_bs))
            for j, time in enumerate(times_coordinate):
                bootstrap_curves[i, j] = np.mean(bootstrap_qf > time)

        curves = (
            1.0 - bootstrap_curves if direction == "increasing" else bootstrap_curves
        )

    return curves[0] if input_was_1d else curves


def interpolated_curve(
    times_coordinate: np.ndarray, curve: np.ndarray, interpolation: str = "Linear"
) -> Union[interp1d, PchipInterpolator]:
    interpolation = interpolation.lower()
    if interpolation == "linear":
        spline = interp1d(
            times_coordinate, curve, kind="linear", fill_value="extrapolate"
        )
    elif interpolation == "pchip":
        spline = PchipInterpolator(times_coordinate, curve)
    else:
        raise ValueError("interpolation must be one of ['Linear', 'Pchip']")
    return spline


def predict_prob_from_curve(
    survival_curve: np.ndarray,
    times_coordinate: np.ndarray,
    target_time: float,
    interpolation: str = "Linear",
) -> float:
    """
    Predict the probability of survival at a given time point from the survival curve. The survival curve is
    interpolated using the specified interpolation method ('Linear' or 'Pchip').
    If the input time grid does not start at 0, a time-zero point with survival
    probability 1 is prepended before interpolation.
    If the target time is greater than the largest time coordinate, the probability
    is extrapolated by the linear function through (0, 1) and the last grid point.

    Parameters
    ----------
    survival_curve: np.ndarray
        Survival curve. 1-D array of survival probabilities.
    times_coordinate: np.ndarray
        Time points corresponding to the survival curve. 1-D array of time
        points. Values must be non-negative. If the first value is greater than
        0, `(0, 1)` is prepended to the curve.
    target_time: float
        Non-negative time point at which to predict the probability of survival.
    interpolation: str
        The monotonic cubic interpolation method. One of ['Linear', 'Pchip']. Default: 'Linear'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.

    Returns
    -------
    predict_probability: float
        Predicted probability of survival at the target time point.
    """
    survival_curve, times_coordinate = zero_padding(survival_curve, times_coordinate)
    if survival_curve.ndim != 1 or times_coordinate.ndim != 1:
        raise ValueError("survival_curve and times_coordinate must be 1-D arrays.")
    target_time = float(target_time)
    if target_time < 0:
        raise ValueError("target_time must be non-negative.")

    spline = interpolated_curve(times_coordinate, survival_curve, interpolation)

    # predicting boundary
    max_time = float(max(times_coordinate))

    # simply calculate the slope by using the [0, 1] - [max_time, S(t|x)]
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # Above the fitted time grid, use the survival tail fit described above;
    # otherwise use the configured interpolator.
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
    interpolation: str = "Linear",
) -> np.ndarray:
    """
    Predict the probability of survival at multiple time points from the survival curve. The survival curve is
    interpolated using the specified interpolation method ('Linear' or 'Pchip').
    If the input time grid does not start at 0, a time-zero point with survival
    probability 1 is prepended before interpolation.
    If a target time is greater than the largest time coordinate, the probability
    is extrapolated by the linear function through (0, 1) and the last grid point.

    Parameters
    ----------
    survival_curve: np.ndarray
        Survival curve. 1-D array of survival probabilities.
    times_coordinate: np.ndarray
        Time points corresponding to the survival curve. 1-D array of time
        points. Values must be non-negative. If the first value is greater than
        0, `(0, 1)` is prepended to the curve.
    target_times: NumericArrayLike
        Non-negative time points at which to predict the probability of survival.
    interpolation: str
        The monotonic cubic interpolation method. One of ['Linear', 'Pchip']. Default: 'Linear'.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.

    Returns
    -------
    predict_probabilities: np.ndarray
        Predicted probabilities of survival at the target time points.
    """
    survival_curve, times_coordinate = zero_padding(survival_curve, times_coordinate)
    if survival_curve.ndim != 1 or times_coordinate.ndim != 1:
        raise ValueError("survival_curve and times_coordinate must be 1-D arrays.")
    target_times = check_and_convert(target_times).astype(float)
    if target_times.ndim != 1:
        raise ValueError("target_times must be a 1-D array.")
    if np.any(target_times < 0):
        raise ValueError("target_times must be non-negative.")

    spline = interpolated_curve(times_coordinate, survival_curve, interpolation)

    # predicting boundary
    max_time = float(max(times_coordinate))

    # simply calculate the slope by using the [0, 1] - [maxtime, S(t|x)]
    slope = (1 - np.array(spline(max_time)).item()) / (0 - max_time)

    # Above the fitted time grid, use the survival tail fit described above;
    # otherwise use the configured interpolator.
    predict_probabilities = np.array(spline(target_times))
    after_grid = target_times > max_time
    predict_probabilities[after_grid] = np.maximum(
        slope * target_times[after_grid] + 1, 0
    )

    return predict_probabilities


def align_curve_and_time_coordinates(
    curves: NumericArrayLike,
    time_coordinates: NumericArrayLike,
    n_samples: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and broadcast curve values and time coordinates to matching 2D arrays.

    Either input may be 1D and shared by all samples, while a 2D input contains
    one row per sample. When both inputs are 1D, they represent one sample unless
    ``n_samples`` is provided.

    Parameters
    ----------
    curves: NumericArrayLike
        Curve values with shape ``(n_time_points,)`` or
        ``(n_samples, n_time_points)``.
    time_coordinates: NumericArrayLike
        Time coordinates with shape ``(n_time_points,)`` or
        ``(n_samples, n_time_points)``.
    n_samples: int, optional
        Expected number of samples. This is useful when both curve inputs are
        shared and the sample count comes from a separate outcome array.

    Returns
    -------
    aligned_curves: np.ndarray, shape = (n_samples, n_time_points)
        Curve values with shared rows broadcast across samples.
    aligned_time_coordinates: np.ndarray, shape = (n_samples, n_time_points)
        Time coordinates with shared rows broadcast across samples.
    """
    curves = np.asarray(curves, dtype=float)
    time_coordinates = np.asarray(time_coordinates, dtype=float)

    if curves.ndim not in (1, 2) or time_coordinates.ndim not in (1, 2):
        raise ValueError("curves and time_coordinates must be 1D or 2D arrays.")
    if curves.shape[-1] != time_coordinates.shape[-1]:
        raise ValueError(
            "curves and time_coordinates must contain the same number of time points."
        )

    inferred_sample_counts = [
        array.shape[0] for array in (curves, time_coordinates) if array.ndim == 2
    ]
    if n_samples is None:
        n_samples = inferred_sample_counts[0] if inferred_sample_counts else 1
    elif not isinstance(n_samples, (int, np.integer)) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")

    if any(sample_count != n_samples for sample_count in inferred_sample_counts):
        raise ValueError("curves and time_coordinates must contain one row per sample.")

    if curves.ndim == 1:
        curves = np.broadcast_to(curves, (n_samples, curves.shape[0]))
    if time_coordinates.ndim == 1:
        time_coordinates = np.broadcast_to(
            time_coordinates, (n_samples, time_coordinates.shape[0])
        )

    return curves, time_coordinates


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
        time coordinate shared by all survival curves.
    interpolation: str
        The integration method. One of ['None', 'Linear', 'Pchip']. Default: 'Linear'.
        If 'None', treat the survival curve as a left-continuous step function.
        If 'Linear', use trapezoidal integration.
        If 'Pchip', integrate a monotonic cubic interpolation.

    Returns
    -------
    restricted_mean_survival_times: float or np.ndarray
        A float when both inputs are 1-D, otherwise one restricted mean
        survival time per survival curve or time coordinate row.
    """
    ndim_surv = survival_curves.ndim
    ndim_time = times_coordinates.ndim
    scalar_output = ndim_surv == 1 and ndim_time == 1
    curves, time_grids = align_curve_and_time_coordinates(
        survival_curves, times_coordinates
    )

    interpolation = interpolation.lower()
    if interpolation == "none":
        width = np.diff(time_grids, axis=1)
        areas = width * curves[:, :-1]
        rmst = np.sum(areas, axis=1)
    elif interpolation == "linear":
        rmst = trapezoid(curves, time_grids, axis=1)
    elif interpolation == "pchip":
        rmst = np.empty(curves.shape[0])
        for i in range(curves.shape[0]):
            spline = PchipInterpolator(time_grids[i], curves[i])
            rmst[i] = spline.integrate(0, np.max(time_grids[i]))
    else:
        raise ValueError("interpolation should be one of ['None', 'Linear', 'Pchip']")

    return float(rmst[0]) if scalar_output else rmst


def predict_mean_st(
    survival_curves: np.ndarray,
    times_coordinates: np.ndarray,
    interpolation: str = "Linear",
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
    mean_survival_times: float or np.ndarray
        A float when both inputs are 1-D, otherwise one mean survival
        time per survival curve or time coordinate row.
    """
    ndim_surv = survival_curves.ndim
    ndim_time = times_coordinates.ndim
    scalar_output = ndim_surv == 1 and ndim_time == 1
    curves, time_grids = align_curve_and_time_coordinates(
        survival_curves, times_coordinates
    )

    rmst = predict_rmst(curves, time_grids, interpolation)

    last_prob = curves[:, -1]
    last_time = time_grids[:, -1]
    # the residual area is calculated as the area of a triangle with height = last_prob
    # and base = extrapolation_time - last_time
    # extrapolation_time is the time point where the survival curve crosses 0 (using the linear function of [0, 1] - [last_time, last_prob])
    residual_area = 0.5 * last_prob**2 * last_time / (1 - last_prob)
    mean_st = rmst + residual_area
    return float(mean_st[0]) if scalar_output else mean_st


def predict_median_st(
    survival_curves: np.ndarray,
    times_coordinates: np.ndarray,
    interpolation: str = "Linear",
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
    ndim_surv = survival_curves.ndim
    ndim_time = times_coordinates.ndim
    scalar_output = ndim_surv == 1 and ndim_time == 1
    curves, time_grids = align_curve_and_time_coordinates(
        survival_curves, times_coordinates
    )

    median_sts = np.empty(curves.shape[0])
    for i in range(curves.shape[0]):
        median_sts[i] = predict_median_st_ind(curves[i], time_grids[i], interpolation)

    return float(median_sts[0]) if scalar_output else median_sts


def predict_median_st_ind(
    survival_curve: np.ndarray,
    times_coordinate: np.ndarray,
    interpolation: str = "Linear",
    discretize_num: int = 1000,
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
        warnings.warn(
            "All the predicted probabilities are 1, the median survival time will be infinite."
        )
        return np.inf

    min_prob = min(survival_curve)

    interpolation = interpolation.lower()

    if min_prob <= 0.5:
        idx_arr = np.where(survival_curve <= 0.5)[0]

        idx_after_median = idx_arr[0]
        if idx_after_median == 0 or survival_curve[idx_after_median] == 0.5:
            median_st = times_coordinate[idx_after_median]
        else:
            t1, t2 = (
                times_coordinate[idx_after_median - 1],
                times_coordinate[idx_after_median],
            )
            if interpolation == "linear":
                # linear interpolation to find the median time
                p1, p2 = (
                    survival_curve[idx_after_median - 1],
                    survival_curve[idx_after_median],
                )
                median_st = t1 + (0.5 - p1) * (t2 - t1) / (p2 - p1)
            elif interpolation == "pchip":
                # reverse the array because the PchipInterpolator requires the x to be strictly increasing
                spline = interpolated_curve(
                    times_coordinate, survival_curve, interpolation
                )
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
        median_st = -0.5 / slope
    return median_st


def quantile_to_survival(
    quantile_levels, quantile_predictions, time_coordinates, interpolate="Pchip"
):
    survival_level = 1 - quantile_levels
    slope = -quantile_levels[-1] / quantile_predictions[:, -1]
    surv_pred = np.empty((quantile_predictions.shape[0], time_coordinates.shape[0]))
    for i in range(quantile_predictions.shape[0]):
        # fit an interpolation function to the cdf
        spline = interpolated_curve(
            quantile_predictions[i, :], survival_level, interpolate
        )

        # if the quantile level is beyond last cdf, we extrapolate the
        beyond_prob_idx = np.where(time_coordinates > quantile_predictions[i, -1])[0]
        surv_pred[i] = spline(time_coordinates)
        surv_pred[i, beyond_prob_idx] = np.clip(
            time_coordinates[beyond_prob_idx] * slope[i] + 1, a_min=0, a_max=1
        )

    # sanity checks
    assert np.all(surv_pred >= 0), "Survival predictions contain negative."
    assert check_monotonicity(
        surv_pred, direction="decreasing"
    ), "Survival predictions are not nonincreasing."
    return surv_pred


def survival_to_quantile(
    surv_prob: NumericArrayLike,
    time_coordinates: NumericArrayLike,
    quantile_levels: NumericArrayLike,
    interpolate: str = "Pchip",
) -> np.ndarray:
    """
    Convert survival curves S(t) to quantile estimates t_q where F(t_q)=q and F=1-S.

    Parameters
    ----------
    surv_prob : (n_samples, n_times) array-like
        Survival probabilities S(t) on a grid of times.
    time_coordinates : (n_samples, n_times) array-like
        Time grid corresponding to `surv_prob` for each sample. Must be strictly increasing per row.
    quantile_levels : (n_quantiles,) array-like
        Values in [0, 1), in increasing order. Each q is mapped to t_q such that F(t_q)=q.
    interpolate : {"Linear", "Pchip"}, default "Pchip"
        Interpolator for the CDF-to-time mapping.

    Returns
    -------
    quantile_predictions : (n_samples, n_quantiles) np.ndarray
        Estimated quantile times for each sample.

    Notes
    -----
    - For q beyond the maximum observed CDF value in a row, extrapolates with a
      linear CDF tail through the origin: t ≈ q / slope, where
      slope = F(T_max)/T_max.
    - Rows with duplicated CDF x-values are de-duplicated (keep first occurrence).
    """
    # Convert to arrays
    surv_prob, time_coordinates = check_and_convert(surv_prob, time_coordinates)
    quantile_levels = check_and_convert(quantile_levels)

    if surv_prob.shape != time_coordinates.shape:
        raise ValueError(
            "`surv_prob` and `time_coordinates` must have identical shapes (n_samples, n_times)."
        )
    if surv_prob.ndim != 2:
        raise ValueError(
            "`surv_prob` and `time_coordinates` must be 2D (n_samples, n_times)."
        )
    if quantile_levels.ndim != 1:
        raise ValueError("`quantile_levels` must be 1D.")

    if not check_monotonicity(surv_prob, direction="decreasing"):
        raise ValueError("Each row of `surv_prob` must be nonincreasing.")

    if np.any(np.diff(time_coordinates, axis=1) <= 0):
        raise ValueError("Each row of `time_coordinates` must be strictly increasing.")

    if not check_monotonicity(quantile_levels, direction="increasing"):
        raise ValueError("`quantile_levels` must be in increasing order.")

    if np.any(quantile_levels < 0) or np.any(quantile_levels >= 1):
        raise ValueError("`quantile_levels` must be in [0, 1).")

    interpolate = interpolate.lower()
    if interpolate == "linear":
        Interpolator = interp1d
    elif interpolate == "pchip":
        Interpolator = PchipInterpolator
    else:
        raise ValueError(f"Unknown interpolation method: {interpolate}")

    # CDF and tail slope (assume linear tail beyond last time point)
    cdf = 1.0 - surv_prob
    # guard against zero last time to avoid divide-by-zero
    if np.any(time_coordinates[:, -1] <= 0):
        raise ValueError("The last time in each `time_coordinates` row must be > 0.")
    slope = cdf[:, -1] / time_coordinates[:, -1]

    n_samples, _ = cdf.shape
    qpred = np.empty((n_samples, quantile_levels.shape[0]), dtype=float)

    for i in range(n_samples):
        cdf_i = cdf[i, :]
        t_i = time_coordinates[i, :]

        # Build monotone x for interpolator: unique CDF values (keep first)
        cdf_i_unique, keep_idx = np.unique(cdf_i, return_index=True)
        t_i_unique = t_i[keep_idx]
        max_cdf = cdf_i_unique[-1]

        if max_cdf <= 0:
            # No observed CDF increase: F(t)=0 on the supplied grid, so only
            # q=0 is reached at the origin; positive quantiles are undefined.
            qpred[i, :] = np.where(quantile_levels == 0.0, 0.0, np.inf)
            continue

        # If the first CDF value is >0, prepend (0, 0) to allow interpolation near q≈0
        if cdf_i_unique[0] > 0.0:
            cdf_i_unique = np.concatenate(([0.0], cdf_i_unique))
            t_i_unique = np.concatenate(([0.0], t_i_unique[:1]))

        # Create CDF^{-1}(q) interpolator (monotone methods preferred)
        if Interpolator is interp1d:
            interp = Interpolator(
                cdf_i_unique,
                t_i_unique,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )
        else:
            # Pchip is shape-preserving; x must be strictly increasing
            interp = Interpolator(cdf_i_unique, t_i_unique, extrapolate=True)

        # Interpolate for all qs, then handle tail beyond observed max CDF
        qpred[i, :] = interp(quantile_levels)

        beyond = np.where(quantile_levels > max_cdf)[0]
        if beyond.size > 0:
            qpred[i, beyond] = quantile_levels[beyond] / slope[i]

    # Sanity checks
    if np.any(qpred < 0):
        raise RuntimeError("Quantile predictions contain negative values.")
    if not check_monotonicity(qpred, direction="increasing"):
        raise RuntimeError(
            "Quantile predictions are not nondecreasing across quantile levels per row."
        )

    return qpred


def get_prob_at_zero(times: np.ndarray, survival_probabilities: np.ndarray) -> float:
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


def _prepend_origin(
    pred_survs: np.ndarray, time_coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepend survival probability 1 and time 0 while preserving input ranks.

    Supports the shape combinations accepted by `zero_padding`: a shared 1-D
    curve or 2-D sample-by-time curves, paired with a 1-D shared time grid or
    2-D sample-specific time grids.
    """
    # Preserve whether the survival input represents one curve or many rows.
    if pred_survs.ndim == 1:
        padded_survs = np.concatenate(([1.0], pred_survs))
    elif pred_survs.ndim == 2:
        padded_survs = np.concatenate(
            (np.ones((pred_survs.shape[0], 1)), pred_survs), axis=1
        )
    else:
        error = (
            "Predicted survival curves must be a 1D or 2D array, got {} instead".format(
                pred_survs.ndim
            )
        )
        raise TypeError(error)

    # Preserve whether time coordinates are shared or sample-specific.
    if time_coordinates.ndim == 1:
        padded_times = np.concatenate(([0.0], time_coordinates))
    elif time_coordinates.ndim == 2:
        padded_times = np.concatenate(
            (np.zeros((time_coordinates.shape[0], 1)), time_coordinates), axis=1
        )
    else:
        error = "Time coordinates must be a 1D or 2D array, got {} instead".format(
            time_coordinates.ndim
        )
        raise TypeError(error)

    return padded_survs, padded_times


def zero_padding(
    pred_survs: np.ndarray, time_coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepend a time-zero survival anchor when a curve grid does not start at 0.

    This helper normalizes survival curve grids to include the conventional
    origin point ``(time=0, survival=1)``. When the first time coordinate is
    already 0, the original arrays are returned unchanged. When the grid starts
    elsewhere, the function prepends 0 to ``time_coordinates`` and 1 to
    ``pred_survs`` while preserving each input's dimensionality. For 2-D time
    grids, all rows must either already start at 0 or all require padding.

    Parameters
    ----------
    pred_survs: np.ndarray, shape = (n_time_points,) or (n_samples, n_time_points)
        Predicted survival probabilities. A 1-D array represents a single
        curve or a shared curve, while a 2-D array represents one curve per
        sample.
    time_coordinates: np.ndarray, shape = (n_time_points,) or (n_samples, n_time_points)
        Time coordinates corresponding to ``pred_survs``. A 1-D array is a
        shared time grid, while a 2-D array contains sample-specific time
        grids. Values must be non-negative, and the last dimension must have
        the same length as
        ``pred_survs``.

    Returns
    -------
    padded_survs: np.ndarray
        Survival probabilities with an origin point prepended when needed. The
        dimensionality of ``pred_survs`` is preserved.
    padded_times: np.ndarray
        Time coordinates with 0 prepended when needed. The dimensionality of
        ``time_coordinates`` is preserved.

    Raises
    ------
    TypeError
        If either input is not a numeric numpy array, or if either input is
        not a 1-D or 2-D array.
    ValueError
        If the inputs are empty, have different numbers of time points, contain
        negative time coordinates, or if a 2-D time grid mixes rows that start
        at 0 with rows that require padding.

    Warns
    -----
    UserWarning
        If an origin point is prepended.
    """
    if not isinstance(pred_survs, np.ndarray):
        raise TypeError("pred_survs must be a numpy array.")
    if not isinstance(time_coordinates, np.ndarray):
        raise TypeError("time_coordinates must be a numpy array.")
    if not np.issubdtype(pred_survs.dtype, np.number):
        raise TypeError("pred_survs must contain numeric values.")
    if not np.issubdtype(time_coordinates.dtype, np.number):
        raise TypeError("time_coordinates must contain numeric values.")

    ndim_time = time_coordinates.ndim
    ndim_surv = pred_survs.ndim
    zero_pad_msg = (
        "The first time coordinate is not 0. An authentic survival curve should start from 0 "
        "with 100% survival probability. Adding 0 to the beginning of the time coordinates "
        "and 1 to the beginning of the predicted curves."
    )

    if ndim_surv not in (1, 2):
        error = (
            "Predicted survival curves must be a 1D or 2D array, got {} instead".format(
                ndim_surv
            )
        )
        raise TypeError(error)
    if ndim_time not in (1, 2):
        error = "Time coordinates must be a 1D or 2D array, got {} instead".format(
            ndim_time
        )
        raise TypeError(error)
    if pred_survs.shape[-1] == 0 or time_coordinates.shape[-1] == 0:
        raise ValueError("pred_survs and time_coordinates must be non-empty.")
    if pred_survs.shape[-1] != time_coordinates.shape[-1]:
        raise ValueError(
            "Predicted survival curves and time coordinates must have "
            "the same number of time points."
        )
    if np.any(time_coordinates < 0):
        raise ValueError("time_coordinates must be non-negative.")

    if ndim_time == 1:
        needs_padding = not np.isclose(time_coordinates[0], 0.0)
    else:
        starts_at_zero = np.isclose(time_coordinates[:, 0], 0.0)
        if np.any(starts_at_zero) and not np.all(starts_at_zero):
            raise ValueError(
                "All rows of 2D time_coordinates must either start at 0 or "
                "all require zero-padding."
            )
        needs_padding = not np.all(starts_at_zero)

    if needs_padding:
        warnings.warn(zero_pad_msg)
        _pred_survs, _time_coordinates = _prepend_origin(pred_survs, time_coordinates)
    else:
        _pred_survs = pred_survs
        _time_coordinates = time_coordinates

    return _pred_survs, _time_coordinates


def fit_least_squares(x, y, left_anchor=True, right_anchor=True) -> tuple[float, float]:
    """
    Fit a least squares line to the given data.
    Parameters
    ----------
    x: np.ndarray, shape = (n_bins, )
        The x-coordinates of the data points.
    y: np.ndarray, shape = (n_bins, )
        The y-coordinates of the data points.
    left_anchor: bool
        Whether to anchor the leftmost point. Default: True.
    right_anchor: bool
        Whether to anchor the rightmost point. Default: True.

    Returns
    -------
    slope: float
        The slope of the fitted line.
    intercept: float
        The intercept of the fitted line.
    """
    n_bins = len(y) - 1
    if left_anchor and right_anchor:
        X = np.column_stack([x, np.ones(n_bins + 1)])
        slope, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
        return slope, intercept
    elif left_anchor and not right_anchor:
        # exclude anchor n_bins
        X = np.column_stack([x[:-1], np.ones(n_bins)])
        slope, intercept = np.linalg.lstsq(X, y[:-1], rcond=None)[0]
        return slope, intercept
    elif not left_anchor and right_anchor:
        # exclude anchor 0
        X = np.column_stack([x[1:], np.ones(n_bins)])
        slope, intercept = np.linalg.lstsq(X, y[1:], rcond=None)[0]
        return slope, intercept
    else:
        # exclude anchors 0 and n_bins
        X = np.column_stack([x[1:-1], np.ones(n_bins - 1)])
        slope, intercept = np.linalg.lstsq(X, y[1:-1], rcond=None)[0]
    return slope, intercept
