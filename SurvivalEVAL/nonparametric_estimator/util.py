"""Counting, integration, and prediction helpers for nonparametric estimators."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from SurvivalEVAL.evaluations.util import get_prob_at_zero


def _predict_step(
    t: ArrayLike,
    event_times: np.ndarray | None,
    values: np.ndarray | None,
    baseline: float | np.ndarray,
    *,
    side: Literal["left", "right"] = "right",
) -> np.ndarray:
    """
    Evaluate scalar/vector/matrix steps at arbitrary shapes and either side.

    Parameters
    ----------
    t : array_like
        Non-negative query times; positive infinity requests the final fitted estimate.
    event_times : ndarray | None
        Sorted fitted update times; None denotes an estimator that has not been fitted.
    values : ndarray | None
        Values after each fitted update; the first axis corresponds to event_times.
    baseline : float | ndarray
        Value to broadcast before the first fitted update.
    side : Literal['left', 'right']
        "right" includes updates at each query time; "left" excludes them.

    Returns
    -------
    ndarray
        Step values with shape shape(t) + values.shape[1:], including baseline before the first update.

    Raises
    ------
    RuntimeError
        If the estimator has not been fitted.
    ValueError
        If a query time is negative or NaN, or side is invalid.
    """
    if event_times is None:
        raise RuntimeError("Call fit() before prediction.")
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    times = np.asarray(t, dtype=float)
    if np.any(np.isnan(times)) or np.any(times < 0):
        raise ValueError(
            "Prediction times must be non-negative and must not contain NaN."
        )

    indices = np.searchsorted(event_times, times.ravel(), side=side) - 1
    result = np.broadcast_to(baseline, (times.size,) + values.shape[1:]).copy()
    after_first = indices >= 0
    result[after_first] = values[indices[after_first]]
    return result.reshape(times.shape + values.shape[1:])


def _compute_event_counts(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Group observed times once and return times, at-risk counts, and events.

    Event/censor order within a tied-time group does not affect its risk set
    or event count. Reduce directly at group starts without re-sorting the
    times or constructing interleaved start/end indices in Python.

    Parameters
    ----------
    event_times : ndarray
        Observed event or censoring times, one per sample.
    event_indicators : ndarray
        Observed status labels; 0 denotes censoring and positive labels denote observed events.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Distinct observed times, numbers at risk immediately before them, and observed-event counts.
    """
    order = np.argsort(event_times)
    sorted_times = event_times[order]
    starts = np.r_[0, np.flatnonzero(sorted_times[1:] != sorted_times[:-1]) + 1]
    events = np.add.reduceat(
        event_indicators[order],
        starts,
        dtype=np.result_type(event_indicators.dtype, int),
    )
    return sorted_times[starts], len(event_times) - starts, events


def km_mean(times: np.ndarray, survival_probabilities: np.ndarray) -> float:
    """
    Calculate the mean of the Kaplan-Meier curve.

    Parameters
    ----------
    times : ndarray, shape = (n_samples, )
        Survival times for KM curve of the testing samples
    survival_probabilities : ndarray, shape = (n_samples, )
        Survival probabilities for KM curve of the testing samples

    Returns
    -------
    float
        Area under the piecewise-linear Kaplan-Meier curve, including its linear tail.
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
    # or the step function rule (deprecated for now)
    # area_subs = area_diff * area_probabilities[0:-1]
    # area_subs[-1] = area_subs[-1] / 2
    # area = np.flip(np.flip(area_subs).cumsum())

    # calculate the mean
    surv_prob = get_prob_at_zero(times, survival_probabilities)
    return area[0] / surv_prob


def infer_survival_probabilities(
    prediction_times: np.ndarray,
    survival_times: np.ndarray,
    survival_probabilities: np.ndarray,
    *,
    side: Literal["left", "right"] = "right",
) -> np.ndarray:
    """
    Evaluate a step survival curve with a linear tail at finite times.

    Left queries exclude an update exactly at the query time; right queries
    include it. Both sides use the same linear tail after the final fitted time.
    Times before the first observation have survival 1. Positive infinity
    has survival 0, including when the fitted tail is flat.

    Parameters
    ----------
    prediction_times : ndarray
        Non-negative query times for the fitted survival curve.
    survival_times : ndarray
        Increasing fitted observation times.
    survival_probabilities : ndarray
        Fitted survival probabilities at the observation times.
    side : Literal['left', 'right']
        "right" includes updates at each query time; "left" excludes them.

    Returns
    -------
    ndarray
        Survival probabilities with the query shape, extrapolated with the fitted linear tail.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    indices = np.searchsorted(survival_times, prediction_times, side=side) - 1
    # Index -1 denotes the pre-observation baseline, where survival is still 1.
    before_first = indices < 0
    indices = np.clip(indices, 0, survival_times.size - 1)
    probs = survival_probabilities[indices].astype(float, copy=True)
    probs[before_first] = 1.0

    # Extrapolate linearly for times beyond the last observed time point
    # using the line connecting (t_last, S(t_last)) and (t0, S(t0))
    beyond_last = np.isfinite(prediction_times) & (
        prediction_times > survival_times[-1]
    )
    if np.any(beyond_last):
        t0, s0 = survival_times[0], survival_probabilities[0]
        t_last, s_last = survival_times[-1], survival_probabilities[-1]
        denom = t_last - t0
        slope = 0.0 if denom == 0 else (s_last - s0) / denom
        extrapolated = s_last + slope * (prediction_times[beyond_last] - t_last)
        probs[beyond_last] = np.maximum(extrapolated, 0.0)
    probs[np.isposinf(prediction_times)] = 0.0
    return probs
