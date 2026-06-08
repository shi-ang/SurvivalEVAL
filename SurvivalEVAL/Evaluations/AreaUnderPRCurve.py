from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from SurvivalEVAL.Evaluations.util import (
    align_curve_and_time_coordinates,
    check_monotonicity,
)


def _interp_cdf_row(F_i, t_grid, t_eval, left_fill=None):
    """
    1D interpolation for a single patient's CDF:
    - left of grid: by default use F(t_min) (or 0.0 if left_fill=0.0 is passed)
    - right of grid: use 1.0 (CDF tail)
    """
    if left_fill is None:
        left_fill = F_i[0]

    f = interp1d(
        t_grid,
        F_i,
        kind="linear",
        fill_value=(left_fill, 1.0),
        bounds_error=False,
        assume_sorted=True,
    )
    return f(t_eval)


def auprc_uncensored_grid(
    pred_cdf: np.ndarray,
    time_grid: np.ndarray,
    event_times: np.ndarray,
    n_quad: int = 256,
) -> np.ndarray:
    """
    Per-patient Survival-AUPRC for an uncensored samples:
        AUPRC(y; F) = ∫_0^1 [ F(y/t) - F(y*t) ] dt
    Returns: (N,) array of scores in [0, 1].
    """
    event_times = np.asarray(event_times, float)
    N = event_times.shape[0]
    pred_cdf, time_grid = align_curve_and_time_coordinates(
        pred_cdf, time_grid, n_samples=N
    )

    assert check_monotonicity(
        pred_cdf, direction="increasing"
    ), "predictions_cdf must be non-decreasing over time"
    assert check_monotonicity(
        time_grid, direction="increasing"
    ), "time_grid must be non-decreasing"

    # Midpoint quadrature over (0, 1]
    ts = np.linspace(0.0, 1.0, n_quad + 1)
    ts_mid = 0.5 * (ts[1:] + ts[:-1])  # (Q,)
    widths = ts[1:] - ts[:-1]  # (Q,)

    scores = np.empty(N, float)
    for i in range(N):
        yi = float(event_times[i])
        # Evaluate F at transformed times
        t_right = yi / ts_mid  # y / t
        t_left = yi * ts_mid  # y * t
        Fi_right = _interp_cdf_row(pred_cdf[i], time_grid[i], t_right)
        Fi_left = _interp_cdf_row(pred_cdf[i], time_grid[i], t_left)
        integrand = Fi_right - Fi_left
        scores[i] = float(np.sum(integrand * widths))
    return scores


def auprc_right_censored_grid(
    pred_cdf: np.ndarray,
    time_grid: np.ndarray,
    censor_times: np.ndarray,
    n_quad: int = 256,
) -> np.ndarray:
    """
    Per-patient Survival-AUPRC for RIGHT-censored samples:
        AUPRC([L,∞); F) = ∫_0^1 [ 1 - F(L*t) ] dt
        (F(t_k) - F(t_k+1))/(t_k+1 - t_k)
    Returns: (Nc,) array of scores in [0, 1].
    """
    censor_times = np.asarray(censor_times, float)
    Nc = censor_times.shape[0]
    pred_cdf, time_grid = align_curve_and_time_coordinates(
        pred_cdf, time_grid, n_samples=Nc
    )

    assert check_monotonicity(
        pred_cdf, direction="increasing"
    ), "predictions_cdf must be non-decreasing over time"
    assert check_monotonicity(
        time_grid, direction="increasing"
    ), "time_grid must be non-decreasing"

    ts = np.linspace(0.0, 1.0, n_quad + 1)
    ts_mid = 0.5 * (ts[1:] + ts[:-1])
    widths = ts[1:] - ts[:-1]

    scores = np.empty(Nc, float)
    for i in range(Nc):
        Fi_at = _interp_cdf_row(
            pred_cdf[i], time_grid[i], float(censor_times[i]) * ts_mid
        )
        integrand = 1.0 - Fi_at
        scores[i] = float(np.sum(integrand * widths))
    return scores


def auprc_right_censor(
    pred_cdf: np.ndarray,
    time_grid: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    n_quad: int = 256,
    return_details: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    AUPRC formula for uncensored and right-censored scenarios

    Proposed in Section 2.3 in
    Countdown Regression: Sharp and Calibrated Survival Predictions, UAI 2019
    https://proceedings.mlr.press/v115/avati20a.html

    Parameters
    ----------
    pred_cdf: np.ndarray, (n_time_points,) or (n_samples, n_time_points)
        Predicted cumulative event probabilities F(t) = 1 - S(t) for the
        testing samples. A 1D array is shared by all samples. Values must be
        nondecreasing over time.
    time_grid: np.ndarray, (n_time_points,) or (n_samples, n_time_points)
        Time grids corresponding to the predicted CDF values. A 1D array is
        shared by all samples.
    event_times: np.ndarray, (n_samples,)
        Observed event or censoring times for the testing samples.
    event_indicators: np.ndarray, (n_samples,)
        Binary indicators of whether the event occurred (1) or was censored (0).
    n_quad: int
        Number of quadrature points for numerical integration.
    return_details: bool
        If True, also return per-patient scores.

    Returns
    -------
    auprc:  float
        The mean AUPRC over all samples.
    scores: np.ndarray, (n_samples,)
        Per-patient AUPRC scores.
    """
    event_times = np.asarray(event_times, float)
    event_indicators = np.asarray(event_indicators, bool)
    if event_times.shape != event_indicators.shape:
        raise ValueError("event_times and event_indicators must have the same shape.")

    pred_cdf, time_grid = align_curve_and_time_coordinates(
        pred_cdf, time_grid, event_times.shape[0]
    )

    # Split rows
    idx_event, idx_cens = np.where(event_indicators)[0], np.where(~event_indicators)[0]

    scores = np.empty(pred_cdf.shape[0], float)
    if idx_event.size:  # for uncensor
        scores[idx_event] = auprc_uncensored_grid(
            pred_cdf[idx_event],
            time_grid[idx_event],
            event_times[idx_event],
            n_quad=n_quad,
        )
    if idx_cens.size:  # for right censor
        scores[idx_cens] = auprc_right_censored_grid(
            pred_cdf[idx_cens],
            time_grid[idx_cens],
            event_times[idx_cens],
            n_quad=n_quad,
        )

    auprc = scores.mean()
    if return_details:
        return auprc, scores
    return auprc


def auprc_ic(
    pred_cdf: np.ndarray,
    time_grid: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_quad: int = 256,
    left_extrapolation_value: float = None,
    return_details: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Per-patient Survival-AUPRC for INTERVAL-censored samples:
        AUPRC([left,right]; F) = ∫_0^1 [ F(right/t) - F(left*t) ] dt

    Proposed in Section 2.3 in
    Countdown Regression: Sharp and Calibrated Survival Predictions, UAI 2019
    https://proceedings.mlr.press/v115/avati20a.html

    Handles special cases automatically:
        - exact: left==right → reduces to uncensored AUPRC
        - right-censored: right==+inf  → term F(right/t)=1
        - left-censored:  left==0     → term F(left*t)=F(0)≈0

    Parameters
    ----------
    pred_cdf: np.ndarray, (n_time_points,) or (n_samples, n_time_points)
        Predicted cumulative event probabilities F(t) = 1 - S(t) for the
        testing samples. A 1D array is shared by all samples. Values must be
        nondecreasing over time.
    time_grid: np.ndarray, (n_time_points,) or (n_samples, n_time_points)
        Time grids corresponding to the predicted CDF values. A 1D array is
        shared by all samples.
    left: np.ndarray, (n_samples,)
        Left interval bounds
    right: np.ndarray, (n_samples,)
        Right interval bounds; use np.inf for right-censoring.
    n_quad: int
        Number of quadrature points for numerical integration.
    left_extrapolation_value: float, optional
        Value to use for CDF evaluations before the first time-grid point.
        When omitted, ``left == 0`` uses ``F(0) = 0`` automatically, while
        other evaluations before the grid use ``F(t_min)``.
    return_details: bool
        If True, also return per-patient scores.

    Returns:
    -------
    auprc: float
        The mean AUPRC over all samples.
    scores: np.ndarray, (n_samples,), optional
        Per-patient AUPRC scores, returned if return_details=True.
    """
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape.")

    N = left.shape[0]
    pred_cdf, time_grid = align_curve_and_time_coordinates(
        pred_cdf, time_grid, n_samples=N
    )

    assert check_monotonicity(
        pred_cdf, direction="increasing"
    ), "predictions_cdf must be non-decreasing over time"
    assert check_monotonicity(
        time_grid, direction="increasing"
    ), "time_grid must be non-decreasing"

    # Midpoint quadrature over (0, 1]
    ts = np.linspace(0.0, 1.0, n_quad + 1)
    ts_mid = 0.5 * (ts[1:] + ts[:-1])  # (Q,)
    widths = ts[1:] - ts[:-1]  # (Q,)

    scores = np.empty(N, float)
    for i in range(N):
        li, ri = float(left[i]), float(right[i])

        # Right term: F(right/t).
        if np.isinf(ri):
            Fi_right = np.ones_like(
                ts_mid
            )  # If ri is +inf, this term is identically 1.
        else:
            t_right = ri / ts_mid  # can exceed grid → right=1.0
            Fi_right = _interp_cdf_row(
                pred_cdf[i],
                time_grid[i],
                t_right,
                left_fill=left_extrapolation_value,
            )

        # For a left-censored interval, L=0 makes this term F(0).
        if li == 0.0:
            left_boundary = (
                0.0 if left_extrapolation_value is None else left_extrapolation_value
            )
            Fi_left = np.full_like(ts_mid, left_boundary)
        else:
            t_left = li * ts_mid  # can be < grid_min, so the left fill is used
            Fi_left = _interp_cdf_row(
                pred_cdf[i],
                time_grid[i],
                t_left,
                left_fill=left_extrapolation_value,
            )

        integrand = Fi_right - Fi_left
        scores[i] = float(np.sum(integrand * widths))
    auprc = scores.mean()
    if return_details:
        return auprc, scores
    return auprc
