"""
Calibration slopes for right- and interval-censored survival predictions.

Author: Weijie Sun
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.optimize import brentq

from SurvivalEVAL.evaluations.util import interpolated_curve


def _invert_from_survival_with_interpolator(
    S_grid: np.ndarray,
    t_grid: np.ndarray,
    ps: Sequence[float],
    method: str,
) -> np.ndarray:
    """
    Invert interpolated survival curves at cumulative event-probability levels.

    Parameters
    ----------
    S_grid : ndarray
        Survival probabilities with shape (n_samples, n_times).
    t_grid : ndarray
        One-dimensional time grid corresponding to the predicted probabilities.
    ps : Sequence[float]
        Cumulative event-probability levels at which to evaluate calibration.
    method : str
        Curve interpolation method, "Linear" or "Pchip".

    Returns
    -------
    ndarray
        Event-time quantiles with shape (n_samples, n_quantiles), bounded by the supplied grid.
    """
    S = np.minimum.accumulate(
        np.clip(S_grid, 0.0, 1.0), axis=1
    )  # keep montonic decreasing
    N, _ = S.shape
    ps = np.asarray(ps, float)
    t_lo, t_hi = float(t_grid[0]), float(t_grid[-1])
    out = np.empty((N, ps.size), float)

    def gap(
        t: float,
        interpolator: Callable[[float], float | np.ndarray],
        target: float,
    ) -> float:
        """
        Evaluate the survival-probability discrepancy used for quantile root finding.

        Parameters
        ----------
        t : float
            Time at which to evaluate the interpolated survival curve.
        interpolator : Callable[[float], float | ndarray]
            Callable mapping one time to its interpolated survival probability.
        target : float
            Target survival probability for quantile inversion.

        Returns
        -------
        float
            Interpolated survival probability minus the target probability.
        """
        return float(interpolator(t)) - target

    for i in range(N):
        spl = interpolated_curve(t_grid, S[i], method)
        s_lo, s_hi = float(spl(t_lo)), float(spl(t_hi))  # s_lo >= s_hi
        for j, p in enumerate(ps):
            target = 1.0 - float(p)
            if target >= s_lo:
                out[i, j] = t_lo
                continue
            if target <= s_hi:
                out[i, j] = t_hi
                continue

            out[i, j] = brentq(gap, t_lo, t_hi, args=(spl, target), maxiter=50)
    return out


def calibration_slope_right_censor(
    event_indicators: np.ndarray,  # bool:
    observed_times: np.ndarray,  # float:
    predictions: np.ndarray,  # (N, T): CDF
    time_grid: np.ndarray,  # (T,)
    ps: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    *,
    quantile_method: str = "Linear",  # "Linear"|"Pchip"
    through_origin: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate calibration proportions and slope for right-censored observations.

    Observed events contribute according to whether their event time precedes the
    predicted quantile; censorings beyond the quantile contribute non-events.
    Calibration slopes for survival predictions are discussed in [1]_.

    Parameters
    ----------
    event_indicators : ndarray
        Observed status labels; 0 denotes censoring and positive labels denote observed events.
    observed_times : ndarray
        Observed event or censoring times, one per sample.
    predictions : ndarray
        Predicted CDF values with shape (n_samples, n_times).
    time_grid : ndarray
        Increasing time coordinates, with shape (n_times,) or (n_samples, n_times).
    ps : Sequence[float]
        Cumulative event-probability levels at which to evaluate calibration.
    quantile_method : str
        Interpolation method for quantile inversion, "Linear" or "Pchip".
    through_origin : bool
        Whether to fit the calibration slope with the intercept fixed at zero.

    Returns
    -------
    tuple[ndarray, ndarray, float]
        Requested probability levels, observed event proportions, and the fitted calibration slope.

    References
    ----------
    .. [1] Avati, A.; Duan, T.; Zhou, S.; Jung, K.; Shah, N. H.; Ng, A. Y. (2020).
       "Countdown Regression: Sharp and Calibrated Survival Predictions".
       Proceedings of the 35th Uncertainty in Artificial Intelligence Conference,
       Proceedings of Machine Learning Research, 115, 145-155.
       https://arxiv.org/pdf/1806.08324
    """
    e = np.asarray(event_indicators, bool)  # event
    y = np.asarray(observed_times, float)  # observation time
    P = np.asarray(predictions, float)
    t = np.asarray(time_grid, float)  # time grid
    assert (
        P.ndim == 2 and P.shape[0] == y.shape[0] and P.shape[1] == t.shape[0]
    ), "shape unmatch"
    assert np.all(np.diff(t) >= 0), "time_grid must monotonic increasing"
    clip_p = 1e-6
    ps = np.clip(np.asarray(ps, float), clip_p, 1.0 - clip_p)

    F = np.clip(P, 0.0, 1.0)
    S = 1.0 - F
    t_all = _invert_from_survival_with_interpolator(S, t, ps, quantile_method)

    p_list, obs_list = [], []
    counts, kept_event, kept_cens = [], [], []
    w = np.ones_like(y, dtype=float)

    for j, p in enumerate(ps):
        t_ip = t_all[:, j]

        # count 1{t_ip >= y}
        keep_e = e
        ind_e = (t_ip >= y) & e
        # right censor only t_ip <= y count
        keep_c = (~e) & (t_ip <= y)

        keep = keep_e | keep_c
        denom_w = np.sum(w[keep])

        num_w = np.sum(w[ind_e])
        obs = num_w / denom_w

        p_list.append(float(p))
        obs_list.append(float(obs))
        counts.append(int(keep.sum()))
        kept_event.append(int(keep_e.sum()))
        kept_cens.append(int(keep_c.sum()))

    p_arr = np.array(p_list, float)
    o_arr = np.array(obs_list, float)

    x, y_fit = p_arr, o_arr
    if through_origin:
        slope = float((x @ y_fit) / (x @ x + 1e-12))
    else:
        X = np.c_[np.ones_like(x), x]
        beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
        slope = float(beta[1])

    return p_arr, o_arr, slope


def calibration_slope_interval_censor(
    left_bounds: np.ndarray,  # (N,) float  L_i
    right_bounds: np.ndarray,  # (N,) float  U_i (use np.inf for right-censor)
    pred_cdf: np.ndarray,  # (N, T)     per-patient CDF on time_grid
    time_grid: np.ndarray,  # (T,)       increasing
    ps: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    *,
    quantile_method: str = "Linear",  # "Linear" | "Pchip"  (for survival interpolator)
    through_origin: bool = True,  # fit slope through origin (default in paper)
    clip_p: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Calculate calibration proportions and slope for interval-censored observations.

    Include an event when its upper bound is at or before the predicted quantile.
    Include a non-event when the quantile is at or before the lower bound;
    omit intervals whose interior contains the quantile.
    Calibration slopes for survival predictions are discussed in [1]_.

    Parameters
    ----------
    left_bounds : ndarray
        Left observation bounds, one per sample.
    right_bounds : ndarray
        Right observation bounds, using positive infinity for right censoring.
    pred_cdf : ndarray
        Nondecreasing cumulative event probabilities, with shape (n_times,) or (n_samples, n_times).
    time_grid : ndarray
        Increasing time coordinates, with shape (n_times,) or (n_samples, n_times).
    ps : Sequence[float]
        Cumulative event-probability levels at which to evaluate calibration.
    quantile_method : str
        Interpolation method for quantile inversion, "Linear" or "Pchip".
    through_origin : bool
        Whether to fit the calibration slope with the intercept fixed at zero.
    clip_p : float
        Distance from 0 and 1 used to clip the requested probability levels.

    Returns
    -------
    tuple[ndarray, ndarray, float]
        Requested probability levels, observed event proportions, and the fitted calibration slope.

    References
    ----------
    .. [1] Avati, A.; Duan, T.; Zhou, S.; Jung, K.; Shah, N. H.; Ng, A. Y. (2020).
       "Countdown Regression: Sharp and Calibrated Survival Predictions".
       Proceedings of the 35th Uncertainty in Artificial Intelligence Conference,
       Proceedings of Machine Learning Research, 115, 145-155.
       https://arxiv.org/pdf/1806.08324
    """
    L = np.asarray(left_bounds, float)
    U = np.asarray(right_bounds, float)
    F = np.asarray(pred_cdf, float)
    t = np.asarray(time_grid, float)

    # basic checks
    assert (
        F.ndim == 2 and F.shape[0] == L.shape[0] and F.shape[1] == t.shape[0]
    ), "shape mismatch"
    assert np.all(np.diff(t) >= 0), "time_grid must be increasing"
    ps = np.clip(np.asarray(ps, float), clip_p, 1.0 - clip_p)

    # sanitize CDF and convert to Survival for quantile inversion
    F = np.maximum.accumulate(np.clip(F, 0.0, 1.0), axis=1)  # enforce monotone ↑
    S = 1.0 - F

    # compute all quantiles t_{i,p} at once
    t_all = _invert_from_survival_with_interpolator(
        S, t, ps, method=quantile_method
    )  # (N, P)

    p_list, obs_list = [], []

    # per-p loop to apply interval rules
    for j, p in enumerate(ps):
        t_ip = t_all[:, j]

        # Include as 1 if t_ip >= U   (note: U may be +inf -> never true)
        keep_one = t_ip >= U

        # Include as 0 if t_ip <= L, but not those already counted as 1 (tie goes to 1)
        keep_zero = (~keep_one) & (t_ip <= L)

        keep = keep_one | keep_zero
        denom = int(np.sum(keep))

        num_ones = int(np.sum(keep_one))
        obs = num_ones / float(denom)

        p_list.append(float(p))
        obs_list.append(float(obs))

    p_arr = np.array(p_list, dtype=float)
    o_arr = np.array(obs_list, dtype=float)

    x, y_fit = p_arr, o_arr
    if through_origin:
        slope = float((x @ y_fit) / (x @ x + 1e-12))
    else:
        X = np.c_[np.ones_like(x), x]
        beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
        slope = float(beta[1])

    return p_arr, o_arr, slope
