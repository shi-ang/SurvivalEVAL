"""
File: OtherMetrics.py

Author: Weijie Sun

Description:
    Various other evaluation metrics for survival analysis models. Not integrated into the main Evaluator classes.
    These functions can be used independently for custom evaluation needs or further research.
"""

from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

from SurvivalEVAL.Evaluations.util import interpolated_curve


# reuse interpolated_survival_curve
def _invert_from_survival_with_interpolator(
    S_grid: np.ndarray, t_grid: np.ndarray, ps: Sequence[float], method: str
) -> np.ndarray:
    """
    S_grid: Survival Prediction curve matrix
    t_grid: time grid
    ps: quantile
    method: "Linear"|"Pchip"
    """
    S = np.minimum.accumulate(
        np.clip(S_grid, 0.0, 1.0), axis=1
    )  # keep montonic decreasing
    N, _ = S.shape
    ps = np.asarray(ps, float)
    t_lo, t_hi = float(t_grid[0]), float(t_grid[-1])
    out = np.empty((N, ps.size), float)
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
            g = lambda t: float(spl(t)) - target
            out[i, j] = brentq(g, t_lo, t_hi, maxiter=50)
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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    right uncensor (p, obs(p))：
      - event count=1{t_ip >= y_i}
      - censor count t_ip <= c_i
    return：(p_list, obs_list, slope)
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
        intercept = 0.0
    else:
        X = np.c_[np.ones_like(x), x]
        beta, *_ = np.linalg.lstsq(X, y_fit, rcond=None)
        intercept, slope = float(beta[0]), float(beta[1])

    return p_arr, o_arr, slope


# --- Interval-censor calibration slope ---
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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute calibration points (p, obs(p)) and slope under INTERVAL censoring:
      - Include as 1 if t_{i,p} >= U_i
      - Include as 0 if t_{i,p} <= L_i
      - Skip if L_i < t_{i,p} < U_i
    Returns (p_list, obs_list, slope).
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


def cov(
    cdf: np.ndarray, t_grid: np.ndarray, return_details: bool = False
) -> float | Tuple[float, np.ndarray]:
    """
    Compute CoV = SD[F]/E[F] from a discretized CDF
    0.1, 0.2, 0.3, 0.4
    1, 5, 8, 10
    CoV = (Var (0.1 * (1+5)/2 + 0.1 * 5+8/2 ,... ))

    Parameters
    ----------
    cdf : np.ndarray, shape (n_samples, n_timepoints)
        The predicted cumulative distribution functions for each patient.
    t_grid : np.ndarray, shape (n_timepoints,)
        The time grid corresponding to the CDF values.
    return_details : bool, optional
        If True, also return the per-patient CoV values.

    Returns
    -------
    float
        The mean CoV across all patients.
    """
    # discrete pmf mass per bin: dF_k = F(t_{k+1}) - F(t_k)
    pmf = np.diff(cdf, axis=1)

    t_mid = 0.5 * (t_grid[1:] + t_grid[:-1])  # (T-1,)
    # midpoint rule to approximate E[T] and E[T^2]
    m1 = np.sum(pmf * t_mid, axis=1)  # E[F]
    m2 = np.sum(pmf * (t_mid**2), axis=1)  # E[F^2]
    var = m2 - m1**2  # Var[F] ≥ 0 = E[F^2] - E[F]^2

    if np.any(var < -1e-8):
        raise ValueError(
            "Zero or negative variance encountered in CoV calculation. Ignore these samples or check CDF input."
        )
    # CoV per patient; guard divide-by-zero
    cov = np.where(m1 > 0, np.sqrt(var) / m1, np.nan)  # (N,)

    mean_cov = float(np.nanmean(cov))
    if return_details:
        return mean_cov, cov
    else:
        return mean_cov
