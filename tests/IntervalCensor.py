import numpy as np
from typing import Sequence, Tuple, Optional, Dict, Any
from SurvivalEVAL.Evaluations.util import interpolated_survival_curve
from scipy.optimize import brentq

# reuse interpolated_survival_curve
def _invert_from_survival_with_interpolator(S_grid: np.ndarray, t_grid: np.ndarray,
                                            ps: Sequence[float], method: str) -> np.ndarray:
    '''
    S_grid: Survival Prediction curve matrix
    t_grid: time grid
    ps: quantile
    method: "Linear"|"Pchip"
    '''
    S = np.minimum.accumulate(np.clip(S_grid, 0.0, 1.0), axis=1)  # keep montonic decreasing
    N, _ = S.shape; ps = np.asarray(ps, float)
    t_lo, t_hi = float(t_grid[0]), float(t_grid[-1])
    out = np.empty((N, ps.size), float)
    for i in range(N):
        spl = interpolated_survival_curve(t_grid, S[i], method)
        s_lo, s_hi = float(spl(t_lo)), float(spl(t_hi))  # s_lo >= s_hi
        for j, p in enumerate(ps):
            target = 1.0 - float(p)
            if target >= s_lo: out[i, j] = t_lo; continue
            if target <= s_hi: out[i, j] = t_hi; continue
            g = lambda t: float(spl(t)) - target
            out[i, j] = brentq(g, t_lo, t_hi, maxiter=50)
    return out

def calibration_slope_right_censor(
    event_indicators: np.ndarray,           # bool: 
    observed_times: np.ndarray,             # float: 
    predictions: np.ndarray,                # (N, T): CDF 
    time_grid: np.ndarray,                  # (T,) 
    ps: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    *,
    quantile_method: str = "Linear",        # "Linear"|"Pchip"
    through_origin: bool = True             
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    right uncensor (p, obs(p))：
      - event count=1{t_ip >= y_i}
      - censor count t_ip <= c_i 
    return：(p_list, obs_list, slope)
    """
    e = np.asarray(event_indicators, bool)   # event
    y = np.asarray(observed_times,   float)   # observation time
    P = np.asarray(predictions,      float)
    t = np.asarray(time_grid,        float)  # time grid
    assert P.ndim == 2 and P.shape[0] == y.shape[0] and P.shape[1] == t.shape[0], "shape unmatch"
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
        ind_e  = (t_ip >= y) & e
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

'''
ps, obs, slope = calibration_slope_right_censor(
    event_indicators=event_indicators,     # bool
    observed_times=observed_times,    # float
    predictions=predictions_cdf,      # (N,T), CDF
    time_grid=time_grid,
    ps=(0.1,0.3,0.5,0.7,0.9),
    quantile_method="Linear", 
    through_origin=True
)
print("slope:", slope)
'''

def cov_from_cdf_grid(
    cdf: np.ndarray,         # (N, T) per-patient CDF on an increasing time grid
    t_grid: np.ndarray,    # (T,)
):
    """
    Compute per-patient CoV = SD[F]/E[F] from a discretized CDF
    0.1, 0.2, 0.3, 0.4
    1, 5, 8, 10
    CoV = (Var (0.1 * (1+5)/2 + 0.1 * 5+8/2 ,... ))
    """

    # # 2) discrete pdf mass per bin: dF_k = F(t_{k+1}) - F(t_k)
    dF = np.diff(cdf, axis=1)

    t_mid = 0.5 * (t_grid[1:] + t_grid[:-1])           # (T-1,)    
    # 3) midpoint rule to approximate E[T] and E[T^2]
    m1 = np.sum(dF * t_mid, axis=1)                    # E[F]
    m2 = np.sum(dF * (t_mid**2), axis=1)               # E[F^2]
    var = m2 - m1**2                                   # Var[F] ≥ 0 = E[F^2] - E[F]^2

    # 4) CoV per patient; guard divide-by-zero
    cov = np.where(m1 > 0, np.sqrt(var) / m1, np.nan)  # (N,)

    return cov                                     # patient-wise CoV

def _prepare_cdf(F, t_grid):
    """Clip CDF to [0,1] and (optionally) enforce non-decreasing over time."""
    F = np.asarray(F, float)
    t_grid = np.asarray(t_grid, float)
    assert F.ndim == 2 and F.shape[1] == t_grid.shape[0], "shape mismatch"
    assert np.all(np.diff(t_grid) >= 0), "time_grid must be increasing"
    return F, t_grid

def _interp_cdf_row(F_i, t_grid, t_eval, left_fill=None):
    """
    1D interpolation for a single patient's CDF:
    - left of grid: by default use F(t_min) (or 0.0 if left_fill=0.0 is passed)
    - right of grid: use 1.0 (CDF tail)
    """
    if left_fill is None:
        left_fill = F_i[0]
    return np.interp(t_eval, t_grid, F_i, left=left_fill, right=1.0)

def survival_auprc_uncensored_grid(
    event_times: np.ndarray,      # (Nu,)
    predictions_cdf: np.ndarray,  # (Nu, T)
    time_grid: np.ndarray,        # (T,)
    n_quad: int = 256,
) -> np.ndarray:
    """
    Per-patient Survival-AUPRC for UNcensored samples:
        AUPRC(y; F) = ∫_0^1 [ F(y/t) - F(y*t) ] dt
    Returns: (N,) array of scores in [0, 1].
    """
    F, t = _prepare_cdf(predictions_cdf, time_grid)
    event_times = np.asarray(event_times, float)
    N = event_times.shape[0]

    # Midpoint quadrature over (0, 1]
    ts = np.linspace(0.0, 1.0, n_quad + 1)
    ts_mid = 0.5 * (ts[1:] + ts[:-1])       # (Q,)
    widths = ts[1:] - ts[:-1]               # (Q,)

    scores = np.empty(N, float)
    for i in range(N):
        yi = float(event_times[i])
        # Evaluate F at transformed times
        t_right = yi / ts_mid               # y / t 
        t_left  = yi * ts_mid               # y * t
        Fi_right = _interp_cdf_row(F[i], t, t_right)
        Fi_left  = _interp_cdf_row(F[i], t, t_left)
        integrand = Fi_right - Fi_left
        scores[i] = float(np.sum(integrand * widths))
    return scores

def survival_auprc_right_censored_grid(
    censor_times: np.ndarray,     # (N_c,)
    predictions_cdf: np.ndarray,  # (N_c, T)
    time_grid: np.ndarray,        # (T,)
    n_quad: int = 256,
) -> np.ndarray:
    """
    Per-patient Survival-AUPRC for RIGHT-censored samples:
        AUPRC([L,∞); F) = ∫_0^1 [ 1 - F(L*t) ] dt
        (F(t_k) - F(t_k+1))/(t_k+1 - t_k)
    Returns: (Nc,) array of scores in [0, 1].
    """
    F, t = _prepare_cdf(predictions_cdf, time_grid)
    censor_times = np.asarray(censor_times, float)
    Nc = censor_times.shape[0]

    ts = np.linspace(0.0, 1.0, n_quad + 1)
    ts_mid = 0.5 * (ts[1:] + ts[:-1])
    widths = ts[1:] - ts[:-1]

    scores = np.empty(Nc, float)
    for i in range(Nc):
        Fi_at = _interp_cdf_row(F[i], t, float(censor_times[i]) * ts_mid)
        integrand = 1.0 - Fi_at
        scores[i] = float(np.sum(integrand * widths))
    return scores

def survival_auprc_right(
    event_indicators: np.ndarray,   # (N,) True=event observed, False=right-censored
    observed_times: np.ndarray,     # (N,) event time or censor time
    predictions_cdf: np.ndarray,    # (N, T)
    time_grid: np.ndarray,          # (T,)
    n_quad: int = 256,
) -> np.ndarray:
    """
    AUPRC formula for uncensored and right-censored scenarios
    Returns: (N,) array of per-patient AUPRC
    """
    e = np.asarray(event_indicators, bool)
    t_obs = np.asarray(observed_times, float)
    F = np.asarray(predictions_cdf, float)

    # Split rows
    idx_event, idx_cens = np.where(e)[0], np.where(~e)[0]

    scores = np.empty(F.shape[0], float)
    if idx_event.size: # for uncensor
        scores[idx_event] = survival_auprc_uncensored_grid(
            t_obs[idx_event], F[idx_event], time_grid,
            n_quad=n_quad
        )
    if idx_cens.size: # for right censor
        scores[idx_cens] = survival_auprc_right_censored_grid(
            t_obs[idx_cens], F[idx_cens], time_grid,
            n_quad=n_quad
        )
    return scores



def survival_auprc_interval(
    left_bounds: np.ndarray,       # (N,) L_i
    right_bounds: np.ndarray,      # (N,) U_i  (use np.inf for right-censor)
    predictions_cdf: np.ndarray,   # (N, T)   per-patient CDF on time_grid
    time_grid: np.ndarray,         # (T,)
    n_quad: int = 256,             # Q
    left_extrapolation_value: float = None,  # set to 0.0 if you want F(t<grid_min)=0
) -> np.ndarray:
    """
    Per-patient Survival-AUPRC for INTERVAL-censored samples:
        AUPRC([L,U]; F) = ∫_0^1 [ F(U/t) - F(L*t) ] dt
    Handles special cases automatically:
        - exact: L==U
        - right-censored: U==+inf  → term F(U/t)=1
        - left-censored:  L==0     → term F(L*t)=F(0)≈0
    Returns: (N,) array of scores in [0, 1].
    """
    F, t = _prepare_cdf(predictions_cdf, time_grid)
    L = np.asarray(left_bounds,  float)
    U = np.asarray(right_bounds, float)
    N = L.shape[0]

    # Midpoint quadrature over (0, 1]
    ts = np.linspace(0.0, 1.0, n_quad + 1)
    ts_mid = 0.5 * (ts[1:] + ts[:-1])   # (Q,)
    widths = ts[1:] - ts[:-1]           # (Q,)

    scores = np.empty(N, float)
    for i in range(N):
        Li, Ui = float(L[i]), float(U[i])

        # Right term: F(U/t). 
        if np.isinf(Ui):
            Fi_right = np.ones_like(ts_mid) #If Ui is +inf, this term is identically 1.
        else:
            t_right = Ui / ts_mid          # can exceed grid → right=1.0
            Fi_right = _interp_cdf_row(F[i], t, t_right, left_fill=left_extrapolation_value)

        # Left term: F(L*t). If L=0 and you want F(0)=0, set left_extrapolation_value=0.0
        t_left  = Li * ts_mid              # can be < grid_min → left fill is used
        Fi_left = _interp_cdf_row(F[i], t, t_left, left_fill=left_extrapolation_value)

        integrand = Fi_right - Fi_left
        scores[i] = float(np.sum(integrand * widths))
    return scores

# --- Interval-censor calibration slope ---
def calibration_slope_interval_censor(
    left_bounds: np.ndarray,                 # (N,) float  L_i
    right_bounds: np.ndarray,                # (N,) float  U_i (use np.inf for right-censor)
    predictions_cdf: np.ndarray,             # (N, T)     per-patient CDF on time_grid
    time_grid: np.ndarray,                   # (T,)       increasing
    ps: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    *,
    quantile_method: str = "Linear",          # "Linear" | "Pchip"  (for survival interpolator)
    through_origin: bool = True,             # fit slope through origin (default in paper)
    clip_p: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute calibration points (p, obs(p)) and slope under INTERVAL censoring:
      - Include as 1 if t_{i,p} >= U_i
      - Include as 0 if t_{i,p} <= L_i
      - Skip if L_i < t_{i,p} < U_i
    Returns (p_list, obs_list, slope).
    """
    L = np.asarray(left_bounds,  float)
    U = np.asarray(right_bounds, float)
    F = np.asarray(predictions_cdf, float)
    t = np.asarray(time_grid,     float)

    # basic checks
    assert F.ndim == 2 and F.shape[0] == L.shape[0] and F.shape[1] == t.shape[0], "shape mismatch"
    assert np.all(np.diff(t) >= 0), "time_grid must be increasing"
    ps = np.clip(np.asarray(ps, float), clip_p, 1.0 - clip_p)

    # sanitize CDF and convert to Survival for quantile inversion
    F = np.maximum.accumulate(np.clip(F, 0.0, 1.0), axis=1)  # enforce monotone ↑
    S = 1.0 - F

    # compute all quantiles t_{i,p} at once
    t_all = _invert_from_survival_with_interpolator(S, t, ps, method=quantile_method)  # (N, P)

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

# TODO survival_auprc_interval

