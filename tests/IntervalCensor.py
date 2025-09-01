import numpy as np
from typing import Sequence, Tuple, Optional, Dict, Any
from SurvivalEVAL.Evaluations.util import interpolated_survival_curve
from scipy.optimize import brentq

# reuse interpolated_survival_curve
def _invert_from_survival_with_interpolator(S_grid: np.ndarray, t_grid: np.ndarray,
                                            ps: Sequence[float], method: str) -> np.ndarray:
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