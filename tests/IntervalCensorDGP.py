import numpy as np
from typing import Tuple, Literal, Optional, Dict

def convert_right_censor_to_interval_censor(
    event_indicators: np.ndarray,   # (N,) bool: True=event, False=right-censor
    observed_times: np.ndarray,     # (N,) float: event or censor time
    max_t = np.inf # max time
):
    """
    Convert (event_indicators, observed_times) to lifelines-style interval data.

    Rules:
      - Event at time t:
          left = event time t
      - Right-censored at time c:
          left = max_t 
    """
    e = np.asarray(event_indicators, bool)
    y = np.asarray(observed_times, float)
    N = y.shape[0]

    left  = np.empty(N, dtype=float)
    right = np.empty(N, dtype=float)

    # small tolerance for alignment checks
    eps = 1e-12

    for i in range(N):
        t = y[i]
        
        if e[i]:
            # event case
            left[i]  = t
            right[i] = t
        else:
            # right-censored case
            left[i]  = t
            right[i] = max_t

    return left, right

def _visits_fixed(end: float, step: float) -> np.ndarray:
    if end <= 0:
        return np.array([0.0])
    start = 0.0
    grid = np.arange(start, end, step)
    if grid.size == 0 or grid[-1] != end:
        grid = np.concatenate([grid, [end]])
    return np.unique(np.clip(grid, 0.0, end))

def _visits_poisson(rng: np.random.Generator, end: float, rate: float) -> np.ndarray:
    if end <= 0:
        return np.array([0.0])
    K = rng.poisson(lam=max(rate * end, 0.0))
    times = rng.uniform(0.0, end, size=K)    
    times = np.concatenate(([0.0], times))
    times = np.concatenate((times, [end]))
    times.sort()
    return np.unique(times)

def _visits_lognormal(rng: np.random.Generator, end: float, mean: float, sigma: float) -> np.ndarray:
    times = [0.0]
    t = float(rng.lognormal(mean, sigma))
    while t < end:
        times.append(t)
        t += float(rng.lognormal(mean, sigma))
    times.append(end)
    return np.unique(np.array(times))

def _visits_exp(rng, end, rate: float = 1.0):
    t = 0.0
    times = [0.0]
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= end:
            break
        times.append(t)
    times.append(end)
    return np.array(times)

def _hawkes_times(rng: np.random.Generator, mu: float = 0.2, alpha: float = 0.3, beta: float = 1.5, end: float = 1000, max_events: int = 100) -> np.ndarray:
    """
    Simulate Hawkes with intensity: λ(t) = μ + α * sum_k exp(-β (t - t_k)).
    Returns sorted event times in (0, T].
    """
    assert mu >= 0 and alpha >= 0 and beta > 0 and end >= 0
    t, g = 0.0, 0.0                  # g = sum exp(-β (t - t_k))
    lam = mu + alpha * g
    times = []

    while t < end and len(times) < max_events:
        if lam <= 0:
            break
        w = rng.exponential(1.0 / lam)     # candidate gap
        t_cand = t + w
        if t_cand > end:
            break
        # decay g over gap
        g *= np.exp(-beta * w)
        lam_cand = mu + alpha * g
        # accept with prob lam_cand / lam
        if rng.random() * lam <= lam_cand and lam_cand > 0:
            # event at t_cand
            times.append(t_cand)
            g += 1.0                       # kernel at 0 is 1
            t = t_cand
            lam = mu + alpha * g
        else:
            t = t_cand
            lam = lam_cand

    return np.array(times, dtype=float)    

def interval_censor_DGP_from_synthetic_times(
    event_times: np.ndarray,           # (N,) true event time
    censoring_times: np.ndarray,       # (N,) right-censor/admin end per subject
    *,
    method: str = "fixed",
    # fixed
    params: Optional[Dict[str, float]] = None,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert (event_times, censoring_times) into interval-censored bounds (left, right):
      - If event in (visit_j, visit_{j+1}]: left=visit_j, right=visit_{j+1} (interval-censor)
      - If event == a visit:             left=right=event (uncensor)
      - If event > last visit:           left=last visit, right=+inf (right-censor)
    Returns: (left, right, n_visits)
    """
    rng = np.random.default_rng(seed)
    e = np.asarray(event_times, float)
    c = np.asarray(censoring_times, float)
    assert e.shape == c.shape
    N = e.shape[0]

    left = np.empty(N, float)
    right = np.empty(N, float)
    n_visits = np.empty(N, int)

    for i in range(N):
        end_i = float(c[i])
        end_i = max(end_i, 0.0)

        if method == "fixed":
            step = params['step'] if 'step' in params else 1.0
            visits = _visits_fixed(end=end_i, step=step)
        elif method == "poisson":
            rate = params['rate'] if 'rate' in params else 1.0
            visits = _visits_poisson(rng, end=end_i, rate=rate)
        elif method == "lognormal":
            ln_mean = params['ln_mean'] if 'ln_mean' in params else 0.0
            ln_sigma = params['ln_sigma'] if 'ln_sigma' in params else 1.0
            visits = _visits_lognormal(rng, end=end_i, mean=ln_mean, sigma=ln_sigma)
        elif method == "exp":
            rate = params['rate'] if 'rate' in params else 2.0            
            visits = _visits_exp(rng, end=end_i, rate = rate)
        elif method == "hawkes":
            mu = params['mu'] if 'mu' in params else 0.2
            alpha = params['alpha'] if 'alpha' in params else 0.3
            beta = params['beta'] if 'beta' in params else 1.5      
            visits = _hawkes_times(rng, end=end_i, mu = mu, alpha = alpha, beta = beta)
        else:
            raise ValueError("method must be 'fixed'|'poisson'|'lognormal'|'hawkes'|'exp'")

        n_visits[i] = visits.size

        t = float(e[i])
        # right censor
        if t > c[i] or t > visits[-1]:
            left[i] = c[i]
            right[i] = np.inf
            continue

        # interval censor: find j let visits[j] < t <= visits[j+1]
        j = int(np.searchsorted(visits, t, side="right") - 1)
        # print (visits, j, t)
        if np.isclose(t, float(visits[j])): 
            left[i] = right[i] = t  # exact        
        elif np.isclose(t, float(visits[j + 1])):
            left[i] = right[i] = t  # exact
        else:
            L, U = float(visits[j]), float(visits[j + 1])                
            left[i], right[i] = L, U

    return left, right, n_visits

