import numpy as np
from typing import Tuple, Literal, Optional

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

VisitMethod = Literal["fixed", "poisson", "lognormal"]

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

def interval_censor_DGP_from_synthetic_times(
    event_times: np.ndarray,           # (N,) true event time
    censoring_times: np.ndarray,       # (N,) right-censor/admin end per subject
    *,
    method: VisitMethod = "fixed",
    # fixed
    step: float = 1.0,
    # poisson
    rate: float = 4.0,                 # expected visits per unit time
    # lognormal
    ln_mean: float = 0.0,              # lognormal(mean=ln_mean, sigma=ln_sigma) for inter-visit gaps
    ln_sigma: float = 1.0,
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
            visits = _visits_fixed(end=end_i, step=step)
        elif method == "poisson":
            visits = _visits_poisson(rng, end=end_i, rate=rate)
        elif method == "lognormal":
            visits = _visits_lognormal(rng, end=end_i, mean=ln_mean, sigma=ln_sigma)
        else:
            raise ValueError("method must be 'fixed'|'poisson'|'lognormal'")

        n_visits[i] = visits.size

        t = float(e[i])
        # right censor
        if t > c[i]:
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
