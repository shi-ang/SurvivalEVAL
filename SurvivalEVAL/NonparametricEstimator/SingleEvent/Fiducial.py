"""
Pure-Python implementation of Algorithm 1 from:
"A unified nonparametric fiducial approach to interval-censored data"
(Cui et al., 2023)

This module provides a fiducial inference method for interval-censored survival data.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Literal
import warnings
import osqp
from scipy import sparse
from scipy.optimize import minimize, Bounds


# Numerical constants
_EPS = 1e-8  # Small epsilon for numerical stability
_RIDGE = 1e-8  # Ridge regularization for PSD enforcement


def _validate_grid(grid_low: float, grid_high: float, ngrid: int) -> Tuple[float, float]:
    """
    Validate and adjust grid bounds to ensure numerical stability.
    
    Returns adjusted (grid_low, grid_high).
    """
    # Handle NaN/Inf
    if not np.isfinite(grid_low):
        grid_low = 0.0
    if not np.isfinite(grid_high):
        grid_high = 100.0  # Default fallback
    
    # Ensure grid_low < grid_high
    if grid_low >= grid_high:
        # Add small margin
        margin = max(abs(grid_low) * 0.1, 1.0)
        grid_high = grid_low + margin
        warnings.warn(f"grid_low >= grid_high, adjusted to [{grid_low}, {grid_high}]")
    
    # Ensure minimum spacing
    min_range = _EPS * ngrid * 10
    if (grid_high - grid_low) < min_range:
        grid_high = grid_low + min_range
        warnings.warn(f"Grid range too small, adjusted to [{grid_low}, {grid_high}]")
    
    return grid_low, grid_high


def _build_qp_matrices(ngrid: int, grid: np.ndarray, lam: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the quadratic programming matrices Q (Hessian) for linear interpolation.
    
    The objective is to minimize:
        0.5 * w^T Q w + q^T w
    subject to box constraints fid_lower <= w <= fid_upper.
    
    Q is a tridiagonal matrix as defined in LinInterpolation.R:
        dgrid = 1 / ((ngrid * diff(grid))^2)
        
        # R code: diag(c(lambda, dgrid) + c(dgrid, lambda))
        # This means:
        #   Q[0,0]         = lambda + dgrid[0]
        #   Q[i,i]         = dgrid[i-1] + dgrid[i]  for i in 1..ngrid-2 (NO lambda!)
        #   Q[ngrid-1,ngrid-1] = dgrid[ngrid-2] + lambda
        
        Q[i,i-1] = Q[i-1,i] = -dgrid[i-1]
    
    Parameters
    ----------
    ngrid : int
        Number of grid points
    grid : np.ndarray
        Grid points (length ngrid)
    lam : float
        Regularization parameter (default 1.0)
    
    Returns
    -------
    Q : np.ndarray
        Hessian matrix (ngrid x ngrid)
    dgrid : np.ndarray
        Differences used in the matrix (length ngrid-1)
    """
    diff_grid = np.diff(grid)
    
    # Ensure diff_grid is strictly positive to avoid division by zero
    diff_grid = np.maximum(diff_grid, _EPS)
    
    dgrid = 1.0 / ((ngrid * diff_grid) ** 2)
    
    # Cap dgrid to avoid numerical overflow
    max_dgrid = 1e12
    dgrid = np.minimum(dgrid, max_dgrid)
    
    # Build tridiagonal matrix exactly as in R:
    # diag(c(lambda, dgrid) + c(dgrid, lambda))
    # c(lambda, dgrid) has length ngrid: [lambda, dgrid[0], dgrid[1], ..., dgrid[ngrid-2]]
    # c(dgrid, lambda) has length ngrid: [dgrid[0], dgrid[1], ..., dgrid[ngrid-2], lambda]
    # Sum gives diagonal: [lambda+dgrid[0], dgrid[0]+dgrid[1], ..., dgrid[ngrid-3]+dgrid[ngrid-2], dgrid[ngrid-2]+lambda]
    
    Q = np.zeros((ngrid, ngrid), dtype=np.float64)
    
    # Diagonal elements (exactly as R: diag(c(lambda, dgrid) + c(dgrid, lambda)))
    # First element: lambda + dgrid[0]
    Q[0, 0] = lam + dgrid[0]
    # Interior elements: dgrid[i-1] + dgrid[i] (NO lambda)
    for i in range(1, ngrid - 1):
        Q[i, i] = dgrid[i - 1] + dgrid[i]
    # Last element: dgrid[ngrid-2] + lambda
    Q[ngrid - 1, ngrid - 1] = dgrid[ngrid - 2] + lam
    
    # Off-diagonal elements
    for i in range(1, ngrid):
        Q[i, i - 1] = -dgrid[i - 1]
        Q[i - 1, i] = -dgrid[i - 1]
    
    # Ensure symmetry and add small ridge for numerical stability (PSD enforcement)
    Q = (Q + Q.T) / 2.0
    Q += np.eye(ngrid) * _RIDGE
    
    return Q, dgrid


def _linear_interpolation_qp(
    grid: np.ndarray,
    fid_lower: np.ndarray,
    fid_upper: np.ndarray,
    Q: np.ndarray,
    lam: float,
    rng: np.random.Generator,
    solver: str = "osqp",
    osqp_solver: Optional[object] = None
) -> Tuple[np.ndarray, Optional[object]]:
    """
    Solve the quadratic programming problem for linear interpolation.
    
    Lines 21-25 in Algorithm 1: Linear interpolations via quadratic programming.
    
    Parameters
    ----------
    grid : np.ndarray
        Grid points
    fid_lower : np.ndarray
        Lower bounds from fiducial samples
    fid_upper : np.ndarray
        Upper bounds from fiducial samples
    Q : np.ndarray
        Hessian matrix
    lam : float
        Regularization parameter
    rng : np.random.Generator
        Random number generator
    solver : str
        Solver to use ("osqp" or "scipy")
    osqp_solver : optional
        Pre-built OSQP solver for warm-starting
    
    Returns
    -------
    w : np.ndarray
        Solution vector
    osqp_solver : optional
        Updated OSQP solver (for warm-starting)
    """
    ngrid = len(grid)
    
    # Ensure bounds are valid (lb <= ub) and finite
    fid_lower = np.asarray(fid_lower, dtype=np.float64)
    fid_upper = np.asarray(fid_upper, dtype=np.float64)
    
    # Handle NaN/Inf in bounds
    fid_lower = np.nan_to_num(fid_lower, nan=0.0, posinf=1.0, neginf=0.0)
    fid_upper = np.nan_to_num(fid_upper, nan=1.0, posinf=1.0, neginf=0.0)
    
    # Ensure lb <= ub
    invalid_mask = fid_lower > fid_upper
    if np.any(invalid_mask):
        # Swap or set to midpoint
        mid = (fid_lower[invalid_mask] + fid_upper[invalid_mask]) / 2
        fid_lower[invalid_mask] = mid
        fid_upper[invalid_mask] = mid
    
    # Clip to [0, 1]
    fid_lower = np.clip(fid_lower, 0.0, 1.0)
    fid_upper = np.clip(fid_upper, 0.0, 1.0)
    
    # Ensure minimum gap to avoid degeneracy
    gap = fid_upper - fid_lower
    min_gap = _EPS
    too_small = gap < min_gap
    if np.any(too_small):
        mid = (fid_lower[too_small] + fid_upper[too_small]) / 2
        fid_lower[too_small] = np.maximum(0.0, mid - min_gap / 2)
        fid_upper[too_small] = np.minimum(1.0, mid + min_gap / 2)
    
    # Randomize boundary conditions using Beta(0.5, 0.5)
    w0 = rng.beta(0.5, 0.5) * (fid_upper[0] - fid_lower[0]) + fid_lower[0]
    wEnd = rng.beta(0.5, 0.5) * (fid_upper[-1] - fid_lower[-1]) + fid_lower[-1]
    
    # Linear term q
    q = np.zeros(ngrid, dtype=np.float64)
    q[0] = -lam * w0
    q[-1] = -lam * wEnd
    
    # Try OSQP first, fall back to scipy on failure
    if solver == "osqp":
        return _solve_qp_osqp(Q, q, fid_lower, fid_upper, osqp_solver)
    else:
        return _solve_qp_scipy(Q, q, fid_lower, fid_upper), None


def _solve_qp_osqp(
    Q: np.ndarray,
    q: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    solver: Optional[object] = None
) -> Tuple[np.ndarray, object]:
    """
    Solve QP using OSQP with warm-starting.
    
    minimize 0.5 * x^T P x + q^T x
    subject to lb <= x <= ub
    
    OSQP formulation requires: l <= Ax <= u
    We use A = I (identity) for box constraints.
    """
    n = len(q)
    
    # Ensure Q is symmetric and convert to sparse
    Q = np.asarray(Q, dtype=np.float64)
    Q = (Q + Q.T) / 2.0
    
    # Check for NaN/Inf in inputs
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q matrix contains NaN/Inf")
    if not np.all(np.isfinite(q)):
        raise ValueError("q vector contains NaN/Inf")
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ValueError("Bounds contain NaN/Inf")
    
    P = sparse.csc_matrix(Q)
    A = sparse.eye(n, format='csc')
    
    if solver is None:
        # Create new solver
        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=A, l=lb, u=ub,
                     verbose=False, polish=True,
                     eps_abs=_EPS, eps_rel=_EPS,
                     max_iter=4000)
    else:
        # Warm-start: update only q, l, u
        solver.update(q=q, l=lb, u=ub)
    
    result = solver.solve()
    
    if result.info.status not in ['solved', 'solved_inaccurate']:
        # Fallback to scipy if OSQP fails
        warnings.warn(f"OSQP did not converge (status: {result.info.status}), falling back to scipy")
        return _solve_qp_scipy(Q, q, lb, ub), solver
    
    # Clip result to bounds (numerical safety)
    x = np.clip(result.x, lb, ub)
    return x, solver


def _solve_qp_scipy(
    Q: np.ndarray,
    q: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray
) -> np.ndarray:
    """
    Solve QP using scipy.optimize.minimize (L-BFGS-B with bounds).
    
    minimize 0.5 * x^T Q x + q^T x
    subject to lb <= x <= ub
    """
    n = len(q)
    
    # Handle any remaining NaN/Inf
    lb = np.nan_to_num(lb, nan=0.0, posinf=1.0, neginf=0.0)
    ub = np.nan_to_num(ub, nan=1.0, posinf=1.0, neginf=0.0)
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    
    def objective(x):
        return 0.5 * x @ Q @ x + q @ x
    
    def gradient(x):
        return Q @ x + q
    
    # Initial guess: midpoint of bounds
    x0 = 0.5 * (lb + ub)
    x0 = np.clip(x0, lb, ub)
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        jac=gradient,
        bounds=Bounds(lb, ub),
        options={'maxiter': 2000, 'ftol': _EPS}
    )
    
    # Clip result to bounds
    return np.clip(result.x, lb, ub)


def _compute_bounds_fast(
    u: np.ndarray,
    l: np.ndarray,
    r: np.ndarray,
    query_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast computation of fiducial bounds using sorting and searchsorted.
    
    For each query point s:
        lower(s) = max{u[j] : r[j] <= s} or 0 if no such j
        upper(s) = min{u[j] : l[j] > s} or 1 if no such j
    
    Parameters
    ----------
    u : np.ndarray
        Current u values (length n)
    l : np.ndarray
        Left endpoints of intervals (length n)
    r : np.ndarray
        Right endpoints of intervals (length n)
    query_points : np.ndarray
        Points at which to evaluate bounds
    
    Returns
    -------
    lower : np.ndarray
        Lower bounds at query points
    upper : np.ndarray
        Upper bounds at query points
    """
    n = len(u)
    nq = len(query_points)
    
    # For lower bound: max u among r <= s
    # Sort by r, then compute cumulative max of u
    r_order = np.argsort(r)
    r_sorted = r[r_order]
    u_by_r = u[r_order]
    cummax_u = np.maximum.accumulate(u_by_r)
    
    # For each query point s, find rightmost r <= s
    # searchsorted gives index where s would be inserted to maintain order
    # So index-1 gives the rightmost element <= s
    r_indices = np.searchsorted(r_sorted, query_points, side='right') - 1
    lower = np.where(r_indices >= 0, cummax_u[r_indices], 0.0)
    
    # For upper bound: min u among l > s
    # Sort by l in ascending order, then compute reverse cumulative min
    l_order = np.argsort(l)
    l_sorted = l[l_order]
    u_by_l = u[l_order]
    
    # Compute reverse cumulative min (from end to beginning)
    # cummin_rev[i] = min(u_by_l[i:])
    cummin_rev = np.minimum.accumulate(u_by_l[::-1])[::-1]
    
    # searchsorted on ascending l, find first index where l > s
    # side='right' gives first index where l_sorted > query_points
    l_indices = np.searchsorted(l_sorted, query_points, side='right')
    
    # Clip indices to valid range, then use where to handle out-of-bounds
    l_indices_clipped = np.clip(l_indices, 0, n - 1)
    upper = np.where(l_indices < n, cummin_rev[l_indices_clipped], 1.0)
    
    return lower, upper


def _gibbs_step(
    u: np.ndarray,
    l: np.ndarray,
    r: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """
    One full sweep of the Gibbs sampler (Lines 6-11 in Algorithm 1).
    
    For each i:
        u_lower = max{u[j] : j != i, r[j] <= l[i]} or 0
        u_upper = min{u[j] : j != i, l[j] >= r[i]} or 1
        u[i] ~ Uniform(u_lower, u_upper)
    
    Parameters
    ----------
    u : np.ndarray
        Current u values (modified in place)
    l : np.ndarray
        Left endpoints
    r : np.ndarray
        Right endpoints
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    u : np.ndarray
        Updated u values
    """
    n = len(u)
    
    for i in range(n):
        # Exclude index i
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        
        u_pre = u[mask]
        l_pre = l[mask]
        r_pre = r[mask]
        
        # u_lower = max(u[j] for j != i if r[j] <= l[i]) else 0
        idx1 = r_pre <= l[i]
        u_lower = np.max(u_pre[idx1]) if np.any(idx1) else 0.0
        
        # u_upper = min(u[j] for j != i if l[j] >= r[i]) else 1
        idx2 = l_pre >= r[i]
        u_upper = np.min(u_pre[idx2]) if np.any(idx2) else 1.0
        
        # Sample new u[i]
        u[i] = rng.uniform(u_lower, u_upper)
    
    return u


def _rank_resample(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Rank-resample step (Line 12 in Algorithm 1).
    
    temp ~ Uniform(0, 1, n)
    u = sort(temp)[rank(u)]
    
    This preserves the ordering of u while resampling the values.
    """
    n = len(u)
    temp = rng.uniform(0, 1, n)
    temp_sorted = np.sort(temp)
    
    # rank(u) gives the position each element would have in sorted order
    # In R, rank() returns 1-based ranks; we need 0-based for indexing
    ranks = np.argsort(np.argsort(u))
    
    return temp_sorted[ranks]


def _isotonic_regression(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
    
    Finds the non-decreasing sequence closest to y in L2 sense.
    Useful for enforcing monotonicity in CDF estimates.
    
    Parameters
    ----------
    y : np.ndarray
        Input array (possibly non-monotonic)
    
    Returns
    -------
    y_iso : np.ndarray
        Isotonic (non-decreasing) approximation
    """
    n = len(y)
    y_iso = y.copy()
    
    # PAVA algorithm
    i = 0
    while i < n - 1:
        if y_iso[i] > y_iso[i + 1]:
            # Pool adjacent blocks
            j = i + 1
            while j < n and y_iso[j] < y_iso[i]:
                j += 1
            # Average the pooled block
            pool_mean = np.mean(y_iso[i:j])
            y_iso[i:j] = pool_mean
            # Go back to check for new violations
            i = max(0, i - 1)
        else:
            i += 1
    
    # Clip to [0, 1] for CDF
    y_iso = np.clip(y_iso, 0.0, 1.0)
    
    return y_iso


def fit_fiducial_interval_censor(
    l: np.ndarray,
    r: np.ndarray,
    mfid: int = 1000,
    mburn: int = 100,
    alpha: float = 0.05,
    ngrid: int = 100,
    ntest: int = 200,
    grid_low: Optional[float] = None,
    grid_high: Optional[float] = None,
    grid_high_override: Optional[float] = None,
    lam: float = 1.0,
    seed: int = 123,
    solver: Literal["osqp", "scipy"] = "osqp",
    enforce_monotonicity: bool = False
) -> Dict:
    """
    Fit fiducial inference for interval-censored data.
    
    Implements Algorithm 1 from "A unified nonparametric fiducial approach to 
    interval-censored data" (Cui et al., 2023).
    
    Parameters
    ----------
    l : np.ndarray
        Left endpoints of intervals (length n)
    r : np.ndarray
        Right endpoints of intervals (length n)
    mfid : int
        Number of fiducial samples (default 1000)
    mburn : int
        Number of burn-in samples (default 100)
    alpha : float
        Confidence level for intervals (default 0.05, gives 95% CI)
    ngrid : int
        Number of internal grid points for QP (default 100)
    ntest : int
        Number of test grid points for output (default 200)
    grid_low : float, optional
        Lower bound of grid (default: min(l))
    grid_high : float, optional
        Upper bound of grid (default: max(r))
    grid_high_override : float, optional
        Override for upper bound of grid (takes precedence)
    lam : float
        Regularization parameter for QP (default 1.0)
    seed : int
        Random seed (default 123)
    solver : str
        QP solver: "osqp" (default) or "scipy"
    enforce_monotonicity : bool
        If True, apply isotonic regression to enforce monotonicity (default False)
    
    Returns
    -------
    dict with keys:
        testgrid : np.ndarray (ntest,)
            Test grid points
        FiducialMidLine : np.ndarray (mfid, ntest)
            Fiducial samples of CDF at testgrid
        F_median : np.ndarray (ntest,)
            Median CDF estimate
        F_low : np.ndarray (ntest,)
            Lower (alpha/2) quantile of CDF
        F_high : np.ndarray (ntest,)
            Upper (1-alpha/2) quantile of CDF
        S_hat : np.ndarray (ntest,)
            Median survival estimate (1 - F_median)
        S_low : np.ndarray (ntest,)
            Lower survival bound (1 - F_high)
        S_high : np.ndarray (ntest,)
            Upper survival bound (1 - F_low)
        F_mean : np.ndarray (ntest,)
            Mean CDF estimate (for debugging)
        S_mean : np.ndarray (ntest,)
            Mean survival estimate (for debugging)
    
    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> n = 50
    >>> true_times = rng.exponential(5, n)
    >>> l = np.maximum(0, true_times - rng.uniform(0, 2, n))
    >>> r = true_times + rng.uniform(0, 2, n)
    >>> result = fit_fiducial_interval_censor(l, r, mfid=100, mburn=20)
    >>> print(result['F_median'].shape)
    (200,)
    """
    # Input validation
    l = np.asarray(l, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    
    if l.shape != r.shape:
        raise ValueError("l and r must have the same shape")
    if np.any(l > r):
        raise ValueError("Left endpoints must be <= right endpoints")
    if np.any(l < 0):
        raise ValueError("Left endpoints must be non-negative")
    
    n = len(l)
    
    if n == 0:
        raise ValueError("Input arrays must not be empty")
    
    # Identify right-censored observations (r = Inf)
    # Unlike previous approach, we keep r=Inf for Gibbs sampling and bounds computation
    # because the R code naturally handles Inf in comparisons:
    #   s >= Inf is FALSE for finite s
    #   Inf >= s is TRUE for finite s
    # So right-censored obs don't contribute to lower bounds in R, and same in Python
    inf_mask = ~np.isfinite(r)
    
    # For setting grid_high, only use finite r values
    '''
    r_for_grid = r.copy()
    if np.any(inf_mask):
        if np.any(~inf_mask):
            finite_r_max = np.max(r[~inf_mask])
        else:
            finite_r_max = np.max(l) * 2
        r_for_grid[inf_mask] = finite_r_max  # Only used for grid_high calculation in the initial round.
    '''
    finite_r = r[np.isfinite(r)]


    # Set grid bounds
    if grid_low is None:
        grid_low = float(np.min(l))

    '''
    if grid_high is None:
        grid_high = float(np.max(r_for_grid))
    '''
    if grid_high is None:
        if finite_r.size > 0:
            grid_high = float(np.max(finite_r))   # default matches R wrapper idea
        else:
            # No finite right endpoints (e.g., no observed events).
            # Use a conservative finite window; better: require user to pass grid_high_override.
            grid_high = float(np.max(l))          # NOT 2*max(l)

    if grid_high_override is not None:
        grid_high = float(grid_high_override)
    
    # Validate and adjust grid bounds
    grid_low, grid_high = _validate_grid(grid_low, grid_high, ngrid)
    
    # Create grids
    grid = np.linspace(grid_low, grid_high, ngrid)
    testgrid = np.linspace(grid_low, grid_high, ntest)
    
    # Verify grid is valid
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"Grid contains NaN/Inf. grid_low={grid_low}, grid_high={grid_high}")
    
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    
    # Initialize u (Lines 1-3 in Algorithm 1)
    # For midpoint calculation, use finite substitute for Inf values
    r_for_mid = r.copy()
    '''
    if np.any(inf_mask):
        # Use a finite value for midpoint ordering only
        # This matches R behavior where (l + Inf) / 2 = Inf, but we need finite for sorting
        r_for_mid[inf_mask] = r_for_grid[inf_mask]  # Use the grid_high based value
    '''        
    r_for_mid[~np.isfinite(r_for_mid)] = grid_high

    mid = (l + r_for_mid) / 2
    u = np.sort(rng.uniform(0, 1, n))
    u = u[np.argsort(np.argsort(mid))]  # Reorder by midpoint order
    
    # Build QP matrices (constant across iterations)
    Q, _ = _build_qp_matrices(ngrid, grid, lam) # lambda parameter in LinInterpolation.R
    
    # Verify Q matrix is valid
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q matrix contains NaN/Inf after construction")

    # Storage for fiducial samples
    FiducialMidLine = np.zeros((mfid, ntest))
    
    # OSQP solver for warm-starting
    osqp_solver = None
    
    # Main loop
    total_iter = mfid + mburn
    for j in range(total_iter):
        # Gibbs sampler (Lines 6-11) - use r directly (with Inf), 
        # numpy comparisons handle Inf correctly like R does
        u = _gibbs_step(u, l, r, rng)
        
        # Rank-resample step (Line 12)
        u = _rank_resample(u, rng)
        
        # Skip burn-in samples
        if j < mburn:
            continue
        
        # Compute fiducial bounds on internal grid (Lines 15-20)
        # use r directly (with Inf), numpy comparisons handle Inf correctly
        fid_lower, fid_upper = _compute_bounds_fast(u, l, r, grid)
        
        # Linear interpolation via QP (Lines 21-25)
        w, osqp_solver = _linear_interpolation_qp(
            grid, fid_lower, fid_upper, Q, lam, rng, solver, osqp_solver
        )
        
        # Interpolate to testgrid
        F_interp = np.interp(testgrid, grid, w)
        
        # Optionally enforce monotonicity via isotonic regression
        if enforce_monotonicity:
            F_interp = _isotonic_regression(F_interp)
        
        # Store fiducial sample
        FiducialMidLine[j - mburn, :] = F_interp
    
    # Compute summary statistics
    F_median = np.median(FiducialMidLine, axis=0)
    F_mean = np.mean(FiducialMidLine, axis=0)
    F_low = np.percentile(FiducialMidLine, 100 * alpha / 2, axis=0)
    F_high = np.percentile(FiducialMidLine, 100 * (1 - alpha / 2), axis=0)
    
    # Survival function estimates
    S_hat = 1 - F_median
    S_mean = 1 - F_mean
    S_low = 1 - F_high
    S_high = 1 - F_low
    
    # Sanity checks
    _validate_results(FiducialMidLine, F_median, F_low, F_high, testgrid)
    
    return {
        'testgrid': testgrid,
        'FiducialMidLine': FiducialMidLine,
        'F_median': F_median,
        'F_low': F_low,
        'F_high': F_high,
        'S_hat': S_hat,
        'S_low': S_low,
        'S_high': S_high,
        'F_mean': F_mean,
        'S_mean': S_mean
    }


def _validate_results(
    FiducialMidLine: np.ndarray,
    F_median: np.ndarray,
    F_low: np.ndarray,
    F_high: np.ndarray,
    testgrid: np.ndarray
) -> None:
    """
    Validate fiducial inference results.
    
    Checks:
    (a) Each curve is within [0, 1]
    (b) Curves are non-decreasing (CDF property)
    (c) lower <= median <= upper pointwise
    """
    # (a) Check [0, 1] bounds
    if np.any(FiducialMidLine < -_EPS) or np.any(FiducialMidLine > 1 + _EPS):
        warnings.warn("Some fiducial samples are outside [0, 1] bounds")
    
    # (b) Check monotonicity (non-decreasing)
    diffs = np.diff(FiducialMidLine, axis=1)
    if np.any(diffs < -_EPS):
        n_violations = np.sum(diffs < -_EPS)
        warnings.warn(f"Monotonicity violated in {n_violations} locations")
    
    # (c) Check ordering: lower <= median <= upper
    if np.any(F_low > F_median + _EPS):
        warnings.warn("F_low > F_median detected")
    if np.any(F_median > F_high + _EPS):
        warnings.warn("F_median > F_high detected")


def _demo():
    """
    Demonstration of fiducial interval censoring with synthetic data.
    """
    print("=" * 60)
    print("Fiducial Interval Censoring Demo")
    print("=" * 60)

    # Generate synthetic interval-censored data
    np.random.seed(42)
    n = 100

    # True survival times from Weibull distribution
    true_times = np.random.weibull(2, n) * 5

    # Create interval censoring
    # Inspection times
    inspection_width = 2.0
    l = np.maximum(0, true_times - np.random.uniform(0, inspection_width, n))
    r = true_times + np.random.uniform(0, inspection_width, n)

    # Some right-censored (r = inf -> use large value)
    right_censored = np.random.random(n) < 0.1
    r[right_censored] = np.inf # np.max(r) * 1.5
    
    print(f"\nSample size: {n}")
    print(f"Number right-censored: {np.sum(right_censored)}")
    print(f"Time range: [{np.min(l):.2f}, {np.max(r):.2f}]")
    
    # Fit fiducial model
    print("\nFitting fiducial model...")
    result = fit_fiducial_interval_censor(
        l, r,
        mfid=500,
        mburn=50,
        alpha=0.05,
        ngrid=100,
        ntest=200,
        seed=123,
        solver="osqp"
    )
    
    print(f"\nResults:")
    print(f"  Test grid: {len(result['testgrid'])} points from "
          f"{result['testgrid'][0]:.2f} to {result['testgrid'][-1]:.2f}")
    print(f"  Fiducial samples shape: {result['FiducialMidLine'].shape}")
    
    # Summary statistics at selected time points
    idx_25 = len(result['testgrid']) // 4
    idx_50 = len(result['testgrid']) // 2
    idx_75 = 3 * len(result['testgrid']) // 4
    
    print("\nSurvival estimates at selected time points:")
    print("-" * 50)
    print(f"{'Time':>10} {'S_hat':>10} {'95% CI':>20}")
    print("-" * 50)
    
    for idx in [idx_25, idx_50, idx_75]:
        t = result['testgrid'][idx]
        s = result['S_hat'][idx]
        s_lo = result['S_low'][idx]
        s_hi = result['S_high'][idx]
        print(f"{t:10.2f} {s:10.3f} [{s_lo:8.3f}, {s_hi:8.3f}]")
    
    print("-" * 50)
    
    # Validation summary
    print("\nValidation:")
    F_in_bounds = np.all((result['F_median'] >= 0) & (result['F_median'] <= 1))
    F_monotonic = np.all(np.diff(result['F_median']) >= -_EPS)
    CI_valid = np.all(result['F_low'] <= result['F_median']) and \
               np.all(result['F_median'] <= result['F_high'])
    
    print(f"  CDF in [0,1]: {'✓' if F_in_bounds else '✗'}")
    print(f"  CDF monotonic: {'✓' if F_monotonic else '✗'}")
    print(f"  CI ordering valid: {'✓' if CI_valid else '✗'}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    _demo()