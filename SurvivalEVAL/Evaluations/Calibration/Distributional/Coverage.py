"""Prediction-interval coverage for interval-censored survival outcomes."""

from __future__ import annotations

import numpy as np

from SurvivalEVAL.NonparametricEstimator.SingleEvent import TurnbullEstimatorLifelines


def coverage_ic(
    pred_l: np.ndarray,
    pred_r: np.ndarray,
    obs_l: np.ndarray,
    obs_r: np.ndarray,
    obs_l_train: np.ndarray | None = None,
    obs_r_train: np.ndarray | None = None,
    cov_level: float = 0.95,
    method: str = "Turnbull",
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """
    Compute the Interval-Censor Coverage (IC) metric.

    The coverage is the proportion of instances where the predicted interval [t_li, t_ui]
    contains the true event time.
    Since the true event time is interval-censored between [L_i, U_i], we consider the event time to be contained
    - 100% if the predicted interval fully covers the censoring interval [L_i, U_i]
    - 0% if there is no overlap between the predicted interval and the censoring interval [L_i, U_i]
    - partial coverage if the predicted interval partially overlaps the censoring interval [L_i, U_i]

    The partial coverage is estimated using the empirical distribution of censoring intervals from the training data.
    Like what we did for the partial weights for concordance_ic().
    That means we estimate the empirical CDF of censoring intervals [L_j, U_j] from the training data
    using the Turnbull estimator,
    and use it to compute S(L_i), S(U_i), S(t_li), S(t_ui) for each test instance.
    The partial coverage is then calculated as:
        partial_coverage = (S(max(L_i, t_li)) - S(min(U_i, t_ui))) / (S(L_i) - S(U_i))

    Parameters
    ----------
    pred_l : ndarray, shape (n_samples,)
        The lower bounds of the predicted intervals.
    pred_r : ndarray, shape (n_samples,)
        The upper bounds of the predicted intervals.
    obs_l : ndarray, shape (n_samples,)
        The lower bounds of the observed censoring intervals.
    obs_r : ndarray, shape (n_samples,)
        The upper bounds of the observed censoring intervals.
    obs_l_train : ndarray, shape (n_train_samples,)
        The lower bounds of the observed censoring intervals in the training data.
    obs_r_train : ndarray, shape (n_train_samples,)
        The upper bounds of the observed censoring intervals in the training data.
    cov_level : float, default 0.95
        Target coverage level for calibration reference.
    method : str, default "Turnbull"
        Method to compute the coverage.
        - "Turnbull": use the empirical distribution (Turnbull estimator) of censoring intervals from the training data.
        - "linear": use the overlap fraction for finite observed intervals.
          For unbounded observed intervals, finite overlap contributes 0 and
          unbounded overlap contributes 1, the limit as the upper bound grows.
    eps : float, default 1e-12
        Numerical tolerance for treating the conditional coverage denominator as zero.

    Returns
    -------
    observed_cov : float
        Average (partial) coverage across samples.
    cov_gap : float
        Difference between observed coverage and the target level (observed_cov - cov_level).
    avg_length : float
        Average length of the predicted intervals. Infinite lower bounds
        indicate unbounded predicted quantiles and give infinite width.
        Predictions [inf, inf] cover no finite event time.
    """
    if pred_l.ndim != 1 or pred_r.ndim != 1:
        raise ValueError("pred_l and pred_r must be 1-dimensional arrays.")
    if obs_l.ndim != 1 or obs_r.ndim != 1:
        raise ValueError("obs_l and obs_r must be 1-dimensional arrays.")
    if not (pred_l.shape == pred_r.shape == obs_l.shape == obs_r.shape):
        raise ValueError(
            "pred_l, pred_r, obs_l, and obs_r must contain the same number of samples."
        )

    overlap_left = np.maximum(obs_l, pred_l)
    overlap_right = np.minimum(obs_r, pred_r)
    intersects = (overlap_left <= overlap_right) & np.isfinite(overlap_left)

    method = method.lower()
    if method == "linear":
        denom = obs_r - obs_l
        numer = np.zeros_like(denom, dtype=float)
        np.subtract(overlap_right, overlap_left, out=numer, where=intersects)
    elif method == "turnbull":
        if obs_l_train is None or obs_r_train is None:
            raise ValueError(
                "obs_l_train and obs_r_train must be provided for Turnbull method."
            )
        if obs_l_train.ndim != 1 or obs_r_train.ndim != 1:
            raise ValueError(
                "obs_l_train and obs_r_train must be 1-dimensional arrays."
            )
        if np.any(obs_l_train > obs_r_train):
            raise ValueError("Found training intervals with left > right.")

        tb = TurnbullEstimatorLifelines(obs_l_train, obs_r_train)
        denom = tb.predict(obs_l) - tb.predict(obs_r)
        numer = tb.predict(overlap_left) - tb.predict(overlap_right)
    else:
        raise ValueError(f"Unknown method: {method}")

    coverage = intersects.astype(float)
    np.divide(numer, denom, out=coverage, where=(denom > eps) & np.isfinite(denom))
    if method == "linear":
        unbounded = np.isposinf(denom)
        coverage[unbounded] = intersects[unbounded] & np.isposinf(
            overlap_right[unbounded]
        )
    coverage = np.clip(coverage, 0.0, 1.0)

    widths = np.full(pred_l.shape, np.inf)
    np.subtract(pred_r, pred_l, out=widths, where=np.isfinite(pred_l))
    observed_cov = float(np.mean(coverage))
    return observed_cov, observed_cov - float(cov_level), float(np.mean(widths))
