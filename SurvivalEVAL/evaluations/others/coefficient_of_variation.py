"""
Coefficient of variation for event-time prediction distributions.

Author: Weijie Sun
"""

from __future__ import annotations

import numpy as np


def cov(
    cdf: np.ndarray,
    t_grid: np.ndarray,
    return_details: bool = False,
) -> float | tuple[float, np.ndarray]:
    """
    Compute the coefficient of variation of event time from a discretized CDF.

    The coefficient of variation, standard deviation divided by mean event time,
    is used to assess predictive sharpness in [1]_.

    Parameters
    ----------
    cdf : ndarray, shape (n_samples, n_timepoints)
        The predicted cumulative distribution functions for each patient.
    t_grid : ndarray, shape (n_timepoints,)
        The time grid corresponding to the CDF values.
    return_details : bool, optional
        If True, also return the per-patient CoV values.

    Returns
    -------
    float
        The mean event-time CoV across all patients.
    ndarray, optional
        The per-patient event-time CoV values, returned when
        ``return_details`` is True.

    Notes
    -----
    For each patient, the increment
    ``F(t_grid[k + 1]) - F(t_grid[k])`` is assigned to the midpoint of that
    interval. These increments are divided by their sum,
    ``F(t_grid[-1]) - F(t_grid[0])``, before the event-time moments are
    calculated.

    Consequently, if ``F(t_grid[0]) = 0`` and ``F(t_grid[-1]) = 1``, this
    approximates the CoV of the full event-time distribution. Otherwise, it is
    the CoV conditional on the event occurring within the grid interval; mass
    before the first grid point and after the last grid point is excluded.

    References
    ----------
    .. [1] Avati, A.; Duan, T.; Zhou, S.; Jung, K.; Shah, N. H.; Ng, A. Y. (2020).
       "Countdown Regression: Sharp and Calibrated Survival Predictions".
       Proceedings of the 35th Uncertainty in Artificial Intelligence Conference,
       Proceedings of Machine Learning Research, 115, 145-155.
       https://arxiv.org/pdf/1806.08324
    """
    cdf = np.asarray(cdf, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)

    if cdf.ndim != 2:
        raise ValueError("cdf must be a two-dimensional array.")
    if t_grid.ndim != 1:
        raise ValueError("t_grid must be a one-dimensional array.")
    if t_grid.size < 2 or cdf.shape[1] != t_grid.size:
        raise ValueError("cdf and t_grid must contain the same number of time points.")
    if not np.all(np.isfinite(cdf)) or not np.all(np.isfinite(t_grid)):
        raise ValueError("cdf and t_grid must contain only finite values.")
    if np.any(t_grid < 0) or np.any(np.diff(t_grid) <= 0):
        raise ValueError("t_grid must be nonnegative and strictly increasing.")
    if np.any((cdf < 0) | (cdf > 1)):
        raise ValueError("cdf values must lie between 0 and 1.")

    cdf_increments = np.diff(cdf, axis=1)
    tolerance = 1e-12
    if np.any(cdf_increments < -tolerance):
        raise ValueError("cdf must be nondecreasing along the time axis.")

    # Treat each CDF increment as event-time mass at its bin midpoint.
    probability_mass = np.clip(cdf_increments, 0.0, None)
    represented_mass = np.sum(probability_mass, axis=1)
    if np.any(represented_mass <= tolerance):
        raise ValueError(
            "Each CDF must contain positive probability mass on the time grid."
        )
    probability_mass = probability_mass / represented_mass[:, None]

    time_midpoints = 0.5 * (t_grid[1:] + t_grid[:-1])
    mean_time = np.sum(probability_mass * time_midpoints, axis=1)
    if np.any(mean_time <= 0):
        raise ValueError("Mean event time must be positive; CoV is undefined.")

    variance_time = np.sum(
        probability_mass * (time_midpoints - mean_time[:, None]) ** 2,
        axis=1,
    )
    cov_values = np.sqrt(variance_time) / mean_time
    mean_cov = float(np.mean(cov_values))
    if return_details:
        return mean_cov, cov_values
    else:
        return mean_cov
