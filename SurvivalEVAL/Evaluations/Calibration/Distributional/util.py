"""Kolmogorov-Smirnov helpers for distributional calibration metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import kstwobign


def discrepancy_to_uniform(
    x: np.ndarray, cdf: np.ndarray, x_support: tuple[float, float] | None = None
) -> float:
    """
    Compute the Kolmogorov-Smirnov (KS) statistic for one-sample test against uniform distribution.
    The KS statistic is defined as:
        D_n = max|F_n(x) - F(x)|,
    where F_n(x) is the empirical CDF and F(x) is the CDF of the uniform distribution.
    Parameters
    ----------
    x: np.ndarray
        The support points of the empirical CDF.
    cdf: np.ndarray
        The values of the empirical CDF at the support points.
    x_support: tuple[float, float] | None
        The support points of the uniform distribution. If None, it is assumed to be [0, 1].
    Returns
    -------
    D_n: float
        The KS statistic.
    """
    # sanity check
    if x.ndim != 1 or cdf.ndim != 1 or x.size != cdf.size:
        raise ValueError("x and cdf must be 1D arrays of the same length.")

    assert np.all(cdf >= 0) and np.all(
        cdf <= 1
    ), "The cdf values must be in the range [0, 1]."

    if not (np.all(np.diff(x) >= 0) and np.all(np.diff(cdf) >= 0)):
        raise ValueError("x and cdf must be nondecreasing.")

    # empirical CDF values at the support points
    F_n = cdf
    # CDF of the uniform distribution at the support points
    if x_support is None:
        F = x
    else:
        F = (x - x_support[0]) / (x_support[1] - x_support[0])

    D_plus = np.max(F_n - F)
    D_minus = np.max(F - np.concatenate(([0], F_n[:-1])))
    D_n = float(max(D_plus, D_minus))
    return D_n


def ks_pvalue(D_n: float, n: int) -> float:
    """
    Compute asymptotic KS one-sample p-value using SciPy's Kolmogorov distribution.
    This function has the same behavior as the last half part of
    scipy.stats.kstest(method='asymp') for one-sample KS test.

    But this function directly takes D_n and n as input, which is more convenient for our use case.

    The reason we do not use 'exact' method from scipy.stats.kstest is that it assumes continuous distribution,
    while in our case, the distribution estimated by empirical estimator is discrete.
    """
    lambda_n = np.sqrt(n) * D_n
    # Survival function (1 - CDF) gives the p-value directly
    return float(kstwobign.sf(lambda_n))
