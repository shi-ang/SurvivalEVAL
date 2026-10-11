"""Shared calibration-curve diagnostics for evaluator reports."""

from __future__ import annotations

import numpy as np


def _maximum_local_deviation(observed, expected) -> float:
    """Measure the largest adjacent calibration slope or its reciprocal.

    Parameters
    ----------
    observed, expected : array-like, shape (n_bins,)
        Observed and expected probabilities in matching bin order.

    Returns
    -------
    float
        Maximum of each slope, diff(observed) / diff(expected), and its
        reciprocal. Returns infinity for a zero slope, or NaN if fewer
        than two bins remain or adjacent expected probabilities are tied.
    """
    increments = np.diff(expected)
    if increments.size == 0 or np.any(increments == 0):
        return np.nan
    slopes = np.diff(observed) / increments
    inverse_slopes = np.divide(
        1.0, slopes, out=np.full(slopes.shape, np.inf), where=slopes != 0
    )
    return float(np.max(np.maximum(slopes, inverse_slopes)))
