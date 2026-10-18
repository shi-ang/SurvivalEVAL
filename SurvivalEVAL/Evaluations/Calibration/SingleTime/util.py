"""Shared calibration-curve diagnostics for evaluator reports."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from SurvivalEVAL.Evaluations.util import fit_least_squares
from SurvivalEVAL.Evaluations.util_plots import pp_plot


def _calibration_curve_details(observed: list, expected: list) -> dict:
    """Describe and plot grouped calibration with predicted risk on the x-axis."""
    obs, exp = np.asarray(observed), np.asarray(expected)
    order = np.argsort(exp, kind="stable")
    slope, intercept = fit_least_squares(exp, obs, left_anchor=True, right_anchor=True)

    fig, ax = plt.subplots()
    groups = np.arange(1, len(obs) + 1)
    ax.bar(groups - 0.2, obs[order], width=0.4, label="Observed", alpha=0.7)
    ax.bar(groups + 0.2, exp[order], width=0.4, label="Expected", alpha=0.7)
    ax.set_xlabel("Groups (from lowest to highest predicted probability)")
    ax.set_xticks(groups)
    ax.set_ylabel("Probabilities")
    ax.legend()
    fig.tight_layout()

    return {
        "slope": slope,
        "intercept": intercept,
        "max_local_deviation": _maximum_local_deviation(obs, exp),
        "histogram_plot": (fig, ax),
        "pp_plot": pp_plot(
            obs[order], exp[order], xlim=(-0.05, 1.05), ylim=(-0.05, 1.05)
        ),
    }


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
