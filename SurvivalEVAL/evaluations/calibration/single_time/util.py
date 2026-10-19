"""Shared probability transforms, spline bases, and calibration-curve diagnostics."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from patsy import dmatrix

from SurvivalEVAL.evaluations.util import fit_least_squares
from SurvivalEVAL.evaluations.util_plots import pp_plot


def _calibration_curve_details(
    observed: list[float] | np.ndarray,
    expected: list[float] | np.ndarray,
) -> dict:
    """
    Describe and plot grouped calibration with predicted risk on the x-axis.

    Parameters
    ----------
    observed : list[float] or ndarray
        Observed probabilities in calibration-bin order.
    expected : list[float] or ndarray
        Expected probabilities in the same calibration-bin order.

    Returns
    -------
    dict
        Slope, intercept, maximum local deviation, and histogram and probability-probability plots.
    """
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


def _maximum_local_deviation(
    observed: list[float] | np.ndarray,
    expected: list[float] | np.ndarray,
) -> float:
    """
    Measure the largest adjacent calibration slope or its reciprocal.

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


def _restricted_cubic_spline(
    x: np.ndarray,
    evaluation_x: np.ndarray,
    knots: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build and standardize a restricted cubic spline basis.

    Derive knots, centering, and scaling from x and reuse them when evaluating
    evaluation_x. Any transformation of the covariate belongs to the caller.

    Restricted cubic splines constrain the tails to be linear [1]_. Percentile
    knot placement follows the Hmisc rule [2]_, without its small-sample or
    tied-value adjustments. Patsy's centered natural cubic basis [3]_
    represents the same spline space.

    Parameters
    ----------
    x : ndarray, shape (n_samples,)
        Finite, nonconstant training covariate values.
    evaluation_x : ndarray, shape (n_evaluation,)
        Finite covariate values at which to evaluate the fitted basis.
    knots : int
        Total knot count, including the two boundary knots.

    Returns
    -------
    tuple[ndarray, ndarray]
        Training and evaluation designs, each with knots - 1
        columns. Both use the training knots, centering, and scaling.

    References
    ----------
    .. [1] Harrell, F. E. "Regression Modeling Strategies", course notes,
       Section 2.4.5, "Restricted Cubic Splines".
       https://hbiostat.org/rmsc/genreg
    .. [2] Harrell, F. E. Hmisc documentation, "rcspline.eval: Restricted
       Cubic Spline Design Matrix", knot-location specification.
       https://github.com/harrelfe/Hmisc/blob/master/man/rcspline.eval.Rd
    .. [3] Patsy documentation, "patsy.builtins.cr".
       https://patsy.readthedocs.io/en/latest/builtins-reference.html#patsy.builtins.cr
    """
    outer = 0.1 if knots == 3 else 0.05 if knots <= 6 else 0.025
    locations = np.quantile(x, np.linspace(outer, 1 - outer, knots))
    if np.unique(x).size < knots or np.unique(locations).size < knots:
        raise ValueError("Values do not support the requested distinct spline knots.")
    settings = {
        "interior_knots": locations[1:-1],
        "lower": locations[0],
        "upper": locations[-1],
    }
    design = dmatrix(
        "cr(x, knots=interior_knots, lower_bound=lower, upper_bound=upper, constraints='center') - 1",
        {"x": x, **settings},
    )
    evaluation = np.asarray(
        dmatrix(
            design.design_info,
            {"x": evaluation_x, **settings},
        )
    )
    center, scale = np.mean(design, axis=0), np.std(design, axis=0)
    if np.any(scale == 0):
        raise ValueError("Values produce a rank-deficient spline design.")
    design = (np.asarray(design) - center) / scale
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Values produce a rank-deficient spline design.")
    return design, (evaluation - center) / scale


def _cloglog(probabilities: np.ndarray) -> np.ndarray:
    """
    Apply the complementary log-log transform with finite boundaries.

    Parameters
    ----------
    probabilities : ndarray
        Event probabilities in [0, 1].

    Returns
    -------
    ndarray
        Float64 values of log(-log(1 - p)), preserving the input shape.
        Inputs are not modified; clipping to [1e-10, 1 - 1e-10] is performed
        after conversion to float64 to handle float32 boundaries safely.
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-10, 1 - 1e-10)
    return np.log(-np.log1p(-probabilities))
