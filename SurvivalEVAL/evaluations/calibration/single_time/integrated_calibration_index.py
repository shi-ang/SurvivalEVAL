"""Integrated calibration indices and calibration curves at a specified time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from SurvivalEVAL.evaluations.calibration.single_time.util import (
    _cloglog,
    _restricted_cubic_spline,
)
from SurvivalEVAL.evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.evaluations.util import (
    _inverse_censoring_weights,
    check_and_convert_event_data,
    check_and_convert_probabilities,
    validate_time_point,
)
from SurvivalEVAL.nonparametric_estimator.competing_risks import (
    AalenJohansenCompetingRisks,
)
from SurvivalEVAL.nonparametric_estimator.single_event import KaplanMeier
from SurvivalEVAL.nonparametric_estimator.util import _predict_step


def integrated_calibration_index(
    preds: NumericArrayLike,
    event_time: NumericArrayLike,
    event_indicator: NumericArrayLike,
    target_time: Numeric,
    knots: int = 3,
    draw_figure: bool = False,
    figure_range: tuple | None = None,
) -> dict | tuple[dict, tuple[plt.Figure, plt.Axes]]:
    """
    Estimate a survival model's calibration curve and ICI at one time.

    Following Appendix A of [1]_, regress the event hazard on a restricted
    cubic spline of the complementary log-log transformed predictions using
    a Cox model. Errors equally weight all original subjects.

    Parameters
    ----------
    preds : NumericArrayLike, shape (n_samples,)
        Predicted event probabilities in [0, 1] at target_time.
    event_time : NumericArrayLike, shape (n_samples,)
        Finite, non-negative observed event or censoring times.
    event_indicator : NumericArrayLike, shape (n_samples,)
        Binary indicators: 1 for observed events and 0 for censoring.
    target_time : Numeric
        Finite, non-negative horizon. Events at this time are included.
    knots : int, default=3
        Total number of restricted cubic spline knots, at least three.
        Knot percentiles are equally spaced between 10 and 90 for three
        knots, 5 and 95 for four through six, and 2.5 and 97.5 thereafter.
        Nonconstant predictions must support distinct knots and a full-rank
        spline design. Constant predictions use the Kaplan-Meier estimate.
    draw_figure : bool, default=False
        Return a calibration plot without displaying it.
    figure_range : tuple, optional
        Plot limits (x_min, x_max, y_min, y_max).

    Returns
    -------
    summary : dict
        ICI, E50, E90, and E_max summarize individual absolute errors.
        The curve entry contains grid and cal_pred arrays of length 100,
        spanning the first through 99th percentiles of predicted risk.
    fig : tuple[plt.Figure, plt.Axes]
        Figure and axes, returned with summary only if draw_figure is True.

    Raises
    ------
    ValueError
        If inputs are invalid, the spline is unsupported, or the calibration
        model is not identifiable or fails to converge.

    Notes
    -----
    Fits use the full observed follow-up. Baseline hazards use right-continuous
    steps, remaining constant after the final event. No events by target_time
    gives a zero calibrated curve. Efron corrections handle tied events.

    References
    ----------
    .. [1] Austin et al. (2020). "Graphical calibration curves and the integrated
       calibration index (ICI) for survival models". Statistics in Medicine,
       39(21), 2714-2742. https://doi.org/10.1002/sim.8570
    """
    event_time, event_indicator = check_and_convert_event_data(
        event_time, event_indicator
    )
    return _integrated_calibration_index(
        preds,
        event_time,
        event_indicator,
        target_time,
        knots,
        draw_figure,
        figure_range,
    )


def integrated_calibration_index_cr(
    preds: NumericArrayLike,
    event_time: NumericArrayLike,
    event_indicator: NumericArrayLike,
    target_time: Numeric,
    event_of_interest: int = 1,
    knots: int = 3,
    draw_figure: bool = False,
    figure_range: tuple | None = None,
) -> dict | tuple[dict, tuple[plt.Figure, plt.Axes]]:
    """
    Estimate a selected cause's calibration curve and ICI using Fine-Gray.

    Following [1]_, regress the subdistribution hazard on a restricted cubic
    spline of the complementary log-log transformed predicted CIF. Competing
    events remain in weighted risk sets; they are not treated as censorings.
    Censoring weights are estimated from the evaluation outcomes.

    Parameters
    ----------
    preds : NumericArrayLike, shape (n_samples,)
        Predicted selected-cause cumulative incidence in [0, 1] at target_time.
    event_time : NumericArrayLike, shape (n_samples,)
        Finite, non-negative observed event or censoring times.
    event_indicator : NumericArrayLike, shape (n_samples,)
        Non-negative integer cause labels; 0 denotes censoring.
    target_time : Numeric
        Finite, non-negative horizon. Events at this time are included.
    event_of_interest : int, default=1
        Positive cause label, including an unobserved cause.
    knots : int, default=3
        Total spline knot count, at least three. Placement and support
        requirements match integrated_calibration_index. Constant predictions
        use the selected-cause Aalen-Johansen incidence estimate.
    draw_figure : bool, default=False
        Return a calibration plot without displaying it.
    figure_range : tuple, optional
        Plot limits (x_min, x_max, y_min, y_max).

    Returns
    -------
    summary : dict
        ICI, E50, E90, E_max, and curve with grid and cal_pred arrays, matching
        integrated_calibration_index. Errors equally weight original subjects.
    fig : tuple[plt.Figure, plt.Axes]
        Figure and axes, returned with summary only if draw_figure is True.

    Raises
    ------
    ValueError
        If inputs are invalid, the spline is unsupported, or the calibration
        model is not identifiable or fails to converge.

    Notes
    -----
    Assumes independent censoring. Fits use the full observed follow-up;
    predictions include events at target_time and stay constant after the
    final selected-cause event. No selected-cause events by target_time gives
    a zero calibrated curve. Efron corrections handle tied events.
    Confidence intervals are not estimated.

    References
    ----------
    .. [1] Austin, Putter, Giardiello and van Klaveren (2022). "Graphical
       calibration curves and the integrated calibration index (ICI) for
       competing risk models". Diagnostic and Prognostic Research, 6, 2.
       https://doi.org/10.1186/s41512-021-00114-6
    """
    event_time, event_indicator = check_and_convert_event_data(
        event_time, event_indicator, competing_risks=True
    )
    if (
        isinstance(event_of_interest, (bool, np.bool_))
        or not isinstance(event_of_interest, (int, np.integer))
        or event_of_interest < 1
    ):
        raise ValueError("event_of_interest must be a positive integer cause label.")
    events = np.where(event_indicator > 0, 2, 0)
    events[event_indicator == event_of_interest] = 1
    return _integrated_calibration_index(
        preds, event_time, events, target_time, knots, draw_figure, figure_range
    )


def _integrated_calibration_index(
    preds: NumericArrayLike,
    event_time: np.ndarray,
    event_indicator: np.ndarray,
    target_time: Numeric,
    knots: int,
    draw_figure: bool,
    figure_range: tuple | None,
) -> dict | tuple[dict, tuple[plt.Figure, plt.Axes]]:
    """
    Fit and summarize calibration with labels 0=censor, 1=target, 2=competitor.

    Parameters
    ----------
    preds : NumericArrayLike, shape (n_samples,)
        Original predicted event probabilities.
    event_time, event_indicator : ndarray, shape (n_samples,)
        Validated observed times and pooled event labels.
    target_time : Numeric
        Evaluation horizon.
    knots : int
        Total spline knot count.
    draw_figure : bool
        Whether to return a plot.
    figure_range : tuple or None
        Optional x and y axis limits.

    Returns
    -------
    dict or tuple[dict, tuple[plt.Figure, plt.Axes]]
        Calibration metrics and curve, optionally with figure and axes.
    """
    preds = check_and_convert_probabilities(preds, "preds")
    if preds.shape != event_time.shape:
        raise ValueError(
            "preds, event_time, and event_indicator must have the same shape."
        )
    target_time = validate_time_point(target_time)
    if not np.isfinite(target_time):
        raise ValueError("target_time must be finite.")
    if (
        isinstance(knots, (bool, np.bool_))
        or not isinstance(knots, (int, np.integer))
        or knots < 3
    ):
        raise ValueError("knots must be an integer of at least 3.")

    grid = np.linspace(*np.quantile(preds, [0.01, 0.99]), 100)
    if not np.any((event_indicator == 1) & (event_time <= target_time)):
        calibrated = np.zeros(preds.size + grid.size)
    elif np.unique(preds).size == 1:
        if np.any(event_indicator == 2):
            observed = (
                AalenJohansenCompetingRisks(n_causes=2)
                .fit(event_time, event_indicator)
                .predict_cif(target_time)[0]
            )
        else:
            km = KaplanMeier(event_time, event_indicator == 1)
            observed = (
                1
                - _predict_step(
                    target_time, km.survival_times, km.survival_probabilities, 1.0
                ).item()
            )
        calibrated = np.full(preds.size + grid.size, observed)
    else:
        transformed = _cloglog(preds)
        evaluation_x = np.concatenate((transformed, _cloglog(grid)))
        spline, evaluation_spline = _restricted_cubic_spline(
            transformed, evaluation_x, knots
        )
        calibrated = _fit_calibration_curve(
            spline, evaluation_spline, event_time, event_indicator, target_time
        )

    abs_err = np.abs(preds - calibrated[: preds.size])
    cal_pred = calibrated[preds.size :]
    summary = {
        "ICI": np.mean(abs_err, dtype=float),
        "E50": np.median(abs_err),
        "E90": np.quantile(abs_err, 0.9),
        "E_max": np.max(abs_err),
        "curve": {"grid": grid, "cal_pred": cal_pred},
    }
    if not draw_figure:
        return summary

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(grid, cal_pred, label="Calibration Curve", color="blue")
    ax.plot(grid, grid, label="Perfect Calibration", linestyle="--", color="grey")
    ax.set_xlabel("Predicted Event Probability")
    ax.set_ylabel("Observed Event Probability")
    ax.set_title("Graphical Calibration Curve")
    ax.legend()
    if figure_range is not None:
        ax.set_xlim(figure_range[0], figure_range[1])
        ax.set_ylim(figure_range[2], figure_range[3])
    return summary, (fig, ax)


def _fit_calibration_curve(
    spline: np.ndarray,
    evaluation_spline: np.ndarray,
    event_time: np.ndarray,
    event_indicator: np.ndarray,
    target_time: float,
) -> np.ndarray:
    """
    Fit Cox/Fine-Gray point estimates and predict calibrated probabilities.

    Parameters
    ----------
    spline : ndarray, shape (n_samples, n_terms)
        Standardized spline design for original subjects.
    evaluation_spline : ndarray, shape (n_predictions, n_terms)
        Design evaluated using the same knots, centering, and scaling.
    event_time, event_indicator : ndarray, shape (n_samples,)
        Observed times and labels 0=censor, 1=target, 2=competitor.
    target_time : float
        Finite evaluation time with at least one selected event by that time.

    Returns
    -------
    ndarray, shape (n_predictions,)
        Fitted cumulative incidence at target_time. Efron corrections are
        used for both partial likelihood and cumulative baseline hazard.

    Notes
    -----
    Fine-Gray regression is fitted directly through its weighted partial
    likelihood. A subject with a competing event at T_i < t remains in the
    risk set with weight G(t-) / G(T_i-), where G is censoring survival.
    Subjects still under observation at t have unit weight. With no competing
    events, this is the ordinary Cox partial likelihood.

    Sorted-time cumulative sums compute these weighted risk sets without
    expanding subjects into repeated rows, as in the R crprep construction.
    """
    order = np.argsort(event_time, kind="stable")
    times, events, x = event_time[order], event_indicator[order], spline[order]
    unique_times, starts = np.unique(times, return_index=True)
    deaths = events == 1
    counts = np.add.reduceat(deaths.astype(int), starts)
    event_groups = counts > 0
    counts = counts[event_groups]
    event_starts = starts[event_groups]
    # One row per Efron denominator, including each member of a tied death set.
    groups = np.repeat(np.arange(counts.size), counts)
    fractions = (
        np.arange(deaths.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    ) / counts[groups]

    # Fine-Gray retains earlier competing events with weight G(t-) / G(T_i-).
    # Reverse KM removes all event causes before updating tied censorings;
    # left limits give the events-first convention used by R's Fine-Gray fit.
    competitors = events == 2
    competitor_weights = np.zeros(times.size)
    censoring_at_events = np.ones(counts.size)
    if np.any(competitors):
        censoring = KaplanMeier(times, events > 0, reverse=True)
        g_left = censoring.predict(times, side="left")
        competitor_weights[competitors] = _inverse_censoring_weights(
            g_left[competitors]
        )
        censoring_at_events = g_left[event_starts]

    n_terms = x.shape[1]
    # Include second moments to check identifiability at the fitted solution.
    values = np.column_stack(
        (
            np.ones(times.size),
            x,
            (x[:, :, None] * x[:, None, :]).reshape(times.size, -1),
        )
    )
    death_x = x[deaths].sum(axis=0)

    def loss_and_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Evaluate mean negative partial likelihood and its analytic gradient.

        Parameters
        ----------
        beta : ndarray, shape (n_terms,)
            Spline coefficients.

        Returns
        -------
        tuple[float, ndarray]
            Loss and gradient, normalized by the number of target events.
        """
        eta, shift, moments = risk_moments(beta)
        denominator = moments[:, 0]
        if np.any(denominator <= 0):
            return np.inf, np.zeros(n_terms)
        loss = (np.log(denominator).sum() - (eta[deaths] - shift).sum()) / deaths.sum()
        gradient = (
            (moments[:, 1 : n_terms + 1] / denominator[:, None]).sum(axis=0) - death_x
        ) / deaths.sum()
        return float(loss), gradient

    def risk_moments(beta: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Sum weighted risk moments with Efron adjustments and no data expansion.

        Parameters
        ----------
        beta : ndarray, shape (n_terms,)
            Spline coefficients.

        Returns
        -------
        tuple[ndarray, float, ndarray]
            Linear predictors, their maximum, and shifted zeroth, first, and
            second risk moments for each selected-event Efron denominator.
        """
        eta = x @ beta
        shift = float(np.max(eta))
        weighted = np.exp(eta - shift)[:, None] * values
        # Subjects with T_i >= t contribute unit-weight Cox risk moments.
        remaining = np.cumsum(weighted[::-1], axis=0)[::-1][event_starts]
        # Earlier competing events contribute moments divided by G(T_i-).
        # Stop before the entire time group so tied events remain in remaining.
        prior = np.vstack(
            (
                np.zeros(values.shape[1]),
                np.cumsum(weighted * competitor_weights[:, None], axis=0),
            )
        )[event_starts]
        # Multiplication by G(t-) completes the Fine-Gray risk weights.
        risk = remaining + censoring_at_events[:, None] * prior
        tied = np.add.reduceat(weighted * deaths[:, None], starts, axis=0)[event_groups]
        # Efron removes successive fractions of the tied target-event moments.
        return eta, shift, risk[groups] - fractions[:, None] * tied[groups]

    # Estimate spline coefficients from the Cox/Fine-Gray partial likelihood.
    fit = minimize(
        loss_and_gradient,
        np.zeros(n_terms),
        method="BFGS",
        jac=True,
        options={"gtol": 1e-7, "maxiter": 200},
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"Calibration model failed to converge: {fit.message}")
    _, shift, moments = risk_moments(fit.x)
    denominator = moments[:, 0]
    means = moments[:, 1 : n_terms + 1] / denominator[:, None]
    second = (
        moments[:, n_terms + 1 :].reshape(-1, n_terms, n_terms)
        / denominator[:, None, None]
    )
    information = np.mean(second - means[:, :, None] * means[:, None, :], axis=0)
    if np.linalg.eigvalsh(information).min() <= 1e-10:
        raise ValueError(
            "Calibration model is not identifiable from the observed risk sets."
        )

    # Each Efron denominator contributes its reciprocal to H0(target_time).
    # Fine-Gray incidence is 1 - exp(-H0(t) * exp(x @ beta)); the exponential
    # shift used in the risk sums cancels when evaluating this expression.
    before_target = unique_times[event_groups][groups] <= target_time
    log_hazard = logsumexp(-np.log(denominator[before_target]))
    with np.errstate(over="ignore"):
        return -np.expm1(-np.exp(log_hazard + evaluation_spline @ fit.x - shift))
