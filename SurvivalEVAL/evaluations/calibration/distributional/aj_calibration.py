"""Calibration of mean predicted CIFs against Aalen–Johansen estimates."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from SurvivalEVAL.evaluations.util import (
    check_and_convert,
    check_and_convert_event_data,
    check_and_convert_probabilities,
    zero_padding,
)
from SurvivalEVAL.nonparametric_estimator.competing_risks import (
    AalenJohansenCompetingRisks,
)


def _aj_calibration_scores(
    pred_cifs: np.ndarray,
    time_coordinates: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    alpha: float = 2,
    min_prop_at_risk: float = 0.05,
    event_of_interest: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate inputs and compute scores with a single Aalen–Johansen fit.

    Parameters
    ----------
    pred_cifs : array-like
        Shape (n_samples, n_causes, n_time_points), or (n_samples,
        n_time_points) when selecting one cause with ``event_of_interest``.
    time_coordinates : array-like, shape (n_time_points,)
        Non-negative, strictly increasing prediction times.
    event_times, event_indicators : array-like, shape (n_samples,)
        Observed times and integer labels, with 0 denoting censoring.
    alpha : float, default=2
        Finite exponent at least 1 for the powered calibration error.
    min_prop_at_risk : float, default=0.05
        Minimum fraction of subjects still at risk at retained grid points.
    event_of_interest : int, optional
        Positive cause label for a single-cause prediction matrix. Otherwise,
        evaluate all predicted causes, including unobserved ones.

    Returns
    -------
    scores : ndarray, shape (n_evaluated_causes,)
        Time-normalized integrated powered errors.
    times : ndarray, shape (n_retained_times,)
        Retained grid, with zero prepended if the input starts after zero.
    mean_pred_cifs, aj_cifs : ndarray, shape (n_evaluated_causes, n_retained_times)
        Mean predictions and right-continuous Aalen–Johansen reference CIFs.
    """
    cifs = check_and_convert_probabilities(pred_cifs, "CIFs")
    times = check_and_convert(time_coordinates)
    observed_times, events = check_and_convert_event_data(
        event_times, event_indicators, competing_risks=True
    )
    if event_of_interest is not None:
        if cifs.ndim != 2:
            raise ValueError(
                "Per-event CIFs must have shape (n_samples, n_time_points)."
            )
        cifs = cifs[:, np.newaxis, :]
    elif cifs.ndim != 3:
        raise ValueError(
            "pred_cifs must have shape (n_samples, n_causes, n_time_points)."
        )
    if times.ndim != 1 or times.size < 2:
        raise ValueError(
            "time_coordinates must be a 1D array with at least two points."
        )
    if cifs.shape[2] != times.size:
        raise ValueError("The CIF time axis must match time_coordinates.")
    if cifs.shape[0] != observed_times.size:
        raise ValueError("Event arrays must have one entry per predicted sample.")
    if (
        not np.all(np.isfinite(times))
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "time_coordinates must be finite, non-negative and strictly increasing."
        )
    if np.any(np.diff(cifs, axis=2) < 0):
        raise ValueError("CIF curves must be nondecreasing.")
    if np.any(cifs.sum(axis=1, dtype=float) > 1 + 1e-6):
        raise ValueError("CIFs across causes must sum to at most one.")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or alpha < 1:
        raise ValueError("alpha must be finite and at least 1.")
    if (
        not np.isscalar(min_prop_at_risk)
        or not np.isfinite(min_prop_at_risk)
        or not 0 <= min_prop_at_risk <= 1
    ):
        raise ValueError("min_prop_at_risk must be finite and between 0 and 1.")

    n_causes = cifs.shape[1]
    if event_of_interest is not None:
        # Pool competing causes without changing the selected marginal CIF.
        events = np.where(events == event_of_interest, 1, np.where(events > 0, 2, 0))
        n_causes = 2
    aj = AalenJohansenCompetingRisks(n_causes=n_causes).fit(observed_times, events)
    mean_cifs, times = zero_padding(
        cifs.mean(axis=0, dtype=float), times, initial_value=0.0
    )

    # Count T >= t, including event/censoring ties, without an n_samples x n_times array.
    at_risk = observed_times.size - np.searchsorted(np.sort(observed_times), times)
    keep = at_risk / observed_times.size >= min_prop_at_risk
    times, mean_cifs = times[keep], mean_cifs[:, keep]
    if times.size < 2:
        raise ValueError(
            "At least two time points must remain after at-risk truncation."
        )
    aj_cifs = aj.predict_cif(times).T
    if event_of_interest is not None:
        aj_cifs = aj_cifs[:1]

    # Integrating on the normalized grid gives the time-averaged powered error.
    scores = trapezoid(np.abs(mean_cifs - aj_cifs) ** alpha, times / times[-1], axis=1)
    return scores, times, mean_cifs, aj_cifs


def aj_calibration_per_event(
    pred_cifs: np.ndarray,
    time_coordinates: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    event_of_interest: int,
    alpha: float = 2,
    min_prop_at_risk: float = 0.05,
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """
    Measure marginal AJ-calibration for one competing cause (lower is better).

    Fits Aalen–Johansen on the complete calibration set, assuming independent
    censoring for marginal estimation. Approximates the integral over [0,
    final retained time] with the trapezoidal rule on the prediction grid.
    This is a marginal calibration error, not a hypothesis-test p-value.

    Uses the hazardous implementation's time-normalized powered error, in
    keeping with KM-calibration. The paper instead defines an unnormalized
    L_alpha norm, with an alpha-th root after integration; scalar scores differ.
    Unlike hazardous, predictions have no survival channel and a missing
    time-zero point is supplied explicitly.

    Parameters
    ----------
    pred_cifs : array-like, shape (n_samples, n_time_points)
        Nondecreasing predicted CIFs in [0, 1] for the selected cause.
    time_coordinates : array-like, shape (n_time_points,)
        Finite, non-negative, strictly increasing shared grid with at least
        two points. If it starts after zero, prepend time zero with zero CIFs.
    event_times : array-like, shape (n_samples,)
        Finite, non-negative event/censoring times of the calibration set.
    event_indicators : array-like, shape (n_samples,)
        Integer labels: 0 for censoring and positive integers for events.
        Other competing causes remain observed events, not censorings.
    event_of_interest : int
        Positive cause label to evaluate, even if absent from observed labels.
    alpha : float, default=2
        Finite exponent at least 1. The default gives time-averaged squared
        error; alpha=1 gives time-averaged absolute error.
    min_prop_at_risk : float, default=0.05
        Retain grid points where at least this fraction of subjects has an
        observed time greater than or equal to that point. Set to 0 to use
        the full prediction grid. At least two retained points are required.
    return_details : bool, default=False
        Also return ``time_coordinates``, ``mean_pred_cifs`` and ``aj_cifs``
        as 1D arrays on the retained grid.

    Returns
    -------
    score : float
        Integrated powered difference between the mean predicted CIF and the
        Aalen–Johansen CIF, divided by the final retained time. Zero denotes
        agreement on the grid; scores lie in [0, 1].
    details : dict, optional
        Time grid and the two marginal CIF curves.

    Notes
    -----
    The method is described in [1]_.
    A related implementation is available in [2]_.

    References
    ----------
    .. [1] Alberge, J.; Haugomat, T.; Varoquaux, G.; Abécassis, J. (2026).
       "On the Calibration of Survival Models with Competing Risks".
       Proceedings of the 29th International Conference on Artificial Intelligence
       and Statistics, Proceedings of Machine Learning Research, 300, 1837-1845.
       https://proceedings.mlr.press/v300/alberge26a.html
    .. [2] Hazardous reference implementation.
       https://github.com/soda-inria/hazardous/blob/main/hazardous/metrics/_aj_calibration.py
    """
    if (
        isinstance(event_of_interest, (bool, np.bool_))
        or not isinstance(event_of_interest, (int, np.integer))
        or event_of_interest < 1
    ):
        raise ValueError("event_of_interest must be a positive integer cause label.")
    scores, times, mean_cifs, aj_cifs = _aj_calibration_scores(
        pred_cifs,
        time_coordinates,
        event_times,
        event_indicators,
        alpha=alpha,
        min_prop_at_risk=min_prop_at_risk,
        event_of_interest=event_of_interest,
    )
    score = float(scores[0])
    if not return_details:
        return score
    return score, {
        "time_coordinates": times,
        "mean_pred_cifs": mean_cifs[0],
        "aj_cifs": aj_cifs[0],
    }


def aj_calibration(
    pred_cifs: np.ndarray,
    time_coordinates: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    alpha: float = 2,
    reduction: str = "mean",
    min_prop_at_risk: float = 0.05,
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """
    Aggregate AJ-calibration over all predicted competing causes.

    All predicted causes are evaluated, including those absent from observed
    labels. Censoring is not scored as a separate outcome. The hazardous
    implementation also scores survival as event 0 and only includes observed
    causes, so its aggregate can differ even with the same prediction grid.

    The marginal calibration framework is described in [1]_. For the normalization
    and integration conventions used here, see ``aj_calibration_per_event``.

    Parameters
    ----------
    pred_cifs : array-like, shape (n_samples, n_causes, n_time_points)
        Nondecreasing CIF curves in [0, 1]. Index k - 1 on the cause axis
        represents cause k. CIFs must sum to at most one at each time point.
    time_coordinates : array-like, shape (n_time_points,)
        Shared prediction grid; see ``aj_calibration_per_event``.
    event_times : array-like, shape (n_samples,)
        Observed event/censoring times of the calibration set.
    event_indicators : array-like, shape (n_samples,)
        Integer labels from 0 to n_causes; 0 denotes censoring.
    alpha : float, default=2
        Finite exponent at least 1 for the time-averaged powered error.
    reduction : {"mean", "sum", "max"}, default="mean"
        Reduction of per-cause scores. Mean is in [0, 1], independent of the
        number of causes; sum and max are also available.
    min_prop_at_risk : float, default=0.05
        Minimum fraction at risk; see ``aj_calibration_per_event``.
    return_details : bool, default=False
        Also return ``per_event_scores`` keyed by cause label, retained
        ``time_coordinates``, and ``mean_pred_cifs`` and ``aj_cifs`` arrays
        of shape (n_causes, n_retained_times).

    Returns
    -------
    score : float
        Aggregated marginal calibration error. Lower is better.
    details : dict, optional
        Per-cause scores and marginal curves from a single AJ fit.

    References
    ----------
    .. [1] Alberge, J.; Haugomat, T.; Varoquaux, G.; Abécassis, J. (2026).
       "On the Calibration of Survival Models with Competing Risks".
       Proceedings of the 29th International Conference on Artificial Intelligence
       and Statistics, Proceedings of Machine Learning Research, 300, 1837-1845.
       https://proceedings.mlr.press/v300/alberge26a.html
    """
    reduction = reduction.lower()
    if reduction not in ("mean", "sum", "max"):
        raise ValueError("reduction must be 'mean', 'sum' or 'max'.")
    scores, times, mean_cifs, aj_cifs = _aj_calibration_scores(
        pred_cifs,
        time_coordinates,
        event_times,
        event_indicators,
        alpha=alpha,
        min_prop_at_risk=min_prop_at_risk,
    )
    score = float({"mean": np.mean, "sum": np.sum, "max": np.max}[reduction](scores))
    if not return_details:
        return score
    return score, {
        "per_event_scores": {k + 1: float(value) for k, value in enumerate(scores)},
        "time_coordinates": times,
        "mean_pred_cifs": mean_cifs,
        "aj_cifs": aj_cifs,
    }
