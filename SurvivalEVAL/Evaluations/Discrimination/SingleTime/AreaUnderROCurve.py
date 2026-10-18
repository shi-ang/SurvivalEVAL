from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from SurvivalEVAL.Evaluations.util import (
    _check_control_definition,
    _inverse_censoring_weights,
    check_and_convert_event_data,
    check_and_convert_probabilities,
    validate_time_point,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier


def auc(
    predict_probs: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    target_time: float | None = None,
    train_event_times: np.ndarray | None = None,
    train_event_indicators: np.ndarray | None = None,
    method: str = "Naive",
) -> float:
    """Calculate cumulative/dynamic AUROC at a single target time.

    Cases are observed events at or before the target. Controls are observations
    after the target and censorings exactly at the target. Earlier censorings
    are excluded because their survival status is unknown.

    Args:
        predict_probs: One finite survival probability in [0, 1] per subject.
        event_times: Finite, non-negative observed event or censoring times.
        event_indicators: Binary indicators: 1 for events and 0 for censoring.
        target_time: Finite, non-negative evaluation time. Defaults to the median
            observed time of the test set.
        train_event_times: Training outcomes for the IPCW censoring model.
        train_event_indicators: Binary training event indicators, required along
            with training times when method is "IPCW" or "Uno".
        method: "Naive" (default), "IPCW", or "Uno", case-insensitive. IPCW assumes
            independent censoring and fits reverse Kaplan-Meier on the training
            outcomes. Cases receive weights 1/G(event_time-), using the left
            limit to respect events-first ordering.

    Returns:
        AUROC at the target time, with half credit for exactly tied predictions.
    """
    ipcw, include_competing = _check_auc_options(method)
    event_times, event_indicators = check_and_convert_event_data(
        event_times, event_indicators
    )
    predict_probs = check_and_convert_probabilities(predict_probs, "predict_probs")
    if predict_probs.shape != event_times.shape:
        raise ValueError("predict_probs must have one entry per observed subject.")
    target_time = _auc_target_time(target_time, event_times)
    censoring_survival = _auc_censoring_survival(
        event_times,
        target_time,
        train_event_times,
        train_event_indicators,
        ipcw,
    )
    score = _auc_at_time(
        1 - predict_probs,
        event_times,
        event_indicators,
        target_time,
        1,
        include_competing,
        censoring_survival,
    )
    if np.isnan(score):
        raise ValueError(
            f"Survival status is all zeros or all ones at time: {target_time}, AUC cannot be computed."
        )
    return score


def auc_cr_per_event(
    predict_probs: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    target_time: float | None = None,
    train_event_times: np.ndarray | None = None,
    train_event_indicators: np.ndarray | None = None,
    method: str = "Naive",
    event_of_interest: int = 1,
    control_definition: str = "NotCase",
) -> float:
    """Calculate cumulative/dynamic AUROC for one competing-risk cause.

    Cases have an observed event of the selected cause at or before the target.
    Earlier censorings are excluded. Censorings exactly at the target remain
    known event-free controls, matching the single-event AUC convention.

    Args:
        predict_probs: One finite CIF probability in [0, 1] per subject for the
            selected cause at the target. Higher values indicate greater risk.
        event_times: Finite, non-negative observed event or censoring times.
        event_indicators: Non-negative integer labels: 0 for censoring and
            positive labels for event causes.
        target_time: Finite, non-negative horizon. Defaults to the test median.
        train_event_times: Training outcomes, required for IPCW.
        train_event_indicators: Training cause labels. All positive labels are
            observed events when fitting the censoring distribution.
        method: "Naive" (default) or "IPCW", case-insensitive. IPCW assumes
            independent censoring and fits reverse Kaplan-Meier on training.
        event_of_interest: Positive integer cause label to evaluate.
        control_definition: "NotCase" (default) includes competing events by
            the target as controls; "EventFree" excludes them. Both include
            subjects observed after the target. Names are case-insensitive.

    Returns:
        AUROC, with half credit for exactly tied predictions.

    Raises:
        ValueError: If inputs are invalid, cases or controls are absent, or
            required IPCW censoring survival is nonfinite or nonpositive.

    References:
        Blanche, Dartigues and Jacqmin-Gadda (2013), Statistics in Medicine,
        https://doi.org/10.1002/sim.5958. Observed cases and competing-event
        controls receive 1/G(event_time-); event-free controls receive
        1/G(target_time-), respecting the package's boundary convention.
    """
    ipcw, include_competing = _check_auc_options(
        method, control_definition, competing_risks=True
    )
    event_times, event_indicators = check_and_convert_event_data(
        event_times, event_indicators, competing_risks=True
    )
    predict_probs = check_and_convert_probabilities(predict_probs, "predict_probs")
    if predict_probs.shape != event_times.shape:
        raise ValueError("predict_probs must have one entry per observed subject.")
    if (
        isinstance(event_of_interest, (bool, np.bool_))
        or not isinstance(event_of_interest, (int, np.integer))
        or event_of_interest < 1
    ):
        raise ValueError("event_of_interest must be a positive integer cause label.")
    target_time = _auc_target_time(target_time, event_times)
    censoring_survival = _auc_censoring_survival(
        event_times,
        target_time,
        train_event_times,
        train_event_indicators,
        ipcw,
        competing_risks=True,
    )
    score = _auc_at_time(
        predict_probs,
        event_times,
        event_indicators,
        target_time,
        event_of_interest,
        include_competing,
        censoring_survival,
    )
    if np.isnan(score):
        raise ValueError(
            f"Cases or controls are absent for cause {event_of_interest} at time: {target_time}, "
            "AUC cannot be computed."
        )
    return score


def auc_cr(
    predict_probs: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    target_time: float | None = None,
    train_event_times: np.ndarray | None = None,
    train_event_indicators: np.ndarray | None = None,
    method: str = "Naive",
    control_definition: str = "NotCase",
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """Calculate the equal-weight mean of competing-risk AUCs across causes.

    Args:
        predict_probs: Finite CIF probabilities in [0, 1], shape
            (n_samples, n_causes). Column k - 1 represents cause k at the target.
        event_times: Finite, non-negative observed event or censoring times.
        event_indicators: Integer labels from 0 through n_causes; 0 is censoring.
        target_time: Finite, non-negative horizon. Defaults to the test median.
        train_event_times: Training outcomes, required for IPCW.
        train_event_indicators: Non-negative integer training cause labels.
        method: "Naive" (default) or "IPCW", case-insensitive.
        control_definition: "NotCase" (default) or "EventFree", case-insensitive.
            See auc_cr_per_event for the control definitions and IPCW weights.
        return_details: Also return per_event_scores, keyed by cause label.

    Returns:
        Arithmetic mean of all predicted causes' AUCs, optionally with details.
        Causes without cases or controls have NaN scores and make the mean NaN.
        Undefined causes are included in details and are never silently omitted.

    Raises:
        ValueError: If inputs are invalid or required censoring survival is
            nonfinite or nonpositive. IPCW fits one censoring model for all causes.
    """
    ipcw, include_competing = _check_auc_options(
        method, control_definition, competing_risks=True
    )
    event_times, event_indicators = check_and_convert_event_data(
        event_times, event_indicators, competing_risks=True
    )
    predict_probs = check_and_convert_probabilities(predict_probs, "predict_probs")
    if predict_probs.ndim != 2 or predict_probs.shape[0] != event_times.size:
        raise ValueError("predict_probs must have shape (n_samples, n_causes).")
    if np.any(event_indicators > predict_probs.shape[1]):
        raise ValueError("Event labels must not exceed the number of predicted causes.")
    target_time = _auc_target_time(target_time, event_times)
    censoring_survival = _auc_censoring_survival(
        event_times,
        target_time,
        train_event_times,
        train_event_indicators,
        ipcw,
        competing_risks=True,
    )
    scores = {
        cause: _auc_at_time(
            probabilities,
            event_times,
            event_indicators,
            target_time,
            cause,
            include_competing,
            censoring_survival,
        )
        for cause, probabilities in enumerate(predict_probs.T, start=1)
    }
    score = float(np.mean(list(scores.values())))
    return (score, {"per_event_scores": scores}) if return_details else score


def _check_auc_options(
    method: str,
    control_definition: str = "EventFree",
    *,
    competing_risks: bool = False,
) -> tuple[bool, bool]:
    """Validate public options once and return IPCW and competing-control flags."""
    method = method.lower()
    methods = {"naive", "ipcw"} if competing_risks else {"naive", "ipcw", "uno"}
    if method not in methods:
        if competing_risks:
            raise ValueError("method must be 'Naive' or 'IPCW'.")
        raise ValueError("method must be 'Naive', 'IPCW', or 'Uno'.")
    return method != "naive", _check_control_definition(control_definition)


def _auc_target_time(target_time: float | None, event_times: np.ndarray) -> float:
    """Resolve the test-median default and validate a finite evaluation horizon."""
    if target_time is None:
        return float(np.median(event_times))
    target_time = validate_time_point(target_time)
    if not np.isfinite(target_time):
        raise ValueError("target_time must be finite.")
    return target_time


def _auc_censoring_survival(
    event_times: np.ndarray,
    target_time: float,
    train_event_times: np.ndarray | None,
    train_event_indicators: np.ndarray | None,
    ipcw: bool,
    *,
    competing_risks: bool = False,
) -> np.ndarray | None:
    """Fit once and return G(min(observed_time, target_time)-), or None for Naive."""
    if not ipcw:
        return None
    if train_event_times is None or train_event_indicators is None:
        raise ValueError(
            "Training event times and indicators are required for IPCW AUROC."
        )
    train_times, train_events = check_and_convert_event_data(
        train_event_times, train_event_indicators, competing_risks=competing_risks
    )
    return KaplanMeier(train_times, train_events > 0, reverse=True).predict(
        np.minimum(event_times, target_time), side="left"
    )


def _auc_at_time(
    predicted_risks: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    target_time: float,
    event_of_interest: int,
    include_competing: bool,
    censoring_survival: np.ndarray | None,
) -> float:
    """Score validated outcomes; return NaN when either comparison class is absent."""
    cases = (event_indicators == event_of_interest) & (event_times <= target_time)
    # A censoring exactly at the target is still known to be event-free there.
    controls = (event_times > target_time) | (
        (event_indicators == 0) & (event_times == target_time)
    )
    if include_competing:
        controls |= (
            (event_indicators > 0)
            & (event_indicators != event_of_interest)
            & (event_times <= target_time)
        )
    if not np.any(cases) or not np.any(controls):
        return float("nan")
    known = cases | controls
    sample_weight = (
        None
        if censoring_survival is None
        else _inverse_censoring_weights(censoring_survival[known])
    )

    return float(
        roc_auc_score(cases[known], predicted_risks[known], sample_weight=sample_weight)
    )
