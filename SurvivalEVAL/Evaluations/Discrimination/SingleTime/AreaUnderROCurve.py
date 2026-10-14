from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from SurvivalEVAL.Evaluations.util import check_and_convert, validate_time_point
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier

from .._concordance_utils import _check_survival_event_data


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
            with training times when method is "IPCW".
        method: "Naive" (default) or "IPCW", case-insensitive. IPCW assumes
            independent censoring and fits reverse Kaplan-Meier on the training
            outcomes. Cases receive weights 1/G(event_time-), using the left
            limit to respect events-first ordering. The common control weight
            1/G(target_time-) cancels, so controls receive unit weights.

    Returns:
        AUROC at the target time, with half credit for exactly tied predictions.

    Raises:
        ValueError: If inputs or the method are invalid, either class is absent,
            IPCW training outcomes are missing, the target exceeds training
            follow-up, or censoring survival is nonpositive.
    """
    method = method.lower()
    if method not in {"naive", "ipcw"}:
        raise ValueError("method must be 'Naive' or 'IPCW'.")

    predict_probs = check_and_convert(predict_probs)
    event_times, event_indicators = _check_survival_event_data(
        event_times, event_indicators
    )
    if predict_probs.ndim != 1 or predict_probs.shape != event_times.shape:
        raise ValueError("predict_probs must have one entry per observed subject.")
    if not np.all(np.isfinite(predict_probs)) or np.any(
        (predict_probs < 0) | (predict_probs > 1)
    ):
        raise ValueError("predict_probs must contain finite probabilities in [0, 1].")
    if target_time is None:
        target_time = float(np.median(event_times))
    else:
        target_time = validate_time_point(target_time)
        if not np.isfinite(target_time):
            raise ValueError("target_time must be finite.")

    # Censorings at the target remain event-free controls, matching the
    # package's boundary convention. Earlier censorings have unknown status.
    known = event_indicators | (event_times >= target_time)
    cases = event_indicators & (event_times <= target_time)
    binary_status = cases[known]
    if np.all(binary_status) or not np.any(binary_status):
        raise ValueError(
            f"Survival status is all zeros or all ones at time: {target_time}, AUC cannot be computed."
        )
    sample_weight = None
    if method == "ipcw":
        if train_event_times is None or train_event_indicators is None:
            raise ValueError(
                "Training event times and indicators are required for IPCW AUROC."
            )
        train_times, train_events = _check_survival_event_data(
            train_event_times, train_event_indicators
        )
        if target_time > train_times.max():
            raise ValueError(
                "target_time must not exceed the maximum training follow-up time for IPCW AUROC."
            )

        censoring_model = KaplanMeier(train_times, train_events, reverse=True)
        # Query only case times and the target, before each time's censoring update.
        # The prepended baseline handles queries before the first update, including zero.
        indices = np.searchsorted(
            censoring_model.survival_times,
            np.r_[event_times[cases], target_time],
            side="left",
        )
        censoring_survival = np.r_[1.0, censoring_model.survival_probabilities][indices]
        if not np.all(censoring_survival > 0):
            raise ValueError(
                "Censoring survival is nonpositive; choose a smaller target_time for IPCW AUROC."
            )
        sample_weight = np.ones(binary_status.shape, dtype=float)
        sample_weight[binary_status] = 1 / censoring_survival[:-1]

    return float(
        roc_auc_score(
            binary_status, 1 - predict_probs[known], sample_weight=sample_weight
        )
    )
