from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from SurvivalEVAL.Evaluations.util import check_and_convert, validate_time_point

from ._concordance_utils import _check_survival_event_data


def auc(
    predict_probs: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    target_time: float | None = None,
) -> float:
    """
    Calculate the Area Under the Curve (AUC) for the survival model.

    Parameters
    ----------
    predict_probs: np.ndarray
        One finite survival probability in [0, 1] per subject.
    event_times: np.ndarray
        Finite, non-negative observed event or censoring times.
    event_indicators: np.ndarray
        The binary indicators of whether the event occurred (1) or was censored (0)
    target_time: float, optional
        The finite, non-negative time point at which to calculate the AUC. If not specified, the median of the event times is used.

    Returns
    -------
    auc: float
        The AUC value calculated at the specified target time.
    """
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
    binary_status = (event_indicators & (event_times <= target_time))[known]
    if np.all(binary_status) or not np.any(binary_status):
        raise ValueError(
            f"Survival status is all zeros or all ones at time: {target_time}, AUC cannot be computed."
        )
    return roc_auc_score(binary_status, 1 - predict_probs[known])
