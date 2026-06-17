from __future__ import annotations

from typing import Optional

import numpy as np


def concordance_time_dependent(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: Optional[np.ndarray] = None,
    train_event_indicators: Optional[np.ndarray] = None,
    method: str = "Antolini",
    ties: str = "Risk",
    tau: Optional[float] = None,
) -> tuple[float, float, float]:
    """
    Calculate the time-dependent concordance index between the predicted risk scores and the true survival times.

    Parameters
    ----------
    risk_scores: np.ndarray, shape = (n_samples, n_anchor_times)
        The predicted risk scores for each sample at each anchor time.
        The risk scores should be ordered such that higher scores indicate higher risk
        (i.e., lower survival probability or lower hazard score).
    event_times: np.ndarray, shape = (n_samples,)
        The true survival times.
    event_indicators: np.ndarray, shape = (n_samples,)
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    train_event_times: np.ndarray, shape = (n_train_samples,), optional
        The true survival times of the training set. Required for "IPCW".
    train_event_indicators: np.ndarray, shape = (n_train_samples,), optional
        Binary training-set event indicators: 1 denotes an observed event and
        0 denotes a censored observation. Required for "IPCW".
    method: str, optional (default="Antolini")
        A string indicating the method for constructing the pairs of samples.
        Options are "Antolini" (default), "Naive", or "IPCW".
        "Antolini": comparable pairs are anchored by samples with observed
        events. If sample i has an observed event at time t_i, it is compared
        with samples whose observed event/censoring time is greater than t_i.
        "Naive": alias of "Antolini".
        "IPCW": Antolini-style comparable pairs weighted by inverse probability
        of censoring weights from the training data.
    ties: str, optional (default="Risk")
        A string indicating the way ties should be handled.
        Options: "None", "Time", "Risk" (default), or "All"
    tau: float, optional
        The time horizon for the time-dependent concordance index. If None,
        the maximum observed time is used.

    Returns
    -------
    c_index: float
        The concordance index.
    num_concordant_pairs: float
        The number of concordant pairs.
    num_total_pairs: float
        The number of total pairs.
    """
    event_indicators = event_indicators.astype(bool)

    if risk_scores.ndim != 2:
        raise ValueError(
            f"risk_scores should be a 2D array of shape (n_samples, n_anchor_times), but got shape {risk_scores.shape}."
        )

    # check if the predicted risk scores has the dimension of (n_samples, n_anchor_times)
    n_samples, n_anchor_times = risk_scores.shape
    assert (
        n_samples == len(event_times) == len(event_indicators)
    ), "The lengths of the predicted times and labels must be the same."

    assert (
        n_anchor_times == event_indicators.sum()
    ), "The number of anchor times (columns in risk_scores) must match the number of observed events."

    method = method.lower()
    ties = ties.lower()

    if method == "antolini" or method == "naive":
        raise NotImplementedError("Antolini's method for time-dependent concordance index is not yet implemented.")
    elif method == "ipcw":
        if train_event_times is None or train_event_indicators is None:
            raise ValueError("train_event_times and train_event_indicators must be provided for IPCW method.")
        raise NotImplementedError("IPCW method for time-dependent concordance index is not yet implemented.")
    else:
        raise ValueError(f"Unsupported method: {method}. Supported methods are 'Antolini', 'Naive', and 'IPCW'.")
