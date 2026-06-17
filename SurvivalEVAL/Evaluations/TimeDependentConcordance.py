from __future__ import annotations

from typing import Optional

import numpy as np

from SurvivalEVAL.Evaluations._concordance_utils import (
    _check_has_any_pairs,
    _ConcordanceCounts,
    _count_directed_risk_pairs,
    _finalize_counts,
    _is_before_tau,
    _iter_time_blocks,
    _same_time_pair_weight,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier


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
        (i.e., lower survival probability or higher hazard score).
        ``risk_scores[i, k]`` is the risk score for test sample ``i``
        evaluated at the kth observed-event anchor time. The kth anchor time
        corresponds to ``event_times[np.flatnonzero(event_indicators)[k]]``.
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
        Truncation time. If provided, only event anchors whose observed time is
        strictly before ``tau`` are counted. If None, no truncation is applied.

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
    if event_times.ndim != 1 or event_indicators.ndim != 1:
        raise ValueError("event_times and event_indicators must be 1-D arrays.")

    # check if the predicted risk scores has the dimension of (n_samples, n_anchor_times)
    n_samples, n_anchor_times = risk_scores.shape
    if not (n_samples == event_times.shape[0] == event_indicators.shape[0]):
        raise ValueError(
            "The lengths of risk_scores, event_times, and event_indicators must be the same."
        )

    n_observed_events = int(event_indicators.sum())
    if n_observed_events == 0:
        raise ValueError(
            "Data has no observed events, cannot estimate time-dependent concordance index."
        )
    if n_anchor_times != n_observed_events:
        raise ValueError(
            "The number of anchor times (columns in risk_scores) must match the number of observed events."
        )

    method = method.lower()
    ties = ties.lower()

    if method == "antolini" or method == "naive":
        sample_weights = None
        anchor_pair_weights = None
    elif method == "ipcw":
        if train_event_times is None or train_event_indicators is None:
            raise ValueError(
                "train_event_times and train_event_indicators must be provided for IPCW method."
            )
        train_event_indicators = train_event_indicators.astype(bool)

        censoring_model = KaplanMeier(train_event_times, ~train_event_indicators)
        censoring_survival = censoring_model.predict(event_times)
        observed_anchors = event_indicators & _is_before_tau(event_times, tau)

        # IPCW only needs positive censoring survival for event anchors whose
        # weights can affect the selected concordance result. Every non-final
        # event-time block has later samples as candidates. In the final block,
        # event anchors still contribute through same-time censored candidates;
        # event-event time ties contribute only when the requested tie policy
        # keeps time ties. Otherwise final events have no effect on the returned
        # index, so exclude them from the zero-survival check and weight
        # assignment.
        final_time = np.max(event_times)
        final_block = event_times == final_time
        final_events = final_block & event_indicators
        final_event_count = np.count_nonzero(final_events)
        final_has_censored_candidate = np.any(final_block & ~event_indicators)
        final_time_ties_counted = ties in {"time", "all"}
        final_events_contribute = final_has_censored_candidate or (
            final_time_ties_counted and final_event_count > 1
        )
        if not final_events_contribute:
            observed_anchors[final_events] = False

        if np.any(censoring_survival[observed_anchors] <= 0):
            raise ValueError(
                "Censoring survival probability is zero for at least one observed event; "
                "choose a smaller tau."
            )

        sample_weights = np.zeros_like(event_times, dtype=float)
        sample_weights[observed_anchors] = 1 / censoring_survival[observed_anchors]

        anchor_pair_weights = np.zeros_like(event_times, dtype=float)
        anchor_pair_weights[observed_anchors] = 1 / np.square(
            censoring_survival[observed_anchors]
        )
    else:
        raise ValueError(
            f"Unsupported method: {method}. Supported methods are 'Antolini', 'Naive', and 'IPCW'."
        )

    counts = _time_dependent_risk_counts(
        risk_scores=risk_scores,
        event_times=event_times,
        event_indicators=event_indicators,
        sample_weights=sample_weights,
        anchor_pair_weights=anchor_pair_weights,
        tau=tau,
    )

    _check_has_any_pairs(counts)
    return _finalize_counts(counts, ties)


def _time_dependent_risk_counts(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    anchor_pair_weights: Optional[np.ndarray] = None,
    tau: Optional[float] = None,
    tied_tol: float = 1e-8,
) -> _ConcordanceCounts:
    """Count Antolini-style time-dependent concordance pairs.

    Parameters
    ----------
    risk_scores: np.ndarray, shape = (n_samples, n_observed_events)
        Risk scores evaluated at observed-event anchor times. Higher scores
        indicate higher event risk.
    event_times: np.ndarray, shape = (n_samples,)
        Observed event or censoring times.
    event_indicators: np.ndarray, shape = (n_samples,)
        Boolean event indicators, where True denotes an observed event.
    sample_weights: np.ndarray, shape = (n_samples,), optional
        Optional symmetric sample weights used for same-time event ties and,
        unless ``anchor_pair_weights`` is provided, comparable pair weights.
    anchor_pair_weights: np.ndarray, shape = (n_samples,), optional
        Optional per-anchor pair weights. If provided, every comparable pair
        anchored by sample ``i`` receives ``anchor_pair_weights[i]``.
    tau: float, optional (default=None)
        Truncation time. If provided, only event anchors whose observed time is
        strictly before ``tau`` are counted. If None, no truncation is applied.
    tied_tol: float, optional (default=1e-8)
        Absolute tolerance for risk-score ties.

    Returns
    -------
    _ConcordanceCounts
        Raw concordance counts before tie-mode finalization.
    """
    if sample_weights is None:
        sample_weights = np.ones(event_times.shape[0], dtype=float)

    anchor_indices = np.flatnonzero(event_indicators)
    anchor_col_by_sample = np.full(event_times.shape[0], -1, dtype=int)
    anchor_col_by_sample[anchor_indices] = np.arange(anchor_indices.shape[0])

    counts = _ConcordanceCounts()
    for block, later_samples in _iter_time_blocks(event_times):
        if tau is not None and event_times[block[0]] >= tau:
            break

        event_anchors = block[event_indicators[block]]
        if event_anchors.shape[0] == 0:
            continue

        counts.time_tie_pairs += _same_time_pair_weight(sample_weights[event_anchors])

        candidate_indices = np.concatenate(
            (block[~event_indicators[block]], later_samples)
        )
        if candidate_indices.shape[0] == 0:
            continue

        for anchor_index in event_anchors:
            anchor_col = anchor_col_by_sample[anchor_index]
            if anchor_pair_weights is None:
                pair_weights = (
                    sample_weights[anchor_index] * sample_weights[candidate_indices]
                )
            else:
                pair_weights = np.full(
                    candidate_indices.shape[0],
                    anchor_pair_weights[anchor_index],
                    dtype=float,
                )
            counts += _count_directed_risk_pairs(
                np.full(candidate_indices.shape[0], anchor_index, dtype=int),
                candidate_indices,
                risk_scores[:, anchor_col],
                pair_weights=pair_weights,
                tied_tol=tied_tol,
            )

    return counts
