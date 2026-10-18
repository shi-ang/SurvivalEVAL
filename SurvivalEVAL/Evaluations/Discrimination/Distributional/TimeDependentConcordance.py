from __future__ import annotations

from collections.abc import Callable

import numpy as np

from SurvivalEVAL.Evaluations.util import (
    check_and_convert,
    check_and_convert_event_data,
)

from .._concordance_utils import (
    _check_has_any_pairs,
    _ConcordanceCounts,
    _count_directed_risk_pairs,
    _finalize_counts,
    _ipcw_weights,
    _iter_time_blocks,
    _normalize_ties,
    _same_time_pair_weight,
    _validate_tau,
)


def _normalize_time_dependent_method(method: str) -> str:
    """
    Normalize and validate a time-dependent concordance method.

    Parameters
    ----------
    method : str
        Time-dependent concordance method, "Antolini", "Naive", or "IPCW", case-insensitive.

    Returns
    -------
    str
        Validated lowercase method name.
    """
    normalized = method.lower()
    if normalized not in {"antolini", "naive", "ipcw"}:
        raise ValueError(
            f"Unsupported method: {normalized}. Supported methods are "
            "'Antolini', 'Naive', and 'IPCW'."
        )
    return normalized


def _select_risk_anchors(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    tau: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return all event anchors and those that require predicted risk scores.

    Every event before the final observed time has a later candidate. Events
    at the final time require risks only when that block also contains a
    censored sample. Final event-only blocks can contribute time ties, but the
    concordance counter never reads their risk columns.

    Parameters
    ----------
    event_times : ndarray
        Observed event or censoring times, one per sample.
    event_indicators : ndarray
        Boolean observed-event mask, with True for events and False for censoring.
    tau : float | None
        Finite, non-negative truncation time; None leaves times untruncated.

    Returns
    -------
    tuple[ndarray, ndarray]
        Observed-event anchor times and a boolean mask identifying anchors that require risk predictions.
    """
    anchor_times = event_times[event_indicators]
    if anchor_times.size == 0:
        return anchor_times, np.zeros(0, dtype=bool)

    final_time = np.max(event_times)
    included = anchor_times < final_time
    if np.any((event_times == final_time) & ~event_indicators):
        included |= anchor_times == final_time
    if tau is not None:
        included &= anchor_times < tau
    return anchor_times, included


def concordance_time_dependent(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None = None,
    train_event_indicators: np.ndarray | None = None,
    method: str = "Antolini",
    ties: str = "Risk",
    tau: float | None = None,
) -> tuple[float, float, float]:
    """
    Calculate the time-dependent concordance index between the predicted risk scores and the true survival times.

    Parameters
    ----------
    risk_scores : ndarray, shape = (n_samples, n_anchor_times)
        The predicted risk scores for each sample at each anchor time.
        The risk scores should be ordered such that higher scores indicate higher risk
        (i.e., lower survival probability or higher hazard score).
        ``risk_scores[i, k]`` is the risk score for test sample ``i``
        evaluated at the kth observed-event anchor time. The kth anchor time
        corresponds to ``event_times[np.flatnonzero(event_indicators)[k]]``.
        Anchor times are sample-level observed events, not unique event times.
        If multiple observed events share the same time, ``risk_scores`` must
        include one column for each of those samples in observed-event sample
        order.
    event_times : ndarray, shape = (n_samples,)
        Finite, non-negative observed event or censoring times.
    event_indicators : ndarray, shape = (n_samples,)
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    train_event_times : ndarray, shape = (n_train_samples,), optional
        The true survival times of the training set. Required for "IPCW".
    train_event_indicators : ndarray, shape = (n_train_samples,), optional
        Binary training-set event indicators: 1 denotes an observed event and
        0 denotes a censored observation. Required for "IPCW".
    method : str, optional, default="Antolini"
        A string indicating the method for constructing the pairs of samples.
        Options are "Antolini" (default), "Naive", or "IPCW".
        "Antolini": comparable pairs are anchored by samples with observed
        events. If sample i has an observed event at time t_i, it is compared
        with samples whose observed event/censoring time is greater than t_i [1]_.
        "Naive": alias of "Antolini".
        "IPCW": Antolini-style comparable pairs weighted by inverse probability
        of censoring weights from the training data [2]_, [3]_.
    ties : str, optional, default="Risk"
        A string indicating the way ties should be handled.
        Options: "None", "Time", "Risk" (default), or "All"
    tau : float, optional
        Finite, non-negative truncation time. If provided, only event anchors whose observed time is
        strictly before ``tau`` are counted. If None, no truncation is applied.

    Returns
    -------
    c_index : float
        The concordance index.
    num_concordant_pairs : float
        The number of concordant pairs.
    num_total_pairs : float
        The number of total pairs.

    References
    ----------
    .. [1] Antolini, L.; Boracchi, P.; Biganzoli, E. (2005). "A time-dependent
       discrimination index for survival data". Statistics in Medicine,
       24(24), 3927-3944. https://doi.org/10.1002/sim.2427
    .. [2] Gandy, A.; Matcham, T. J. (2025). "A proper concordance index for
       models with crossing hazards". Scandinavian Journal of Statistics,
       52(4), 1479-1504. https://doi.org/10.1111/sjos.70000
    .. [3] Faruk, A.; Palczewski, J.; Aivaliotis, G. (2025). "A new discrimination
       measure for assessing predictive performance of non-linear survival models".
       arXiv preprint, arXiv:2504.05630. https://arxiv.org/abs/2504.05630
    """
    if callable(risk_scores):
        raise TypeError("risk_scores must be a numeric array.")

    return _concordance_time_dependent(
        risk_scores,
        event_times,
        event_indicators,
        train_event_times,
        train_event_indicators,
        method,
        ties,
        tau,
    )


def _concordance_time_dependent(
    risk_scores: np.ndarray | Callable[[int, np.ndarray], np.ndarray],
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None = None,
    train_event_indicators: np.ndarray | None = None,
    method: str = "Antolini",
    ties: str = "Risk",
    tau: float | None = None,
) -> tuple[float, float, float]:
    """
    Validate outcomes and evaluate a risk array or a per-sample predictor.

    A callable receives a sample index and unique, increasing anchor times,
    and returns a 1-D array of that sample's risks at those times. This lets
    the evaluator count pairs without allocating the dense risk matrix.

    Parameters
    ----------
    risk_scores : ndarray | Callable[[int, ndarray], ndarray]
        Risk matrix indexed by sample and event anchor, or a callable taking (sample_index, anchor_times).
    event_times : ndarray
        Observed event or censoring times, one per sample.
    event_indicators : ndarray
        Observed status labels; 0 denotes censoring and positive labels denote observed events.
    train_event_times : ndarray | None
        Training event or censoring times used to estimate the censoring distribution.
    train_event_indicators : ndarray | None
        Training status labels, with 0 for censoring and positive labels for observed events.
    method : str
        Time-dependent concordance method, "Antolini", "Naive", or "IPCW", case-insensitive.
    ties : str
        Tie policy: "None", "Time", "Risk", or "All", case-insensitive.
    tau : float | None
        Finite, non-negative truncation time; None leaves times untruncated.

    Returns
    -------
    tuple[float, float, float]
        Concordance index, weighted concordant-pair count, and weighted comparable-pair count.
    """
    event_times, event_indicators = check_and_convert_event_data(
        event_times, event_indicators
    )
    if not np.any(event_indicators):
        raise ValueError(
            "Data has no observed events, cannot estimate time-dependent concordance index."
        )
    if not callable(risk_scores):
        risk_scores = check_and_convert(risk_scores)
        if risk_scores.ndim != 2:
            raise ValueError(
                "risk_scores must have shape (n_samples, n_observed_events)."
            )
        if risk_scores.shape[0] != event_times.size:
            raise ValueError("risk_scores must have one row per observed subject.")
        if risk_scores.shape[1] != np.count_nonzero(event_indicators):
            raise ValueError("Risk columns must match the number of observed events.")

    method = _normalize_time_dependent_method(method)
    ties = _normalize_ties(ties)
    tau = _validate_tau(tau)

    if method == "antolini" or method == "naive":
        sample_weights = None
        anchor_pair_weights = None
    elif method == "ipcw":
        sample_weights = _ipcw_weights(
            event_times,
            event_indicators,
            train_event_times,
            train_event_indicators,
            ties,
            tau,
        )
        anchor_pair_weights = np.square(sample_weights)
    count_risks = (
        _time_dependent_risk_counts_from_predictions
        if callable(risk_scores)
        else _time_dependent_risk_counts
    )
    counts = count_risks(
        risk_scores=risk_scores,
        event_times=event_times,
        event_indicators=event_indicators,
        sample_weights=sample_weights,
        anchor_pair_weights=anchor_pair_weights,
        tau=tau,
    )

    _check_has_any_pairs(counts)
    return _finalize_counts(counts, ties)


def _time_dependent_risk_counts_from_predictions(
    risk_scores: Callable[[int, np.ndarray], np.ndarray],
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    sample_weights: np.ndarray | None = None,
    anchor_pair_weights: np.ndarray | None = None,
    tau: float | None = None,
    tied_tol: float = 1e-8,
) -> _ConcordanceCounts:
    """
    Count pairs by ranking risks within equal-event-time groups.

    Visit samples in increasing observed-time order, processing events before
    censorings at the same time. Sort each event group's own risks once; later
    samples query weighted ranks in those groups instead of visiting each
    anchor. Each sample is predicted at most once, at unique contributing
    anchor times up to its observed time.

    Counting uses O(n) working memory and O(n log n + n U log(M + 1)) time,
    where U is the number of contributing event times and M is the largest
    event group. Small prediction batches contain at most n scores; no dense
    sample-by-anchor matrix is constructed. These bounds exclude prediction
    costs and the stored input curves.

    Parameters
    ----------
    risk_scores : Callable[[int, ndarray], ndarray]
        Callable taking (sample_index, anchor_times) and returning that sample's risk scores.
    event_times : ndarray
        Observed event or censoring times, one per sample.
    event_indicators : ndarray
        Observed status labels; 0 denotes censoring and positive labels denote observed events.
    sample_weights : ndarray | None
        Non-negative weights for individual samples; None assigns unit weights.
    anchor_pair_weights : ndarray | None
        Pair weights associated with event anchors; None assigns unit weights.
    tau : float | None
        Finite, non-negative truncation time; None leaves times untruncated.
    tied_tol : float
        Maximum absolute risk difference treated as a risk tie.

    Returns
    -------
    _ConcordanceCounts
        Weighted counts of concordant, discordant, risk-tied, and time-tied pairs.
    """
    if sample_weights is None:
        sample_weights = np.ones(event_times.shape[0], dtype=float)

    _, included = _select_risk_anchors(event_times, event_indicators, tau)
    anchor_indices = np.flatnonzero(event_indicators)[included]
    anchor_indices = anchor_indices[
        np.argsort(event_times[anchor_indices], kind="stable")
    ]
    anchor_times = event_times[anchor_indices]
    unique_times, group_starts, group_sizes = np.unique(
        anchor_times, return_index=True, return_counts=True
    )
    group_ends = group_starts + group_sizes
    anchor_col_by_sample = np.full(event_times.shape[0], -1, dtype=int)
    anchor_col_by_sample[anchor_indices] = np.arange(anchor_indices.size)
    anchor_risks = np.empty(anchor_indices.size, dtype=float)
    anchor_weights = (
        sample_weights[anchor_indices]
        if anchor_pair_weights is None
        else anchor_pair_weights[anchor_indices]
    )
    # Each group has a compact range-sum tree so interval queries never
    # subtract unrelated weights, which can erase small pair counts.
    weight_trees = np.empty(2 * anchor_indices.size, dtype=float)
    batch_size = min(256, max(1, event_times.size // max(1, unique_times.size)))

    counts = _ConcordanceCounts()
    for block, _ in _iter_time_blocks(event_times):
        block_time = event_times[block[0]]
        events = block[event_indicators[block]]
        censored = block[~event_indicators[block]]
        if tau is None or block_time < tau:
            counts.time_tie_pairs += _same_time_pair_weight(sample_weights[events])

        before = np.searchsorted(unique_times, block_time, side="left")
        through = np.searchsorted(unique_times, block_time, side="right")
        if through == 0:
            continue
        target_times = unique_times[:through]

        # Events at the same time are time ties, not directed risk pairs.
        # Sort their own risks after predicting all events and before querying
        # the same-time censorings, which are comparable with those events.
        for is_event, samples in ((True, events), (False, censored)):
            stop = before if is_event else through
            for offset in range(0, samples.size, batch_size):
                batch = samples[offset : offset + batch_size]
                predicted_risks = np.empty((batch.size, through))
                for row, sample_index in enumerate(batch):
                    anchor_col = anchor_col_by_sample[sample_index]
                    if stop == 0 and anchor_col < 0:
                        continue
                    predicted_risks[row] = risk_scores(sample_index, target_times)
                    if anchor_col >= 0:
                        anchor_risks[anchor_col] = predicted_risks[row, -1]
                if stop == 0:
                    continue

                ranks = _grouped_risk_ranks(
                    anchor_risks,
                    group_starts[:stop],
                    group_ends[:stop],
                    predicted_risks[:, :stop],
                    tied_tol,
                )
                range_weights = _grouped_risk_weights(
                    weight_trees, group_starts[:stop], group_ends[:stop], ranks
                )
                pair_weights = (
                    sample_weights[batch] if anchor_pair_weights is None else 1.0
                )
                discordant, risk_ties, concordant = np.sum(
                    range_weights.sum(axis=2) * pair_weights, axis=1
                )
                counts.discordant += discordant
                counts.risk_tie_pairs += risk_ties
                counts.concordant += concordant

            if is_event and through > before:
                start, end = group_starts[before], group_ends[before]
                group_risks = anchor_risks[start:end]
                # NaN risks always count as discordant. Sort them with -inf
                # so they remain in every candidate's discordant prefix.
                risk_order = np.argsort(
                    np.where(np.isnan(group_risks), -np.inf, group_risks),
                    kind="stable",
                )
                anchor_risks[start:end] = group_risks[risk_order]
                group_weights = anchor_weights[start:end][risk_order]
                tree = weight_trees[2 * start : 2 * end]
                size = end - start
                tree[size:] = group_weights
                for node in range(size - 1, 0, -1):
                    tree[node] = tree[2 * node] + tree[2 * node + 1]

    return counts


def _grouped_risk_ranks(
    anchor_risks: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    candidate_risks: np.ndarray,
    tied_tol: float,
) -> np.ndarray:
    """
    Find discordant and nonconcordant prefixes of sorted anchor groups.

    The returned array has shape (2, batch_size, n_groups). Binary searches
    compare candidate-minus-anchor risks directly: searching for risk plus or
    minus the tolerance can round differently at floating-point boundaries.

    Parameters
    ----------
    anchor_risks : ndarray
        Sorted anchor risks concatenated in group order.
    group_starts : ndarray
        Start offsets of the concatenated anchor groups.
    group_ends : ndarray
        Exclusive end offsets of the concatenated anchor groups.
    candidate_risks : ndarray
        Candidate risks at each anchor group, with shape (batch_size, n_groups).
    tied_tol : float
        Maximum absolute risk difference treated as a risk tie.

    Returns
    -------
    ndarray
        Prefix lengths for discordant and nonconcordant risks, with shape (2, batch_size, n_groups).
    """
    shape = (2, *candidate_risks.shape)
    low = np.broadcast_to(group_starts, shape).copy()
    high = np.broadcast_to(group_ends, shape).copy()
    while np.any(low < high):
        active = low < high
        middle = (low + high) // 2
        differences = candidate_risks - anchor_risks[np.minimum(middle, group_ends - 1)]
        # NaN risks and equal infinities subtract to NaN, which the directed
        # pair counter treats as discordant rather than as a risk tie.
        differences[np.isnan(differences)] = np.inf
        included = np.stack((differences[0] > tied_tol, differences[1] >= -tied_tol))
        low = np.where(active & included, middle + 1, low)
        high = np.where(active & ~included, middle, high)
    return low


def _grouped_risk_weights(
    weight_trees: np.ndarray,
    group_starts: np.ndarray,
    group_ends: np.ndarray,
    ranks: np.ndarray,
) -> np.ndarray:
    """
    Sum discordant, tied, and concordant intervals without subtraction.

    Each group's tree occupies twice its size, with sorted weights in the
    second half and parent sums above them (index zero is unused). Queries
    add only nodes entirely inside their interval, preserving small weights
    even when much larger weights lie outside it. The result has shape
    (3, batch_size, n_groups), matching the interval order above.

    Parameters
    ----------
    weight_trees : ndarray
        Concatenated segment trees containing anchor weights for each group.
    group_starts : ndarray
        Start offsets of the concatenated anchor groups.
    group_ends : ndarray
        Exclusive end offsets of the concatenated anchor groups.
    ranks : ndarray
        Discordant and nonconcordant prefix lengths, with shape (2, batch_size, n_groups).

    Returns
    -------
    ndarray
        Discordant, tied, and concordant weight sums, with shape (3, batch_size, n_groups).
    """
    sizes = group_ends - group_starts
    left = np.empty((3, *ranks.shape[1:]), dtype=int)
    left[0] = sizes
    left[1:] = ranks - group_starts + sizes
    right = np.empty_like(left)
    right[:-1] = left[1:]
    right[-1] = 2 * sizes

    totals = np.zeros(left.shape, dtype=float)
    offsets = 2 * group_starts
    while np.any(left < right):
        active = left < right
        take_left = active & (left % 2 == 1)
        take_right = active & (right % 2 == 1)
        totals[take_left] += weight_trees[(offsets + left)[take_left]]
        right -= take_right
        totals[take_right] += weight_trees[(offsets + right)[take_right]]
        left = (left + take_left) // 2
        right //= 2
    return totals


def _time_dependent_risk_counts(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    sample_weights: np.ndarray | None = None,
    anchor_pair_weights: np.ndarray | None = None,
    tau: float | None = None,
    tied_tol: float = 1e-8,
) -> _ConcordanceCounts:
    """
    Count Antolini-style time-dependent concordance pairs.

    Parameters
    ----------
    risk_scores : ndarray, shape = (n_samples, n_observed_events)
        Risk scores evaluated at observed-event anchor times. Higher scores
        indicate higher event risk.
    event_times : ndarray, shape = (n_samples,)
        Observed event or censoring times.
    event_indicators : ndarray, shape = (n_samples,)
        Boolean event indicators, where True denotes an observed event.
    sample_weights : ndarray, shape = (n_samples,), optional
        Optional symmetric sample weights used for same-time event ties and,
        unless ``anchor_pair_weights`` is provided, comparable pair weights.
    anchor_pair_weights : ndarray, shape = (n_samples,), optional
        Optional per-anchor pair weights. If provided, every comparable pair
        anchored by sample ``i`` receives ``anchor_pair_weights[i]``.
    tau : float, optional, default=None
        Truncation time. If provided, only event anchors whose observed time is
        strictly before ``tau`` are counted. If None, no truncation is applied.
    tied_tol : float, optional, default=1e-8
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
