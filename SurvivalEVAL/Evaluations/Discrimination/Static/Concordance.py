from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from SurvivalEVAL.Evaluations.util import (
    _inverse_censoring_weights,
    check_and_convert,
    check_and_convert_event_data,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    KaplanMeier,
    KaplanMeierArea,
)

from .._concordance_utils import (
    _check_has_any_pairs,
    _ConcordanceCounts,
    _count_directed_risk_pairs,
    _FenwickTree,
    _finalize_counts,
    _ipcw_weights,
    _is_before_tau,
    _iter_time_blocks,
    _normalize_ties,
    _same_time_pair_weight,
    _validate_tau,
)


def concordance(
    predicted_times: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None = None,
    train_event_indicators: np.ndarray | None = None,
    method: str = "Harrell",
    ties: str = "Risk",
    tau: float | None = None,
) -> tuple[float, float, float]:
    """
    Calculate the concordance index between the predicted survival times and the true survival times.

    Parameters
    ----------
    predicted_times: np.ndarray, shape = (n_samples,)
        One predicted time per subject. Infinite predictions are allowed to
        represent survival beyond the model horizon; NaN is invalid.
    event_times: np.ndarray, shape = (n_samples,)
        Finite, non-negative observed event or censoring times.
    event_indicators: np.ndarray, shape = (n_samples,)
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    train_event_times: np.ndarray, shape = (n_train_samples,), optional
        The true survival times of the training set. Required for "Uno",
        "IPCW", and "Margin".
    train_event_indicators: np.ndarray, shape = (n_train_samples,), optional
        Binary training-set event indicators: 1 denotes an observed event and
        0 denotes a censored observation. Required for "Uno", "IPCW", and
        "Margin".
    method: str, optional (default="Harrell")
        A string indicating the method for constructing the pairs of samples.
        Options are "Harrell" (default), "Naive", "Uno", "IPCW", or "Margin".
        "Harrell": comparable pairs are anchored by samples with observed
        events. If sample i has an observed event at time t_i, it is compared
        with samples whose observed event/censoring time is greater than t_i.
        "Naive": alias of "Harrell".
        "Uno": Harrell-style comparable pairs weighted by inverse probability
        of censoring weights from the training data, using 1/G(anchor_time-)^2.
        Reverse KM retains its linear tail beyond training follow-up; required
        censoring probabilities must be finite and positive.
        "IPCW": alias of "Uno".
        "Margin": the pairs are constructed between all samples. A best-guess
        time for the censored samples will be calculated and used to construct
        the pairs.
    ties: str, optional (default="Risk")
        A string indicating the way ties should be handled.
        Options: "None", "Time", "Risk" (default), or "All"
        "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
        "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
        "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
        "All" includes all ties.
        Note the concordance calculation is given by
        (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
    tau: float, optional (default=None)
        Finite, non-negative truncation time. If provided, only pairs whose effective earlier or
        anchor time is strictly before ``tau`` are counted. If None, no
        truncation is applied.

    Returns
    -------
    c_index: float
        The concordance index.
    num_concordant_pairs: float
        The number of concordant pairs.
    num_total_pairs: float
        The number of total pairs.
    """
    predicted_times = check_and_convert(predicted_times)
    event_times, event_indicators = check_and_convert_event_data(
        event_times, event_indicators
    )
    if predicted_times.ndim != 1 or predicted_times.shape != event_times.shape:
        raise ValueError("predicted_times must have one entry per observed subject.")
    method = method.lower()
    if method not in {"harrell", "naive", "uno", "ipcw", "margin"}:
        raise ValueError(
            "method must be 'Harrell', 'Naive', 'Uno', 'IPCW', or 'Margin'."
        )
    ties = _normalize_ties(ties)
    tau = _validate_tau(tau)

    if method == "harrell" or method == "naive":
        risks = -1 * predicted_times
        counts = _right_censored_risk_counts(
            event_indicators, event_times, risks, tau=tau
        )
    elif method == "uno" or method == "ipcw":
        ipcw_weights = _ipcw_weights(
            event_times,
            event_indicators,
            train_event_times,
            train_event_indicators,
            ties,
            tau,
        )
        counts = _right_censored_risk_counts(
            event_indicators,
            event_times,
            -1 * predicted_times,
            sample_weights=ipcw_weights,
            anchor_pair_weights=np.square(ipcw_weights),
            tau=tau,
        )
    elif method == "margin":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Margin' is chosen, training set information must be provided."
            raise ValueError(error)

        train_event_times, train_event_indicators = check_and_convert_event_data(
            train_event_times, train_event_indicators
        )

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / (
            (1 - min(km_model.survival_probabilities))
            / (0 - max(km_model.survival_times))
        )
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)
        risks = -1 * predicted_times

        censor_times = event_times[~event_indicators]
        partial_weights = np.ones_like(event_indicators, dtype=float)
        partial_weights[~event_indicators] = 1 - km_model.predict(censor_times)

        best_guesses = km_model.best_guess(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[
            censor_times > km_linear_zero
        ]

        bg_event_times = np.copy(event_times)
        bg_event_times[~event_indicators] = best_guesses
        counts = _margin_counts(
            event_indicators,
            event_times,
            estimate=risks,
            bg_event_time=bg_event_times,
            partial_weights=partial_weights,
            tau=tau,
        )

    _check_has_any_pairs(counts)
    return _finalize_counts(counts, ties)


def concordance_cr_per_event(
    predicted_risks: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None = None,
    train_event_indicators: np.ndarray | None = None,
    method: str = "Naive",
    ties: str = "Risk",
    tau: float | None = None,
    event_of_interest: int = 1,
) -> tuple[float, float, float]:
    """Calculate concordance for one cause in the presence of competing risks.

    Parameters
    ----------
    predicted_risks : array-like, shape (n_samples,)
        Finite risk scores for the selected cause; higher means greater risk.
        For horizon-specific concordance, supply the selected cause's CIF at
        the same horizon for every subject. Scores need not be probabilities.
    event_times : array-like, shape (n_samples,)
        Finite, non-negative observed event or censoring times.
    event_indicators : array-like, shape (n_samples,)
        Non-negative integer labels: 0 for censoring, ``event_of_interest``
        for the selected cause, and other positive labels for competing events.
    train_event_times, train_event_indicators : array-like, optional
        Training outcomes with the same label convention. Both are required
        for IPCW. All positive labels denote observed events when fitting
        the censoring distribution; competing events are not censorings.
    method : {"Naive", "IPCW"}, default="Naive"
        Naive uses unit pair weights. IPCW uses reverse Kaplan-Meier
        censoring survival G from the training data: weight 1/G(T_i-)^2 for
        type A pairs and 1/(G(T_i-) G(T_j-)) for type B pairs.
    ties : {"None", "Time", "Risk", "All"}, default="Risk"
        Same policy as ``concordance``. Risk ties (absolute tolerance 1e-8)
        receive half credit for Risk/All and are excluded otherwise. Same-time
        events of the selected cause receive half credit for Time/All and are
        excluded otherwise. Same-time competing events remain comparable.
    tau : float, optional
        Finite, non-negative truncation time. Only selected-cause anchors with
        event time strictly less than tau contribute, matching ``concordance``.
        None applies no truncation. This does not change the supplied risks.
    event_of_interest : int, default=1
        Positive integer cause label to evaluate.

    Returns
    -------
    c_index : float
        Weighted fraction of comparable pairs ranked correctly.
    num_concordant_pairs : float
        Concordant pair count, including half credit for retained ties.
    num_total_pairs : float
        Total pair count after the tie policy. Counts are weighted for IPCW.

    Notes
    -----
    An anchor i must have an observed event of the selected cause. Type A
    comparators have T_j > T_i, or T_j = T_i with j censored. Type B comparators
    have a competing event at T_j <= T_i. Unlike ordinary cause-specific
    concordance obtained by recoding competing events as censorings, this also
    includes earlier competing events.

    Uses the Wolbers pair definition implemented in hazardous, with Risk ties
    as its default. The array API accepts precomputed scores rather than
    interpolating CIF curves. Truncation is strict (< tau), whereas hazardous
    includes anchors at tau. Time/All additionally support selected-cause time
    ties according to SurvivalEVAL's existing convention.

    IPCW assumes independent censoring for the marginal censoring estimator.
    As in ``concordance``, evaluate left-limit G(t-) with the events-first
    convention for training ties, retaining linear tail extrapolation beyond follow-up.
    Nonpositive or nonfinite G at a contributing observation raises ValueError; excluded anchors and unused comparators do not require G.
    Counting takes O(n log n) time and O(n) memory without constructing pairs.

    References
    ----------
    Wolbers et al. (2014), "Concordance for prognostic models with competing
    risks", https://doi.org/10.1093/biostatistics/kxt059
    """
    risks = check_and_convert(predicted_risks)
    times, events = check_and_convert_event_data(
        event_times, event_indicators, competing_risks=True
    )
    if risks.ndim != 1 or risks.shape != times.shape:
        raise ValueError("predicted_risks must have one entry per observed subject.")
    if not np.all(np.isfinite(risks)):
        raise ValueError("predicted_risks must contain finite scores.")
    if (
        isinstance(event_of_interest, (bool, np.bool_))
        or not isinstance(event_of_interest, (int, np.integer))
        or event_of_interest < 1
    ):
        raise ValueError("event_of_interest must be a positive integer cause label.")
    ties = _normalize_ties(ties)
    tau = _validate_tau(tau)
    censoring_survival = _cr_censoring_survival(
        times, train_event_times, train_event_indicators, method
    )

    counts = _competing_risk_counts(
        risks, times, events, event_of_interest, censoring_survival, ties, tau
    )
    _check_has_any_pairs(counts)
    return _finalize_counts(counts, ties)


def concordance_cr(
    predicted_risks: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None = None,
    train_event_indicators: np.ndarray | None = None,
    method: str = "Naive",
    ties: str = "Risk",
    tau: float | None = None,
    reduction: str = "mean",
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """Summarize competing-risks concordance across all predicted causes.

    Parameters
    ----------
    predicted_risks : array-like, shape (n_samples, n_causes)
        Finite risk scores, with column k - 1 representing cause k. Higher
        scores indicate greater cause-specific risk. For horizon-specific
        concordance, supply each cause's CIF at the same prediction horizon.
    event_times : array-like, shape (n_samples,)
        Finite, non-negative event or censoring times.
    event_indicators : array-like, shape (n_samples,)
        Integer labels from 0 through n_causes; 0 denotes censoring.
    train_event_times, train_event_indicators : array-like, optional
        Training outcomes, required for IPCW. All positive training labels
        denote observed events when fitting censoring survival.
    method : {"Naive", "IPCW"}, default="Naive"
        Unit pair weights or censoring weights. See ``concordance_cr_per_event``.
    ties : {"None", "Time", "Risk", "All"}, default="Risk"
        Tie policy applied separately to each cause.
    tau : float, optional
        Strict truncation of selected-cause anchors (< tau), without changing
        the supplied risks. See ``concordance_cr_per_event``.
    reduction : {"mean", "pooled"}, default="mean"
        Mean weights every cause equally. Pooled divides the sum of concordant
        pair counts by the sum of total pair counts, weighting each cause by
        its available pair count (or pair weight for IPCW).
    return_details : bool, default=False
        Also return dictionaries ``per_event_scores``,
        ``per_event_concordant_pairs`` and ``per_event_total_pairs``, keyed
        by cause label.

    Returns
    -------
    c_index : float
        Aggregate of per-cause concordance scores, in [0, 1] when defined.
    details : dict, optional
        Per-cause scores and counts after applying the tie policy.

    Notes
    -----
    This is a summary of the event-specific Wolbers C-indexes. The mean is NaN
    if any cause is undefined; undefined causes are not silently omitted or
    assigned a chance score. Causes with no retained pairs have zero weight
    under pooled reduction, which is NaN if all causes have zero pair counts.
    Every predicted cause is included in the details, even if unobserved or
    excluded by tau. The IPCW censoring model is fitted once for all causes.

    See Also
    --------
    concordance_cr_per_event : Pair definition, weighting and assumptions.
    """
    risks = check_and_convert(predicted_risks)
    times, events = check_and_convert_event_data(
        event_times, event_indicators, competing_risks=True
    )
    if risks.ndim != 2 or risks.shape[0] != times.size:
        raise ValueError("predicted_risks must have shape (n_samples, n_causes).")
    if not np.all(np.isfinite(risks)):
        raise ValueError("predicted_risks must contain finite scores.")
    if np.any(events > risks.shape[1]):
        raise ValueError("Event labels must not exceed the number of predicted causes.")
    reduction = reduction.lower()
    if reduction not in {"mean", "pooled"}:
        raise ValueError("reduction must be 'mean' or 'pooled'.")
    ties = _normalize_ties(ties)
    tau = _validate_tau(tau)
    censoring_survival = _cr_censoring_survival(
        times, train_event_times, train_event_indicators, method
    )

    per_event_results = np.empty((risks.shape[1], 3), dtype=float)
    for k in range(risks.shape[1]):
        counts = _competing_risk_counts(
            risks[:, k],
            times,
            events,
            k + 1,
            censoring_survival,
            ties,
            tau,
        )
        per_event_results[k] = _finalize_counts(counts, ties)
    scores, concordant, total = per_event_results.T
    if reduction == "mean":
        score = float(scores.mean())
    else:
        total_pairs = total.sum()
        score = float(concordant.sum() / total_pairs) if total_pairs else float("nan")
    if not return_details:
        return score
    return score, {
        "per_event_scores": {k + 1: float(value) for k, value in enumerate(scores)},
        "per_event_concordant_pairs": {
            k + 1: float(value) for k, value in enumerate(concordant)
        },
        "per_event_total_pairs": {k + 1: float(value) for k, value in enumerate(total)},
    }


def _cr_censoring_survival(
    event_times: np.ndarray,
    train_event_times: np.ndarray | None,
    train_event_indicators: np.ndarray | None,
    method: str,
) -> np.ndarray | None:
    """Validate the method and return G(t-) for IPCW, or None for Naive.

    Fit reverse Kaplan-Meier on validated training outcomes, treating all
    positive cause labels as observed events. Return censoring survival at
    each test event time; the counter checks positivity only where needed.
    """
    method = method.lower()
    if method not in {"naive", "ipcw"}:
        raise ValueError("method must be 'Naive' or 'IPCW'.")
    if method == "naive":
        return None
    if train_event_times is None or train_event_indicators is None:
        raise ValueError("Training event times and indicators are required for IPCW.")
    train_times, train_events = check_and_convert_event_data(
        train_event_times, train_event_indicators, competing_risks=True
    )
    return KaplanMeier(train_times, train_events > 0, reverse=True).predict(
        event_times, side="left"
    )


def _competing_risk_counts(
    predicted_risks: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    event_of_interest: int,
    censoring_survival: np.ndarray | None = None,
    ties: str = "risk",
    tau: float | None = None,
) -> _ConcordanceCounts:
    """Count validated competing-risk pairs in O(n log n) time and O(n) memory.

    Parameters
    ----------
    predicted_risks, event_times, event_indicators : ndarray, shape (n_samples,)
        Finite risks, observed times, and integer cause labels.
    event_of_interest : int
        Selected positive cause label.
    censoring_survival : ndarray, shape (n_samples,), optional
        Marginal censoring survival G(t-) at observed times. None uses unit weights.
    ties : str, default="risk"
        Normalized tie policy; controls which observations need positive G(t-).
    tau : float, optional
        Strict truncation of selected-cause anchor times.

    Returns
    -------
    _ConcordanceCounts
        Weighted directed pair counts and selected-cause time ties.
    """
    # Evaluate tolerance bounds in float64; float32 rounding can otherwise
    # treat distinct risks more than 1e-8 apart as ties.
    predicted_risks = predicted_risks.astype(float, copy=False)
    selected = event_indicators == event_of_interest
    competing = (event_indicators > 0) & ~selected
    sample_weights = np.ones(event_times.shape, dtype=float)
    if censoring_survival is not None:
        anchors = selected & _is_before_tau(event_times, tau)
        # Every non-final anchor has type A comparators. Final anchors can
        # still contribute through tied non-selected subjects or earlier competitors.
        final_time = event_times.max()
        final_block = event_times == final_time
        final_has_comparators = np.any(final_block & ~selected) or (
            ties in {"time", "all"} and np.count_nonzero(final_block & selected) > 1
        )
        first_competing_time = (
            event_times[competing].min() if np.any(competing) else np.inf
        )
        contributing = anchors & (
            (event_times < final_time)
            | (event_times > first_competing_time)
            | final_has_comparators
        )
        needed = contributing.copy()
        if np.any(contributing):
            needed |= competing & (event_times < event_times[contributing].max())
        sample_weights[needed] = _inverse_censoring_weights(censoring_survival[needed])

    counts = _right_censored_risk_counts(
        selected,
        event_times,
        predicted_risks,
        sample_weights=sample_weights,
        anchor_pair_weights=np.square(sample_weights),
        tau=tau,
    )
    # The ordinary counter also includes tied competing events as comparators.
    # Their type A and B weights coincide under marginal KM, so add only
    # strictly earlier competing events here to avoid counting ties twice.
    if not np.any(competing):
        return counts
    risk_values, risk_ranks = np.unique(predicted_risks, return_inverse=True)
    tie_starts = np.searchsorted(risk_values, predicted_risks - 1e-8, side="left")
    tie_stops = np.searchsorted(risk_values, predicted_risks + 1e-8, side="right")
    tree = _FenwickTree(risk_values.size)
    candidate_total = 0.0
    for block, _ in _iter_time_blocks(event_times):
        if tau is not None and event_times[block[0]] >= tau:
            break
        for i in block[selected[block]]:
            lower = tree.prefix_sum(int(tie_starts[i]))
            through_ties = tree.prefix_sum(int(tie_stops[i]))
            counts.concordant += sample_weights[i] * lower
            counts.risk_tie_pairs += sample_weights[i] * (through_ties - lower)
            counts.discordant += sample_weights[i] * (candidate_total - through_ties)
        for j in block[competing[block]]:
            tree.add(int(risk_ranks[j]), sample_weights[j])
            candidate_total += sample_weights[j]
    return counts


def _right_censored_risk_counts(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    estimate: np.ndarray,
    sample_weights: np.ndarray | None = None,
    anchor_pair_weights: np.ndarray | None = None,
    tau: float | None = None,
    tied_tol: float = 1e-8,
) -> _ConcordanceCounts:
    """Count right-censored comparable pairs for a risk score.

    Candidate risks are maintained in a Fenwick tree, giving ``O(n log n)``
    time and ``O(n)`` memory without materializing individual pairs.

    Parameters
    ----------
    event_indicator: np.ndarray, shape = (n_samples,)
        Boolean event indicators, where True denotes an observed event.
    event_time: np.ndarray, shape = (n_samples,)
        Observed event or censoring times.
    estimate: np.ndarray, shape = (n_samples,)
        Risk scores. Higher scores indicate higher risk.
    sample_weights: np.ndarray, shape = (n_samples,), optional
        Optional symmetric sample weights. Pair weights are products of anchor
        and candidate sample weights unless ``anchor_pair_weights`` is provided.
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
        Raw concordance counts before tie-mode finalization. ``concordant``,
        ``discordant``, and ``risk_tie_pairs`` count comparable pairs;
        ``time_tie_pairs`` counts same-time event pairs.
    """
    if event_time.shape[0] == 0:
        return _ConcordanceCounts()
    if sample_weights is None:
        sample_weights = np.ones(event_time.shape[0], dtype=float)

    # Candidate masses are indexed by risk rank. For the symmetric weighted
    # case a pair receives w_i * w_j. With anchor_pair_weights, every candidate
    # has unit mass and the anchor supplies the complete pair weight.
    if anchor_pair_weights is None:
        candidate_mass = sample_weights
        anchor_scale = sample_weights
    else:
        candidate_mass = np.ones(event_time.shape[0], dtype=float)
        anchor_scale = anchor_pair_weights

    risk_values, risk_ranks = np.unique(estimate, return_inverse=True)
    tie_starts = np.searchsorted(risk_values, estimate - tied_tol, side="left")
    tie_stops = np.searchsorted(risk_values, estimate + tied_tol, side="right")
    risk_tree = _FenwickTree(risk_values.shape[0])
    candidate_total = 0.0

    counts = _ConcordanceCounts()
    order = np.argsort(event_time, kind="stable")
    end = order.shape[0]
    while end > 0:
        start = end - 1
        block_time = event_time[order[start]]
        while start > 0 and event_time[order[start - 1]] == block_time:
            start -= 1
        block = order[start:end]
        censored_candidates = block[~event_indicator[block]]
        event_anchors = block[event_indicator[block]]

        # Censoring at the anchor time is comparable, whereas an event at the
        # same time is not. Add censored samples before querying event anchors.
        for index in censored_candidates:
            mass = float(candidate_mass[index])
            risk_tree.add(int(risk_ranks[index]), mass)
            candidate_total += mass

        if (tau is None or block_time < tau) and event_anchors.shape[0] > 0:
            counts.time_tie_pairs += _same_time_pair_weight(
                sample_weights[event_anchors]
            )
            for index in event_anchors:
                lower_mass = risk_tree.prefix_sum(int(tie_starts[index]))
                through_ties_mass = risk_tree.prefix_sum(int(tie_stops[index]))
                tie_mass = through_ties_mass - lower_mass
                scale = float(anchor_scale[index])
                counts.concordant += scale * lower_mass
                counts.risk_tie_pairs += scale * tie_mass
                counts.discordant += scale * (candidate_total - through_ties_mass)

        # This block becomes strictly later than all subsequent blocks.
        for index in event_anchors:
            mass = float(candidate_mass[index])
            risk_tree.add(int(risk_ranks[index]), mass)
            candidate_total += mass
        end = start

    return counts


def _margin_counts(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    estimate: np.ndarray,
    bg_event_time: np.ndarray,
    partial_weights: np.ndarray,
    tau: float | None = None,
    tied_tol: float = 1e-8,
) -> _ConcordanceCounts:
    """Count Margin concordance pairs from best-guess times.

    Margin first scores all samples as events at their best-guess event times
    with partial weights. Harrell-comparable observed pairs are then restored
    to their original observed ordering at full weight.

    Parameters
    ----------
    event_indicator: np.ndarray, shape = (n_samples,)
        Boolean event indicators, where True denotes an observed event.
    event_time: np.ndarray, shape = (n_samples,)
        Observed event or censoring times.
    estimate: np.ndarray, shape = (n_samples,)
        Risk scores. Higher scores indicate higher risk.
    bg_event_time: np.ndarray, shape = (n_samples,)
        Best-guess event times for all samples.
    partial_weights: np.ndarray, shape = (n_samples,)
        Per-sample weights used for the best-guess baseline.
    tau: float, optional (default=None)
        Truncation time. Best-guess baseline pairs use the best-guess earlier
        time, while observed-pair replacements use the observed event anchor
        time. In both cases the effective time must be strictly before ``tau``.
        If None, no truncation is applied.
    tied_tol: float, optional (default=1e-8)
        Absolute tolerance for risk-score ties.

    Returns
    -------
    _ConcordanceCounts
        Raw Margin concordance counts before tie-mode finalization.
    """
    counts = _right_censored_risk_counts(
        np.ones_like(event_indicator, dtype=bool),
        bg_event_time,
        estimate,
        sample_weights=partial_weights,
        tau=tau,
        tied_tol=tied_tol,
    )

    for anchor_indices, candidate_indices in _iter_comparable_event_pairs(
        event_indicator, event_time
    ):
        baseline_weights = (
            partial_weights[anchor_indices] * partial_weights[candidate_indices]
        )
        anchor_before = bg_event_time[anchor_indices] < bg_event_time[candidate_indices]
        candidate_before = (
            bg_event_time[anchor_indices] > bg_event_time[candidate_indices]
        )
        tied_time = ~(anchor_before | candidate_before)
        baseline_anchor_before_tau = _is_before_tau(bg_event_time[anchor_indices], tau)
        baseline_candidate_before_tau = _is_before_tau(
            bg_event_time[candidate_indices], tau
        )
        tied_time_before_tau = baseline_anchor_before_tau

        anchor_baseline_included = anchor_before & baseline_anchor_before_tau
        candidate_baseline_included = candidate_before & baseline_candidate_before_tau
        tied_time_baseline_included = tied_time & tied_time_before_tau
        observed_replacement_included = _is_before_tau(event_time[anchor_indices], tau)

        counts += _count_directed_risk_pairs(
            anchor_indices[anchor_baseline_included],
            candidate_indices[anchor_baseline_included],
            estimate,
            pair_weights=-baseline_weights[anchor_baseline_included],
            tied_tol=tied_tol,
        )
        counts += _count_directed_risk_pairs(
            candidate_indices[candidate_baseline_included],
            anchor_indices[candidate_baseline_included],
            estimate,
            pair_weights=-baseline_weights[candidate_baseline_included],
            tied_tol=tied_tol,
        )
        counts.time_tie_pairs -= baseline_weights[tied_time_baseline_included].sum()
        counts += _count_directed_risk_pairs(
            anchor_indices[observed_replacement_included],
            candidate_indices[observed_replacement_included],
            estimate,
            pair_weights=np.ones(observed_replacement_included.sum(), dtype=float),
            tied_tol=tied_tol,
        )

    return counts


def _iter_comparable_event_pairs(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield directed comparable event pairs in sample-index coordinates.

    Parameters
    ----------
    event_indicator: np.ndarray, shape = (n_samples,)
        Boolean event indicators, where True denotes an observed event.
    event_time: np.ndarray, shape = (n_samples,)
        Observed event or censoring times.

    Yields
    ------
    anchor_indices: np.ndarray, shape = (n_pairs,)
        Repeated sample indices for the observed-event anchor side of each pair.
    candidate_indices: np.ndarray, shape = (n_pairs,)
        Sample indices that are either later in observed time or censored at
        the same observed time as the anchor.
    """
    for block, later_samples in _iter_time_blocks(event_time):
        event_anchors = block[event_indicator[block]]
        if event_anchors.shape[0] > 0:
            same_time_censored = block[~event_indicator[block]]
            candidate_indices = np.concatenate((same_time_censored, later_samples))
            for anchor_index in event_anchors:
                if candidate_indices.shape[0] > 0:
                    yield (
                        np.full(candidate_indices.shape[0], anchor_index, dtype=int),
                        candidate_indices,
                    )
