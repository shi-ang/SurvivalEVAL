from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from typing_extensions import Self

from SurvivalEVAL.evaluations.util import (
    _inverse_censoring_weights,
    check_and_convert_event_data,
    validate_time_point,
)
from SurvivalEVAL.nonparametric_estimator.single_event import KaplanMeier


def _validate_tau(tau: float | None) -> float | None:
    """
    Return None or a finite, non-negative scalar truncation time.

    Parameters
    ----------
    tau : float | None
        Finite, non-negative truncation time; None leaves times untruncated.

    Returns
    -------
    float | None
        Validated truncation horizon, or None if no truncation was requested.
    """
    if tau is None:
        return None
    tau = validate_time_point(tau, "tau")
    if not np.isfinite(tau):
        raise ValueError("tau must be finite.")
    return tau


def _ipcw_weights(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None,
    train_event_indicators: np.ndarray | None,
    ties: str,
    tau: float | None,
) -> np.ndarray:
    """
    Estimate inverse censoring weights for ordinary-survival event anchors.

    Parameters
    ----------
    event_times, event_indicators : ndarray, shape (n_samples,)
        Validated observed times and binary indicators of the testing data.
    train_event_times, train_event_indicators : array-like, shape (n_train_samples,)
        Training outcomes, required to fit reverse Kaplan-Meier.
    ties : str
        Normalized concordance tie policy.
    tau : float, optional
        Validated strict truncation time for event anchors.

    Returns
    -------
    weights : ndarray, shape (n_samples,)
        1/G(t-) for contributing anchors; zero elsewhere. Directed comparable
        pairs receive the square of the anchor weight. Same-time event pairs
        receive the product of their weights when the tie policy retains them.

    Notes
    -----
    Final event-only anchors contribute only when Time/All retains their
    time ties. Unused observations do not require positive censoring survival.
    G(t-) excludes the censoring update at the anchor time, matching events-first ties.
    """
    if train_event_times is None or train_event_indicators is None:
        raise ValueError("Training event times and indicators are required for IPCW.")
    train_times, train_events = check_and_convert_event_data(
        train_event_times, train_event_indicators
    )
    contributing = event_indicators & _is_before_tau(event_times, tau)
    final_block = event_times == event_times.max()
    final_has_comparators = np.any(final_block & ~event_indicators) or (
        ties in {"time", "all"} and np.count_nonzero(final_block & event_indicators) > 1
    )
    if not final_has_comparators:
        contributing &= ~final_block

    censoring_model = KaplanMeier(train_times, train_events, reverse=True)
    survival = censoring_model.predict(event_times[contributing], side="left")
    weights = np.zeros(event_times.shape, dtype=float)
    weights[contributing] = _inverse_censoring_weights(survival)
    return weights


@dataclass
class _ConcordanceCounts:
    """
    Raw concordance pair counts before tie-mode finalization.

    Attributes
    ----------
    concordant: float
        Weighted count of comparable pairs with correctly ordered risks.
    discordant: float
        Weighted count of comparable pairs with incorrectly ordered risks.
    risk_tie_pairs: float
        Weighted count of comparable pairs tied on risk.
    time_tie_pairs: float
        Weighted count of same-time event pairs.
    """

    concordant: float = 0.0
    discordant: float = 0.0
    risk_tie_pairs: float = 0.0
    time_tie_pairs: float = 0.0

    def __iadd__(self, other: _ConcordanceCounts) -> Self:
        """
        Add another count object in place.

        Parameters
        ----------
        other : _ConcordanceCounts
            Concordance counts to add to this object.

        Returns
        -------
        Self
            This count object after mutation.
        """
        self.concordant += other.concordant
        self.discordant += other.discordant
        self.risk_tie_pairs += other.risk_tie_pairs
        self.time_tie_pairs += other.time_tie_pairs
        return self


class _FenwickTree:
    """Maintain weighted prefix sums with logarithmic updates and queries."""

    def __init__(self, size: int) -> None:
        """
        Initialize a weighted prefix-sum tree with the requested number of positions.

        Parameters
        ----------
        size : int
            Number of indexed positions.
        """
        if size <= 0:
            raise ValueError("Fenwick tree size must be positive.")
        self._tree = np.zeros(size + 1, dtype=float)

    def add(self, index: int, value: float) -> None:
        """
        Add ``value`` to one zero-based position.

        Parameters
        ----------
        index : int
            Zero-based position to update.
        value : float
            Amount to add at the selected position.
        """
        position = index + 1
        while position < self._tree.shape[0]:
            self._tree[position] += value
            position += position & -position

    def prefix_sum(self, stop: int) -> float:
        """
        Return the sum over zero-based positions in ``[0, stop)``.

        Parameters
        ----------
        stop : int
            Exclusive zero-based end of the prefix to sum.

        Returns
        -------
        float
            Sum of the stored values at positions smaller than stop.
        """
        total = 0.0
        position = stop
        while position > 0:
            total += self._tree[position]
            position -= position & -position
        return total


def _finalize_counts(
    counts: _ConcordanceCounts,
    ties: str,
) -> tuple[float, float, float]:
    """
    Apply the requested tie policy to raw concordance counts.

    Parameters
    ----------
    counts : _ConcordanceCounts
        Raw concordance pair counts.
    ties : str
        One of ``"None"``, ``"Time"``, ``"Risk"``, or ``"All"``.

    Returns
    -------
    c_index : float
        The concordance index after applying the requested tie policy.
    concordant_pairs : float
        The concordant-pair count after applying the requested tie policy.
    total_pairs : float
        The total pair count after applying the requested tie policy.
    """
    ties = _normalize_ties(ties)

    concordant_pairs = counts.concordant
    if ties == "none":
        total_pairs = counts.concordant + counts.discordant
    elif ties == "time":
        total_pairs = counts.concordant + counts.discordant + counts.time_tie_pairs
        concordant_pairs += 0.5 * counts.time_tie_pairs
    elif ties == "risk":
        total_pairs = counts.concordant + counts.discordant + counts.risk_tie_pairs
        concordant_pairs += 0.5 * counts.risk_tie_pairs
    elif ties == "all":
        total_pairs = (
            counts.concordant
            + counts.discordant
            + counts.risk_tie_pairs
            + counts.time_tie_pairs
        )
        concordant_pairs += 0.5 * (counts.risk_tie_pairs + counts.time_tie_pairs)
    c_index = concordant_pairs / total_pairs if total_pairs != 0 else float("nan")
    return c_index, concordant_pairs, total_pairs


def _normalize_ties(ties: str) -> str:
    """
    Normalize and validate a concordance tie policy.

    Parameters
    ----------
    ties : str
        Tie policy: "None", "Time", "Risk", or "All", case-insensitive.

    Returns
    -------
    str
        Validated lowercase tie policy.
    """
    normalized = ties.lower()
    if normalized not in {"none", "time", "risk", "all"}:
        raise ValueError(
            "Please enter one of 'None', 'Time', 'Risk', or 'All' for "
            "handling ties for concordance."
        )
    return normalized


def _check_has_any_pairs(counts: _ConcordanceCounts) -> None:
    """
    Raise when no comparable or tied-time pairs were counted.

    Parameters
    ----------
    counts : _ConcordanceCounts
        Raw concordance pair counts.

    Raises
    ------
    ValueError
        If there are no pairs available to estimate concordance.
    """
    if (
        counts.concordant
        + counts.discordant
        + counts.risk_tie_pairs
        + counts.time_tie_pairs
        == 0
    ):
        raise ValueError(
            "Data has no comparable pairs, cannot estimate concordance index."
        )


def _is_before_tau(times: np.ndarray, tau: float | None) -> np.ndarray:
    """
    Return a boolean mask for times that pass the strict tau truncation.

    Parameters
    ----------
    times : ndarray
        Non-negative query times.
    tau : float | None
        Finite, non-negative truncation time; None leaves times untruncated.

    Returns
    -------
    ndarray
        Boolean mask selecting times strictly before tau, or all times when tau is None.
    """
    if tau is None:
        return np.ones(times.shape, dtype=bool)
    return times < tau


def _count_directed_risk_pairs(
    anchor_indices: np.ndarray,
    candidate_indices: np.ndarray,
    estimate: np.ndarray,
    pair_weights: np.ndarray,
    tied_tol: float = 1e-8,
) -> _ConcordanceCounts:
    """
    Count concordance outcomes for directed risk-score pairs.

    Parameters
    ----------
    anchor_indices : ndarray, shape = (n_pairs,)
        Sample indices for the earlier or anchor side of each pair.
    candidate_indices : ndarray, shape = (n_pairs,)
        Sample indices for the later or candidate side of each pair.
    estimate : ndarray, shape = (n_samples,)
        Risk scores. Higher scores indicate higher risk.
    pair_weights : ndarray, shape = (n_pairs,)
        Weight for each directed pair.
    tied_tol : float, optional, default=1e-8
        Absolute tolerance for risk-score ties.

    Returns
    -------
    _ConcordanceCounts
        Raw concordance counts for the supplied directed pairs.
    """
    if anchor_indices.shape[0] == 0:
        return _ConcordanceCounts()

    risk_diff = estimate[candidate_indices] - estimate[anchor_indices]
    tied = np.absolute(risk_diff) <= tied_tol
    concordant = (risk_diff < 0) & ~tied

    concordant_weight = pair_weights[concordant].sum()
    risk_tie_weight = pair_weights[tied].sum()
    return _ConcordanceCounts(
        concordant=concordant_weight,
        discordant=pair_weights.sum() - concordant_weight - risk_tie_weight,
        risk_tie_pairs=risk_tie_weight,
    )


def _iter_time_blocks(
    event_time: np.ndarray,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """
    Yield equal-time sample blocks and the samples with later observed times.

    Parameters
    ----------
    event_time : ndarray, shape = (n_samples,)
        Observed event or censoring times.

    Yields
    ------
    block : ndarray
        Sample indices with the same observed time.
    later_samples : ndarray
        Sample indices whose observed time is greater than the block time.
    """
    order = np.argsort(event_time, kind="stable")
    n_samples = order.shape[0]

    start = 0
    while start < n_samples:
        end = start + 1
        time_i = event_time[order[start]]
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        yield order[start:end], order[end:]
        start = end


def _same_time_pair_weight(sample_weights: np.ndarray) -> float:
    """
    Return the total pair weight within one same-time event block.

    Parameters
    ----------
    sample_weights : ndarray
        Per-sample weights for samples in the same event-time block.

    Returns
    -------
    float
        The sum of pairwise weight products for all unique pairs in the block.
    """
    if sample_weights.shape[0] < 2:
        return 0.0
    total_weight = sample_weights.sum()
    return 0.5 * (total_weight * total_weight - np.dot(sample_weights, sample_weights))
