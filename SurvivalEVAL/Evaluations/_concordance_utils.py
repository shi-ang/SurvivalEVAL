from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass
class _ConcordanceCounts:
    """Raw concordance pair counts before tie-mode finalization.

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

    def __iadd__(self, other: "_ConcordanceCounts") -> "_ConcordanceCounts":
        """Add another count object in place.

        Parameters
        ----------
        other: _ConcordanceCounts
            Concordance counts to add to this object.

        Returns
        -------
        _ConcordanceCounts
            This count object after mutation.
        """
        self.concordant += other.concordant
        self.discordant += other.discordant
        self.risk_tie_pairs += other.risk_tie_pairs
        self.time_tie_pairs += other.time_tie_pairs
        return self


def _finalize_counts(
    counts: _ConcordanceCounts, ties: str
) -> tuple[float, float, float]:
    """Apply the requested tie policy to raw concordance counts.

    Parameters
    ----------
    counts: _ConcordanceCounts
        Raw concordance pair counts.
    ties: str
        One of ``"None"``, ``"Time"``, ``"Risk"``, or ``"All"``.

    Returns
    -------
    c_index: float
        The concordance index after applying the requested tie policy.
    concordant_pairs: float
        The concordant-pair count after applying the requested tie policy.
    total_pairs: float
        The total pair count after applying the requested tie policy.
    """
    ties = ties.lower()

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
    else:
        error = "Please enter one of 'None', 'Time', 'Risk', or 'All' for handling ties for concordance."
        raise ValueError(error)

    c_index = concordant_pairs / total_pairs if total_pairs != 0 else float("nan")
    return c_index, concordant_pairs, total_pairs


def _check_has_any_pairs(counts: _ConcordanceCounts) -> None:
    """Raise when no comparable or tied-time pairs were counted.

    Parameters
    ----------
    counts: _ConcordanceCounts
        Raw concordance pair counts.

    Returns
    -------
    None

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


def _is_before_tau(times: np.ndarray, tau: Optional[float]) -> np.ndarray:
    """Return a boolean mask for times that pass the strict tau truncation."""
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
    """Count concordance outcomes for directed risk-score pairs.

    Parameters
    ----------
    anchor_indices: np.ndarray, shape = (n_pairs,)
        Sample indices for the earlier or anchor side of each pair.
    candidate_indices: np.ndarray, shape = (n_pairs,)
        Sample indices for the later or candidate side of each pair.
    estimate: np.ndarray, shape = (n_samples,)
        Risk scores. Higher scores indicate higher risk.
    pair_weights: np.ndarray, shape = (n_pairs,)
        Weight for each directed pair.
    tied_tol: float, optional (default=1e-8)
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
    """Yield equal-time sample blocks and the samples with later observed times.

    Parameters
    ----------
    event_time: np.ndarray, shape = (n_samples,)
        Observed event or censoring times.

    Yields
    ------
    block: np.ndarray
        Sample indices with the same observed time.
    later_samples: np.ndarray
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
    """Return the total pair weight within one same-time event block.

    Parameters
    ----------
    sample_weights: np.ndarray
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
