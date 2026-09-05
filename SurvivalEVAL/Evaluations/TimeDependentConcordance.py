from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from SurvivalEVAL.Evaluations._concordance_utils import (
    _check_has_any_pairs,
    _ConcordanceCounts,
    _finalize_counts,
    _normalize_ties,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier

_DEFAULT_WORKING_MEMORY = 256.0
_MEBIBYTE = 1 << 20
_RiskBatchFiller = Callable[[np.ndarray, np.ndarray, np.ndarray], None]


@dataclass(frozen=True)
class _EventBlockLayout:
    """The single ordered representation used by all concordance stages."""

    order: np.ndarray
    event_starts: np.ndarray
    event_counts: np.ndarray
    candidate_starts: np.ndarray
    event_times: np.ndarray
    risk_block_indices: np.ndarray
    risk_times: np.ndarray
    risk_event_starts: np.ndarray
    risk_event_counts: np.ndarray
    risk_candidate_starts: np.ndarray
    subject_risk_prefixes: np.ndarray


@dataclass(frozen=True)
class _PreparedConcordance:
    """Validated inputs and weights associated with one event-block layout."""

    event_times: np.ndarray
    event_indicators: np.ndarray
    method: str
    ties: str
    working_bytes: int
    layout: _EventBlockLayout
    block_weights: np.ndarray


def _as_array(values: object, name: str) -> np.ndarray:
    """Convert an array-like input without silently accepting malformed data."""
    try:
        return np.asarray(values)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an array of real numeric values.") from error


def _is_real_numeric_array(values: np.ndarray, *, allow_bool: bool = False) -> bool:
    """Return whether an array has a supported, non-complex numeric dtype."""
    if allow_bool and np.issubdtype(values.dtype, np.bool_):
        return True
    return np.issubdtype(values.dtype, np.integer) or np.issubdtype(
        values.dtype, np.floating
    )


def _is_real_numeric_scalar(value: object) -> bool:
    """Return whether a value is a non-Boolean real numeric scalar."""
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return False
        value = value.item()
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, float, np.integer, np.floating)
    )


def _contains_nonfinite(values: np.ndarray) -> bool:
    """Check finiteness without allocating an array-shaped Boolean mask."""
    if values.size == 0 or not np.issubdtype(values.dtype, np.floating):
        return False
    return not np.isfinite(values.min()) or not np.isfinite(values.max())


def _validate_time_array(
    values: object,
    name: str,
    *,
    allow_empty: bool,
) -> np.ndarray:
    """Validate a one-dimensional array of finite, non-negative times."""
    array = _as_array(values, name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array.")
    if not allow_empty and array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not _is_real_numeric_array(array):
        raise TypeError(f"{name} must contain real numeric values.")
    if _contains_nonfinite(array):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array < 0):
        raise ValueError(f"{name} must contain only non-negative values.")
    return array


def _validate_event_data(
    event_times: object,
    event_indicators: object,
    *,
    times_name: str = "event_times",
    indicators_name: str = "event_indicators",
) -> tuple[np.ndarray, np.ndarray]:
    """Validate survival outcomes and return Boolean event indicators."""
    times = _validate_time_array(event_times, times_name, allow_empty=False)
    indicators = _as_array(event_indicators, indicators_name)
    if indicators.ndim != 1:
        raise ValueError(f"{indicators_name} must be a 1-D array.")
    if times.shape != indicators.shape:
        raise ValueError(
            f"{times_name} and {indicators_name} must have the same shape."
        )
    if not _is_real_numeric_array(indicators, allow_bool=True):
        raise TypeError(
            f"{indicators_name} must contain Boolean or real numeric values."
        )
    if not np.issubdtype(indicators.dtype, np.bool_) and not np.all(
        (indicators == 0) | (indicators == 1)
    ):
        raise ValueError(f"{indicators_name} must contain only 0 and 1.")
    return times, indicators.astype(bool, copy=False)


def _validate_tau(tau: object | None) -> object | None:
    """Validate an optional concordance truncation time without narrowing it."""
    if tau is None:
        return None
    if not _is_real_numeric_scalar(tau):
        raise TypeError("tau must be a non-negative finite numeric scalar.")
    scalar = tau.item() if isinstance(tau, np.ndarray) else tau
    if scalar < 0:
        raise ValueError("tau must be a non-negative finite numeric scalar.")
    if isinstance(scalar, (float, np.floating)) and not np.isfinite(scalar):
        raise ValueError("tau must be a non-negative finite numeric scalar.")
    return scalar


def _validate_working_memory(working_memory: object) -> int:
    """Validate a MiB risk-buffer target and return its byte count."""
    if not _is_real_numeric_scalar(working_memory):
        raise ValueError("working_memory must be a positive finite number of MiB.")
    scalar = (
        working_memory.item()
        if isinstance(working_memory, np.ndarray)
        else working_memory
    )
    try:
        memory_mib = float(scalar)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "working_memory must be a positive finite number of MiB."
        ) from error
    if not np.isfinite(memory_mib) or memory_mib <= 0:
        raise ValueError("working_memory must be a positive finite number of MiB.")
    memory_bytes = memory_mib * _MEBIBYTE
    if not np.isfinite(memory_bytes):
        raise ValueError("working_memory is too large.")
    return max(1, int(memory_bytes))


def _normalize_time_dependent_method(method: str) -> str:
    """Normalize and validate a time-dependent concordance method."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    normalized = method.lower()
    if normalized not in {"antolini", "naive", "ipcw"}:
        raise ValueError(
            f"Unsupported method: {normalized}. Supported methods are "
            "'Antolini', 'Naive', and 'IPCW'."
        )
    return normalized


def _numpy_comparison_is_exact(*values: np.ndarray) -> bool:
    """Return whether NumPy promotion preserves all supplied numeric values."""
    try:
        comparison_dtype = np.result_type(*(value.dtype for value in values))
    except TypeError:
        return False
    if np.issubdtype(comparison_dtype, np.integer):
        return True
    if not np.issubdtype(comparison_dtype, np.floating):
        return False

    integer_limit = 1 << (np.finfo(comparison_dtype).nmant + 1)
    for value in values:
        if not np.issubdtype(value.dtype, np.integer) or value.size == 0:
            continue
        if int(value.min()) < -integer_limit or int(value.max()) > integer_limit:
            return False
    return True


def _comparison_scalar(value: object, dtype: np.dtype) -> int | float:
    """Convert a NumPy value to a Python scalar for exact mixed comparisons."""
    if np.issubdtype(dtype, np.integer) or isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _searchsorted_exact(
    sorted_values: np.ndarray,
    query_values: np.ndarray,
    *,
    side: str,
) -> np.ndarray:
    """Search sorted coordinates exactly when NumPy promotion would be lossy."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    queries = np.atleast_1d(query_values)
    if _numpy_comparison_is_exact(sorted_values, queries):
        return np.searchsorted(sorted_values, queries, side=side)

    positions = np.empty(queries.size, dtype=np.intp)
    for query_index, query in enumerate(queries):
        query_scalar = _comparison_scalar(query, queries.dtype)
        start = 0
        stop = sorted_values.size
        while start < stop:
            middle = (start + stop) // 2
            middle_scalar = _comparison_scalar(
                sorted_values[middle], sorted_values.dtype
            )
            move_right = (
                middle_scalar <= query_scalar
                if side == "right"
                else middle_scalar < query_scalar
            )
            if move_right:
                start = middle + 1
            else:
                stop = middle
        positions[query_index] = start
    return positions.reshape(queries.shape)


def _scalar_query(value: object) -> np.ndarray:
    """Wrap a scalar for exact searching, retaining arbitrarily large integers."""
    try:
        return np.asarray([value])
    except OverflowError:
        return np.asarray([value], dtype=object)


def _build_event_block_layout(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    tau: object | None,
) -> _EventBlockLayout:
    """Sort once and describe all active event blocks and risk prefixes."""
    order = np.lexsort((~event_indicators, event_times))
    ordered_times = event_times[order]
    ordered_indicators = event_indicators[order]
    block_starts = np.concatenate(
        (
            np.array([0], dtype=np.intp),
            np.flatnonzero(ordered_times[1:] != ordered_times[:-1]) + 1,
        )
    )
    block_event_counts = np.add.reduceat(ordered_indicators, block_starts).astype(
        np.intp, copy=False
    )
    block_times = ordered_times[block_starts]

    tau_stop = block_starts.size
    if tau is not None:
        tau_stop = int(
            _searchsorted_exact(
                block_times,
                _scalar_query(tau),
                side="left",
            )[0]
        )
    active = np.flatnonzero(block_event_counts[:tau_stop] > 0)
    event_starts = block_starts[active]
    event_counts = block_event_counts[active]
    candidate_starts = event_starts + event_counts
    active_event_times = block_times[active]

    risk_block_indices = np.flatnonzero(candidate_starts < order.size)
    risk_event_starts = event_starts[risk_block_indices]
    risk_event_counts = event_counts[risk_block_indices]
    risk_candidate_starts = candidate_starts[risk_block_indices]
    risk_times = active_event_times[risk_block_indices]

    subject_risk_prefixes = np.zeros(order.size, dtype=np.intp)
    if risk_block_indices.size:
        subject_risk_prefixes[risk_candidate_starts] = 1
        np.cumsum(subject_risk_prefixes, out=subject_risk_prefixes)
        for risk_column, (event_start, event_count) in enumerate(
            zip(risk_event_starts, risk_event_counts)
        ):
            subject_risk_prefixes[event_start : event_start + event_count] = (
                risk_column + 1
            )

    return _EventBlockLayout(
        order=order,
        event_starts=event_starts,
        event_counts=event_counts,
        candidate_starts=candidate_starts,
        event_times=active_event_times,
        risk_block_indices=risk_block_indices,
        risk_times=risk_times,
        risk_event_starts=risk_event_starts,
        risk_event_counts=risk_event_counts,
        risk_candidate_starts=risk_candidate_starts,
        subject_risk_prefixes=subject_risk_prefixes,
    )


def _ipcw_block_weights(
    layout: _EventBlockLayout,
    train_event_times: object | None,
    train_event_indicators: object | None,
    ties: str,
) -> np.ndarray:
    """Calculate one right-continuous IPCW weight per active event block."""
    if train_event_times is None or train_event_indicators is None:
        raise ValueError(
            "train_event_times and train_event_indicators must be provided "
            "for IPCW method."
        )
    train_times, train_indicators = _validate_event_data(
        train_event_times,
        train_event_indicators,
        times_name="train_event_times",
        indicators_name="train_event_indicators",
    )

    weights = np.ones(layout.event_times.size, dtype=float)
    needs_weight = np.zeros(layout.event_times.size, dtype=bool)
    needs_weight[layout.risk_block_indices] = True
    if ties in {"time", "all"}:
        needs_weight |= layout.event_counts > 1
    contributing_times = layout.event_times[needs_weight]
    if contributing_times.size == 0:
        return weights

    support = np.asarray([train_times.max()])
    beyond_support = _searchsorted_exact(
        support,
        contributing_times,
        side="left",
    )
    if np.any(beyond_support == support.size):
        raise ValueError(
            "Contributing event times must not exceed the largest training "
            "time; choose a smaller tau."
        )

    censoring_model = KaplanMeier(train_times, ~train_indicators)
    survival_indices = (
        _searchsorted_exact(
            censoring_model.survival_times,
            contributing_times,
            side="right",
        )
        - 1
    )
    censoring_survival = censoring_model.survival_probabilities[survival_indices]
    if np.any(censoring_survival <= 0):
        raise ValueError(
            "Censoring survival probability is zero for at least one observed "
            "event; choose a smaller tau."
        )
    weights[needs_weight] = 1.0 / np.square(censoring_survival)
    return weights


def _prepare_concordance(
    event_times: object,
    event_indicators: object,
    train_event_times: object | None,
    train_event_indicators: object | None,
    method: str,
    ties: str,
    tau: object | None,
    working_memory: object,
) -> _PreparedConcordance:
    """Validate shared inputs and construct the sole event-block layout."""
    normalized_method = _normalize_time_dependent_method(method)
    if not isinstance(ties, str):
        raise TypeError("ties must be a string.")
    normalized_ties = _normalize_ties(ties)
    validated_tau = _validate_tau(tau)
    working_bytes = _validate_working_memory(working_memory)
    times, indicators = _validate_event_data(event_times, event_indicators)
    if not np.any(indicators):
        raise ValueError(
            "Data has no observed events, cannot estimate time-dependent "
            "concordance index."
        )

    layout = _build_event_block_layout(times, indicators, validated_tau)
    if normalized_method == "ipcw":
        block_weights = _ipcw_block_weights(
            layout,
            train_event_times,
            train_event_indicators,
            normalized_ties,
        )
    else:
        block_weights = np.ones(layout.event_times.size, dtype=float)

    return _PreparedConcordance(
        event_times=times,
        event_indicators=indicators,
        method=normalized_method,
        ties=normalized_ties,
        working_bytes=working_bytes,
        layout=layout,
        block_weights=block_weights,
    )


def _required_risk_columns(
    risk_times: np.ndarray,
    required_times: np.ndarray,
) -> np.ndarray:
    """Locate every required event time on an explicit risk-time grid."""
    left = _searchsorted_exact(risk_times, required_times, side="left")
    right = _searchsorted_exact(risk_times, required_times, side="right")
    matched = right > left
    if not np.all(matched):
        missing = required_times[~matched]
        raise ValueError(
            "risk_times must include every contributing event time; missing "
            f"{missing.tolist()}."
        )
    return left


def _fill_matrix_risk_batch(
    risk_scores: np.ndarray,
    required_columns: np.ndarray,
    sample_indices: np.ndarray,
    risk_prefixes: np.ndarray,
    output: np.ndarray,
) -> None:
    """Copy compact matrix risks without a second batch-sized allocation."""
    for output_column, source_column in enumerate(required_columns[: output.shape[1]]):
        first_row = int(np.searchsorted(risk_prefixes, output_column, side="right"))
        np.take(
            risk_scores[:, source_column],
            sample_indices[first_row:],
            out=output[first_row:, output_column],
            mode="clip",
        )


def _validate_matrix_risk_source(
    risk_scores: object,
    risk_times: object,
    prepared: _PreparedConcordance,
) -> tuple[np.dtype, _RiskBatchFiller]:
    """Validate matrix-specific inputs and return a compact batch filler."""
    scores = _as_array(risk_scores, "risk_scores")
    coordinates = _validate_time_array(risk_times, "risk_times", allow_empty=True)
    if scores.ndim != 2:
        raise ValueError(
            "risk_scores should be a 2D array of shape "
            f"(n_samples, n_risk_times), but got shape {scores.shape}."
        )
    if scores.shape[0] != prepared.event_times.size:
        raise ValueError(
            "The lengths of risk_scores, event_times, and event_indicators "
            "must be the same."
        )
    if scores.shape[1] != coordinates.size:
        raise ValueError(
            "The number of columns in risk_scores must match the length of risk_times."
        )
    if not _is_real_numeric_array(scores):
        raise TypeError("risk_scores must contain real numeric values.")
    if _contains_nonfinite(scores):
        raise ValueError("risk_scores must contain only finite values.")
    if coordinates.size > 1 and np.any(coordinates[1:] <= coordinates[:-1]):
        raise ValueError("risk_times must be strictly increasing and unique.")

    required_columns = _required_risk_columns(
        coordinates,
        prepared.layout.risk_times,
    )
    return scores.dtype, partial(
        _fill_matrix_risk_batch,
        scores,
        required_columns,
    )


def _validate_tied_tolerance(tied_tol: object) -> float:
    """Validate the non-negative finite risk-tie tolerance."""
    if not _is_real_numeric_scalar(tied_tol):
        raise ValueError("tied_tol must be a non-negative finite number.")
    try:
        tolerance = float(tied_tol)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("tied_tol must be a non-negative finite number.") from error
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tied_tol must be a non-negative finite number.")
    return tolerance


def _count_streamed_risks(
    prepared: _PreparedConcordance,
    risk_dtype: np.dtype,
    fill_risk_batch: _RiskBatchFiller,
    *,
    tied_tol: object = 1e-8,
) -> _ConcordanceCounts:
    """Count each comparable directed pair once using a bounded risk buffer."""
    tolerance = _validate_tied_tolerance(tied_tol)
    layout = prepared.layout
    tie_pair_counts = 0.5 * layout.event_counts * (layout.event_counts - 1)
    counts = _ConcordanceCounts(
        time_tie_pairs=float(np.dot(tie_pair_counts, prepared.block_weights))
    )
    n_risk_times = layout.risk_times.size
    if n_risk_times == 0:
        return counts

    dtype = np.dtype(risk_dtype)
    bytes_per_row = max(1, n_risk_times * dtype.itemsize)
    batch_size = min(
        layout.order.size,
        max(1, prepared.working_bytes // bytes_per_row),
    )
    risk_buffer = np.empty((batch_size, n_risk_times), dtype=dtype, order="F")
    selected = np.empty(batch_size, dtype=bool)

    integer_risks = np.issubdtype(dtype, np.integer)
    if integer_risks:
        integer_limits = np.iinfo(dtype)
        integer_tolerance = int(np.floor(tolerance))
        comparison_buffer = None
    else:
        comparison_dtype = np.result_type(dtype, np.float64)
        comparison_buffer = np.empty(batch_size, dtype=comparison_dtype)

    anchor_offsets = np.concatenate(
        (
            np.array([0], dtype=np.intp),
            np.cumsum(layout.risk_event_counts, dtype=np.intp),
        )
    )
    anchor_risks = np.empty(anchor_offsets[-1], dtype=dtype)
    next_anchor_block = 0

    for batch_start in range(0, layout.order.size, batch_size):
        batch_stop = min(batch_start + batch_size, layout.order.size)
        current_size = batch_stop - batch_start
        prefixes = layout.subject_risk_prefixes[batch_start:batch_stop]
        active_columns = int(prefixes[-1])
        if active_columns == 0:
            continue

        current_risks = risk_buffer[:current_size, :active_columns]
        fill_risk_batch(
            layout.order[batch_start:batch_stop],
            prefixes,
            current_risks,
        )

        while (
            next_anchor_block < n_risk_times
            and layout.risk_event_starts[next_anchor_block] < batch_stop
        ):
            event_start = int(layout.risk_event_starts[next_anchor_block])
            event_stop = event_start + int(layout.risk_event_counts[next_anchor_block])
            overlap_start = max(event_start, batch_start)
            overlap_stop = min(event_stop, batch_stop)
            if overlap_start < overlap_stop:
                destination_start = int(anchor_offsets[next_anchor_block]) + (
                    overlap_start - event_start
                )
                destination_stop = destination_start + overlap_stop - overlap_start
                anchor_risks[destination_start:destination_stop] = current_risks[
                    overlap_start - batch_start : overlap_stop - batch_start,
                    next_anchor_block,
                ]
            if event_stop <= batch_stop:
                next_anchor_block += 1
            else:
                break

        active_blocks = int(
            np.searchsorted(layout.risk_candidate_starts, batch_stop, side="left")
        )
        for risk_column in range(active_blocks):
            first_candidate = max(
                int(layout.risk_candidate_starts[risk_column]),
                batch_start,
            )
            candidate_risks = current_risks[
                first_candidate - batch_start :, risk_column
            ]
            n_candidates = candidate_risks.size
            block_selected = selected[:n_candidates]
            block_weight = float(
                prepared.block_weights[layout.risk_block_indices[risk_column]]
            )
            anchors = anchor_risks[
                anchor_offsets[risk_column] : anchor_offsets[risk_column + 1]
            ]

            for anchor_risk_value in anchors:
                if integer_risks:
                    anchor_risk = int(anchor_risk_value)
                    lower_tie_bound = dtype.type(
                        max(integer_limits.min, anchor_risk - integer_tolerance)
                    )
                    upper_tie_bound = dtype.type(
                        min(integer_limits.max, anchor_risk + integer_tolerance)
                    )
                    np.less(candidate_risks, lower_tie_bound, out=block_selected)
                    concordant = float(np.count_nonzero(block_selected))
                    np.less_equal(
                        candidate_risks,
                        upper_tie_bound,
                        out=block_selected,
                    )
                else:
                    difference = comparison_buffer[:n_candidates]
                    with np.errstate(over="ignore", invalid="ignore"):
                        np.subtract(candidate_risks, anchor_risk_value, out=difference)
                    np.less(difference, -tolerance, out=block_selected)
                    concordant = float(np.count_nonzero(block_selected))
                    np.less_equal(difference, tolerance, out=block_selected)

                through_ties = float(np.count_nonzero(block_selected))
                counts.concordant += block_weight * concordant
                counts.risk_tie_pairs += block_weight * (through_ties - concordant)
                counts.discordant += block_weight * (n_candidates - through_ties)

    return counts


def _concordance_from_risk_batches(
    prepared: _PreparedConcordance,
    risk_dtype: np.dtype,
    fill_risk_batch: _RiskBatchFiller,
) -> tuple[float, float, float]:
    """Count and finalize concordance for an already prepared risk source."""
    counts = _count_streamed_risks(prepared, risk_dtype, fill_risk_batch)
    _check_has_any_pairs(counts)
    return _finalize_counts(counts, prepared.ties)


def concordance_time_dependent(
    risk_scores: object,
    risk_times: object,
    event_times: object,
    event_indicators: object,
    train_event_times: object | None = None,
    train_event_indicators: object | None = None,
    method: str = "Antolini",
    ties: str = "Risk",
    tau: object | None = None,
    working_memory: object = _DEFAULT_WORKING_MEMORY,
) -> tuple[float, float, float]:
    """Calculate exact, memory-bounded time-dependent concordance.

    Parameters
    ----------
    risk_scores : array-like, shape = (n_samples, n_risk_times)
        Finite risk scores at ``risk_times``. Higher values indicate greater
        event risk.
    risk_times : array-like, shape = (n_risk_times,)
        Finite, non-negative, strictly increasing unique coordinates. Every
        event time with a comparable candidate must be present. Repeated event
        times share one column; unused extra coordinates are allowed.
    event_times : array-like, shape = (n_samples,)
        Finite, non-negative observed event or censoring times.
    event_indicators : array-like, shape = (n_samples,)
        Binary indicators where 1 denotes an observed event.
    train_event_times : array-like, optional
        Training event or censoring times, required for ``"IPCW"``.
    train_event_indicators : array-like, optional
        Binary training indicators, required for ``"IPCW"``.
    method : str, default="Antolini"
        One of ``"Antolini"``, ``"Naive"``, or ``"IPCW"``.
    ties : str, default="Risk"
        One of ``"None"``, ``"Time"``, ``"Risk"``, or ``"All"``.
    tau : real scalar, optional
        Count only event anchors strictly before this non-negative finite time.
    working_memory : real scalar, default=256
        Target MiB for the compact risk buffer. If one complete compact row is
        larger than the target, one row is still allocated.

    Returns
    -------
    tuple[float, float, float]
        Concordance index, concordant-pair mass, and total-pair mass.

    Notes
    -----
    The calculation takes ``O(n log n + R + P)`` time and
    ``O(n + Bm + B)`` auxiliary space. ``P`` is the number of comparable
    directed pairs, ``R`` the number of streamed risk cells, ``m`` the number
    of contributing event times, and ``B`` the memory-budgeted batch size.
    """
    prepared = _prepare_concordance(
        event_times,
        event_indicators,
        train_event_times,
        train_event_indicators,
        method,
        ties,
        tau,
        working_memory,
    )
    risk_dtype, filler = _validate_matrix_risk_source(
        risk_scores,
        risk_times,
        prepared,
    )
    return _concordance_from_risk_batches(prepared, risk_dtype, filler)
