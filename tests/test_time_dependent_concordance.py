import itertools
import tracemalloc

import numpy as np
import pytest

from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.Evaluations import TimeDependentConcordance as td_concordance
from SurvivalEVAL.Evaluations.TimeDependentConcordance import (
    concordance_time_dependent,
)


def _python_number(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    return float(value)


def _oracle_censoring_survival(
    time,
    train_event_times,
    train_event_indicators,
):
    query = _python_number(time)
    train_times = [_python_number(value) for value in train_event_times]
    if query > max(train_times):
        raise ValueError("outside training support")

    survival = 1.0
    for jump_time in sorted(set(train_times)):
        if jump_time > query:
            break
        at_risk = sum(value >= jump_time for value in train_times)
        censorings = sum(
            value == jump_time and not bool(indicator)
            for value, indicator in zip(train_times, train_event_indicators)
        )
        survival *= 1.0 - censorings / at_risk
    if survival <= 0:
        raise ValueError("zero censoring survival")
    return survival


def _oracle_time_dependent_concordance(
    risk_scores,
    risk_times,
    event_times,
    event_indicators,
    *,
    method="Antolini",
    ties="Risk",
    tau=None,
    train_event_times=None,
    train_event_indicators=None,
    tied_tol=1e-8,
):
    """Independent pair-enumeration oracle with its own IPCW estimator."""
    method = method.lower()
    ties = ties.lower()
    times = [_python_number(value) for value in event_times]
    coordinates = [_python_number(value) for value in risk_times]
    indicators = [bool(value) for value in event_indicators]
    tau_value = None if tau is None else _python_number(tau)
    integer_risks = np.issubdtype(np.asarray(risk_scores).dtype, np.integer)

    concordant = 0.0
    discordant = 0.0
    risk_ties = 0.0
    time_ties = 0.0
    has_raw_pair = False
    weight_cache = {}

    def before_tau(time):
        return tau_value is None or time < tau_value

    def pair_weight(time):
        if method != "ipcw":
            return 1.0
        if time not in weight_cache:
            survival = _oracle_censoring_survival(
                time,
                train_event_times,
                train_event_indicators,
            )
            weight_cache[time] = 1.0 / survival**2
        return weight_cache[time]

    for first in range(len(times)):
        for second in range(first + 1, len(times)):
            if (
                indicators[first]
                and indicators[second]
                and times[first] == times[second]
                and before_tau(times[first])
            ):
                has_raw_pair = True
                weight = pair_weight(times[first]) if ties in {"time", "all"} else 1.0
                time_ties += weight

    for anchor, anchor_time in enumerate(times):
        if not indicators[anchor] or not before_tau(anchor_time):
            continue
        candidates = [
            sample
            for sample, sample_time in enumerate(times)
            if sample != anchor
            and (
                sample_time > anchor_time
                or (sample_time == anchor_time and not indicators[sample])
            )
        ]
        if not candidates:
            continue
        has_raw_pair = True
        try:
            risk_column = next(
                index
                for index, coordinate in enumerate(coordinates)
                if coordinate == anchor_time
            )
        except StopIteration as error:
            raise ValueError("missing risk coordinate") from error
        weight = pair_weight(anchor_time)
        anchor_risk = risk_scores[anchor, risk_column]

        for candidate in candidates:
            candidate_risk = risk_scores[candidate, risk_column]
            if integer_risks:
                difference = int(candidate_risk) - int(anchor_risk)
            else:
                difference = float(candidate_risk) - float(anchor_risk)
            if abs(difference) <= tied_tol:
                risk_ties += weight
            elif difference < 0:
                concordant += weight
            else:
                discordant += weight

    if not has_raw_pair:
        return None
    numerator = concordant
    denominator = concordant + discordant
    if ties in {"risk", "all"}:
        numerator += 0.5 * risk_ties
        denominator += risk_ties
    if ties in {"time", "all"}:
        numerator += 0.5 * time_ties
        denominator += time_ties
    c_index = numerator / denominator if denominator else float("nan")
    return c_index, numerator, denominator


def _assert_result_equal(actual, expected):
    assert expected is not None
    if np.isnan(expected[0]):
        assert np.isnan(actual[0])
        np.testing.assert_allclose(actual[1:], expected[1:])
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_exhaustive_small_datasets_match_independent_oracle():
    risk_times = np.array([0.0, 1.0, 2.0, 4.0])
    train_times = np.array([0.0, 1.0, 2.0, 3.0, 3.0])
    train_indicators = np.array([1, 0, 1, 0, 1])

    for event_times_tuple in itertools.product(range(3), repeat=3):
        event_times = np.asarray(event_times_tuple, dtype=float)
        for indicators_tuple in itertools.product((0, 1), repeat=3):
            if not any(indicators_tuple):
                continue
            event_indicators = np.asarray(indicators_tuple)
            risk_scores = np.fromfunction(
                lambda row, column: (3 * row + 2 * column) % 5,
                (3, risk_times.size),
                dtype=int,
            ).astype(float)

            for tau, ties, method, working_memory in itertools.product(
                (None, 1.0, 2.0),
                ("None", "Time", "Risk", "All"),
                ("Antolini", "Naive", "IPCW"),
                (1e-6, 1.0),
            ):
                kwargs = {
                    "method": method,
                    "ties": ties,
                    "tau": tau,
                    "working_memory": working_memory,
                }
                oracle_kwargs = {
                    "method": method,
                    "ties": ties,
                    "tau": tau,
                }
                if method == "IPCW":
                    kwargs.update(
                        train_event_times=train_times,
                        train_event_indicators=train_indicators,
                    )
                    oracle_kwargs.update(
                        train_event_times=train_times,
                        train_event_indicators=train_indicators,
                    )
                expected = _oracle_time_dependent_concordance(
                    risk_scores,
                    risk_times,
                    event_times,
                    event_indicators,
                    **oracle_kwargs,
                )
                if expected is None:
                    with pytest.raises(ValueError, match="no comparable pairs"):
                        concordance_time_dependent(
                            risk_scores,
                            risk_times,
                            event_times,
                            event_indicators,
                            **kwargs,
                        )
                else:
                    actual = concordance_time_dependent(
                        risk_scores,
                        risk_times,
                        event_times,
                        event_indicators,
                        **kwargs,
                    )
                    _assert_result_equal(actual, expected)


@pytest.mark.parametrize("risk_dtype", [np.int64, np.uint64, np.float32, np.float64])
def test_randomized_permutations_and_batches_match_independent_oracle(risk_dtype):
    rng = np.random.default_rng(20260905)
    train_times = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    train_indicators = np.array([1, 0, 1, 1, 0, 1, 1])

    for _ in range(30):
        n_samples = int(rng.integers(2, 9))
        event_times = rng.integers(0, 6, size=n_samples)
        event_indicators = rng.integers(0, 2, size=n_samples)
        event_indicators[int(rng.integers(n_samples))] = 1
        risk_times = np.arange(0, 6.5, 0.5)
        raw_risks = rng.integers(0, 8, size=(n_samples, risk_times.size))
        if np.issubdtype(risk_dtype, np.floating):
            raw_risks = raw_risks + rng.choice(
                [0.0, 0.5e-8, 1.0e-8, 1.5e-8],
                size=raw_risks.shape,
            )
        risk_scores = raw_risks.astype(risk_dtype)
        permutation = rng.permutation(n_samples)
        event_times = event_times[permutation]
        event_indicators = event_indicators[permutation]
        risk_scores = risk_scores[permutation]
        tau = rng.choice([None, 1, 3, 5])

        for method, ties, working_memory in itertools.product(
            ("Antolini", "Naive", "IPCW"),
            ("None", "Time", "Risk", "All"),
            (1e-6, 0.01),
        ):
            kwargs = {
                "method": method,
                "ties": ties,
                "tau": tau,
                "working_memory": working_memory,
            }
            oracle_kwargs = {"method": method, "ties": ties, "tau": tau}
            if method == "IPCW":
                kwargs.update(
                    train_event_times=train_times,
                    train_event_indicators=train_indicators,
                )
                oracle_kwargs.update(
                    train_event_times=train_times,
                    train_event_indicators=train_indicators,
                )
            expected = _oracle_time_dependent_concordance(
                risk_scores,
                risk_times,
                event_times,
                event_indicators,
                **oracle_kwargs,
            )
            if expected is None:
                with pytest.raises(ValueError, match="no comparable pairs"):
                    concordance_time_dependent(
                        risk_scores,
                        risk_times,
                        event_times,
                        event_indicators,
                        **kwargs,
                    )
            else:
                actual = concordance_time_dependent(
                    risk_scores,
                    risk_times,
                    event_times,
                    event_indicators,
                    **kwargs,
                )
                _assert_result_equal(actual, expected)


def test_repeated_events_share_one_column_and_extra_coordinates_are_ignored():
    event_times = np.array([1.0, 1.0, 2.0, 3.0])
    event_indicators = np.array([1, 1, 0, 1])
    risk_times = np.array([0.5, 1.0, 1.5, 8.0])
    risk_scores = np.array(
        [
            [-99.0, 4.0, 99.0, -99.0],
            [-99.0, 3.0, 99.0, -99.0],
            [-99.0, 2.0, 99.0, -99.0],
            [-99.0, 1.0, 99.0, -99.0],
        ]
    )

    expected = _oracle_time_dependent_concordance(
        risk_scores,
        risk_times,
        event_times,
        event_indicators,
        ties="All",
    )
    actual = concordance_time_dependent(
        risk_scores,
        risk_times,
        event_times,
        event_indicators,
        ties="All",
    )

    _assert_result_equal(actual, expected)
    np.testing.assert_allclose(actual, (0.9, 4.5, 5.0))


@pytest.mark.parametrize("risk_dtype", [np.int64, np.uint64])
def test_integer_risk_ordering_is_exact_at_dtype_boundaries(risk_dtype):
    if risk_dtype == np.int64:
        high = np.iinfo(np.int64).max
        low = np.iinfo(np.int64).min
    else:
        high = np.iinfo(np.uint64).max
        low = 0
    result = concordance_time_dependent(
        np.array([[high], [low]], dtype=risk_dtype),
        np.array([1]),
        np.array([1, 2]),
        np.array([1, 1]),
        working_memory=1e-9,
    )
    np.testing.assert_allclose(result, (1.0, 1.0, 1.0))


@pytest.mark.parametrize("risk_dtype", [np.int64, np.uint64])
def test_integer_risks_above_float_precision_do_not_collapse(risk_dtype):
    boundary = 2**53
    scores = np.array([[boundary + 1], [boundary]], dtype=risk_dtype)
    np.testing.assert_allclose(
        concordance_time_dependent(
            scores,
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.ones(2),
        ),
        (1.0, 1.0, 1.0),
    )


@pytest.mark.parametrize("risk_dtype", [np.float32, np.float64])
def test_floating_risk_tolerance_boundaries_match_oracle(risk_dtype):
    risk_times = np.array([1.0])
    event_times = np.array([1.0, 2.0, 3.0, 4.0])
    indicators = np.array([1, 0, 0, 0])
    scores = np.array([[0.0], [-0.5e-8], [1.0e-8], [1.5e-8]], dtype=risk_dtype)

    actual = concordance_time_dependent(
        scores,
        risk_times,
        event_times,
        indicators,
        ties="Risk",
    )
    expected = _oracle_time_dependent_concordance(
        scores,
        risk_times,
        event_times,
        indicators,
        ties="Risk",
    )
    _assert_result_equal(actual, expected)


def test_finite_float_extremes_are_compared_without_overflow_errors():
    maximum = np.finfo(np.float64).max
    result = concordance_time_dependent(
        np.array([[maximum], [-maximum]]),
        np.array([1.0]),
        np.array([1.0, 2.0]),
        np.ones(2),
    )
    np.testing.assert_allclose(result, (1.0, 1.0, 1.0))


def test_mixed_time_matching_is_exact_around_float64_boundary():
    boundary = 2**53
    with pytest.raises(ValueError, match="missing"):
        concordance_time_dependent(
            np.array([[2.0], [1.0]]),
            np.array([boundary + 1], dtype=np.uint64),
            np.array([boundary, boundary + 2], dtype=np.float64),
            np.array([1, 0]),
        )

    np.testing.assert_allclose(
        concordance_time_dependent(
            np.array([[0.0, 2.0], [3.0, 1.0]]),
            np.array([boundary, boundary + 1], dtype=np.uint64),
            np.array([boundary + 1, boundary + 2], dtype=np.int64),
            np.ones(2),
        ),
        (1.0, 1.0, 1.0),
    )


def test_mixed_time_matching_handles_signed_unsigned_limits():
    boundary = np.iinfo(np.int64).max
    result = concordance_time_dependent(
        np.array([[2], [1]], dtype=np.int8),
        np.array([boundary], dtype=np.int64),
        np.array([boundary, boundary + 1], dtype=np.uint64),
        np.array([1, 0]),
    )
    np.testing.assert_allclose(result, (1.0, 1.0, 1.0))


def test_tau_comparisons_are_exact_at_float32_and_uint64_boundaries():
    float_boundary = 2**24
    np.testing.assert_allclose(
        concordance_time_dependent(
            np.array([[2.0], [1.0]]),
            np.array([float_boundary], dtype=np.float32),
            np.array([float_boundary, float_boundary + 2], dtype=np.float32),
            np.ones(2),
            tau=float_boundary + 1,
        ),
        (1.0, 1.0, 1.0),
    )

    uint_boundary = 2**54
    np.testing.assert_allclose(
        concordance_time_dependent(
            np.array([[2.0], [1.0]]),
            np.array([uint_boundary + 3], dtype=np.uint64),
            np.array([uint_boundary + 3, uint_boundary + 5], dtype=np.uint64),
            np.ones(2),
            tau=np.float64(uint_boundary + 4),
        ),
        (1.0, 1.0, 1.0),
    )


def test_arbitrarily_large_integer_tau_remains_valid():
    np.testing.assert_allclose(
        concordance_time_dependent(
            np.array([[2.0], [1.0]]),
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.ones(2),
            tau=10**400,
        ),
        (1.0, 1.0, 1.0),
    )


def test_ipcw_uses_right_continuous_squared_block_weights():
    risk_scores = np.array(
        [
            [4.0, 0.0, 0.0],
            [3.0, 4.0, 0.0],
            [2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0],
        ]
    )
    risk_times = np.array([1.0, 2.0, 3.0])
    event_times = np.array([1.0, 2.0, 3.0, 4.0])
    indicators = np.ones(4)
    train_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    train_indicators = np.array([1, 0, 1, 0, 1])

    actual = concordance_time_dependent(
        risk_scores,
        risk_times,
        event_times,
        indicators,
        train_times,
        train_indicators,
        method="IPCW",
    )
    expected = _oracle_time_dependent_concordance(
        risk_scores,
        risk_times,
        event_times,
        indicators,
        method="IPCW",
        train_event_times=train_times,
        train_event_indicators=train_indicators,
    )
    _assert_result_equal(actual, expected)


def test_ipcw_censoring_jump_at_anchor_is_applied_immediately():
    kwargs = {
        "risk_scores": np.array([[3.0], [2.0], [1.0]]),
        "risk_times": np.array([2.0]),
        "event_times": np.array([2.0, 3.0, 4.0]),
        "event_indicators": np.array([1, 0, 0]),
        "train_event_times": np.array([1.0, 2.0, 3.0, 4.0]),
        "train_event_indicators": np.array([1, 0, 1, 1]),
        "method": "IPCW",
    }
    result = concordance_time_dependent(**kwargs)
    # G(2) = 2/3 after the censoring jump, so each of two pairs weighs 9/4.
    np.testing.assert_allclose(result, (1.0, 4.5, 4.5))


def test_ipcw_training_support_and_zero_survival_are_checked_only_when_used():
    kwargs = {
        "risk_scores": np.array([[3.0, 0.0], [2.0, 3.0], [1.0, 1.0]]),
        "risk_times": np.array([1.0, 3.0]),
        "event_times": np.array([1.0, 3.0, 4.0]),
        "event_indicators": np.array([1, 1, 0]),
        "train_event_times": np.array([1.0, 2.0]),
        "train_event_indicators": np.array([1, 1]),
        "method": "IPCW",
    }
    with pytest.raises(ValueError, match="largest training time"):
        concordance_time_dependent(**kwargs)
    np.testing.assert_allclose(
        concordance_time_dependent(**kwargs, tau=2.0),
        (1.0, 2.0, 2.0),
    )

    final_ties = {
        "risk_scores": np.array([[3.0], [2.0], [1.0]]),
        "risk_times": np.array([1.0]),
        "event_times": np.array([1.0, 2.0, 2.0]),
        "event_indicators": np.array([1, 1, 1]),
        "train_event_times": np.array([1.0, 2.0]),
        "train_event_indicators": np.array([1, 0]),
        "method": "IPCW",
    }
    np.testing.assert_allclose(
        concordance_time_dependent(**final_ties, ties="Risk"),
        (1.0, 2.0, 2.0),
    )
    with pytest.raises(ValueError, match="zero"):
        concordance_time_dependent(**final_ties, ties="Time")


def test_ipcw_mixed_training_support_and_step_lookup_are_exact():
    boundary = 2**53
    with pytest.raises(ValueError, match="largest training time"):
        concordance_time_dependent(
            np.array([[2.0], [1.0]]),
            np.array([boundary + 1], dtype=np.uint64),
            np.array([boundary + 1, boundary + 2], dtype=np.uint64),
            np.array([1, 0]),
            np.array([boundary], dtype=np.float64),
            np.array([1]),
            method="IPCW",
        )

    actual = concordance_time_dependent(
        np.array([[3.0, 0.0], [2.0, 1.0], [1.0, 2.0]]),
        np.array([boundary, boundary + 1], dtype=np.uint64),
        np.array([boundary, boundary + 1, boundary + 2], dtype=np.uint64),
        np.array([1, 1, 0]),
        np.array([boundary, boundary + 1, boundary + 2], dtype=np.uint64),
        np.array([1, 0, 1]),
        method="IPCW",
    )
    np.testing.assert_allclose(actual, (1 / 3, 2.0, 6.0))


def test_explicit_ipcw_is_honored_for_fully_observed_test_data():
    scores = np.array([[3.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    common = {
        "risk_scores": scores,
        "risk_times": np.array([1.0, 2.0]),
        "event_times": np.array([1.0, 2.0, 3.0]),
        "event_indicators": np.ones(3),
    }
    antolini = concordance_time_dependent(**common)
    ipcw = concordance_time_dependent(
        **common,
        train_event_times=np.array([1.5, 3.0]),
        train_event_indicators=np.array([0, 1]),
        method="IPCW",
    )
    np.testing.assert_allclose(antolini, (2 / 3, 2.0, 3.0))
    np.testing.assert_allclose(ipcw, (1 / 3, 2.0, 6.0))


@pytest.mark.parametrize(
    ("field", "value", "error_type", "message"),
    [
        ("risk_scores", np.ones(2), ValueError, "2D array"),
        ("risk_scores", np.array([[True], [False]]), TypeError, "real numeric"),
        ("risk_scores", np.array([[1j], [0j]]), TypeError, "real numeric"),
        ("risk_scores", np.array([["1"], ["0"]]), TypeError, "real numeric"),
        ("risk_scores", np.array([[np.nan], [0.0]]), ValueError, "finite"),
        ("risk_scores", np.array([[np.inf], [0.0]]), ValueError, "finite"),
        ("risk_times", np.array([[1.0]]), ValueError, "1-D"),
        ("risk_times", np.array([True]), TypeError, "real numeric"),
        ("risk_times", np.array([1j]), TypeError, "real numeric"),
        ("risk_times", np.array(["1"]), TypeError, "real numeric"),
        ("risk_times", np.array([np.nan]), ValueError, "finite"),
        ("risk_times", np.array([-1.0]), ValueError, "non-negative"),
        ("event_times", np.array([[1.0, 2.0]]), ValueError, "1-D"),
        ("event_times", np.array([True, False]), TypeError, "real numeric"),
        ("event_times", np.array([1j, 2j]), TypeError, "real numeric"),
        ("event_times", np.array(["1", "2"]), TypeError, "real numeric"),
        ("event_times", np.array([np.nan, 2.0]), ValueError, "finite"),
        ("event_times", np.array([-1.0, 2.0]), ValueError, "non-negative"),
        ("event_indicators", np.array([[1, 0]]), ValueError, "1-D"),
        ("event_indicators", np.array([1j, 0j]), TypeError, "Boolean or real"),
        ("event_indicators", np.array(["1", "0"]), TypeError, "Boolean or real"),
        ("event_indicators", np.array([2, 0]), ValueError, "only 0 and 1"),
        ("event_indicators", np.array([np.nan, 0]), ValueError, "only 0 and 1"),
    ],
)
def test_public_api_rejects_malformed_and_nonfinite_inputs(
    field,
    value,
    error_type,
    message,
):
    kwargs = {
        "risk_scores": np.array([[2.0], [1.0]]),
        "risk_times": np.array([1.0]),
        "event_times": np.array([1.0, 2.0]),
        "event_indicators": np.array([1, 0]),
    }
    kwargs[field] = value
    with pytest.raises(error_type, match=message):
        concordance_time_dependent(**kwargs)


def test_public_api_rejects_shape_grid_and_event_errors():
    with pytest.raises(ValueError, match="same"):
        concordance_time_dependent(
            np.ones((2, 1)),
            np.array([1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.ones(3),
        )
    with pytest.raises(ValueError, match="columns"):
        concordance_time_dependent(
            np.ones((2, 2)),
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.ones(2),
        )
    with pytest.raises(ValueError, match="missing"):
        concordance_time_dependent(
            np.ones((3, 1)),
            np.array([1.0]),
            np.array([1.0, 2.0, 3.0]),
            np.ones(3),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        concordance_time_dependent(
            np.ones((3, 3)),
            np.array([1, 2, 0], dtype=np.uint64),
            np.array([1, 2, 3], dtype=np.uint64),
            np.ones(3),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        concordance_time_dependent(
            np.ones((2, 2)),
            np.array([1.0, 1.0]),
            np.array([1.0, 2.0]),
            np.ones(2),
        )
    with pytest.raises(ValueError, match="no observed events"):
        concordance_time_dependent(
            np.empty((2, 0)),
            np.empty(0),
            np.array([1.0, 2.0]),
            np.zeros(2),
        )


@pytest.mark.parametrize(
    ("tau", "error_type"),
    [
        (-1, ValueError),
        (np.inf, ValueError),
        (np.nan, ValueError),
        (True, TypeError),
        ("2", TypeError),
        (1 + 0j, TypeError),
        (np.array([2.0]), TypeError),
    ],
)
def test_tau_validation(tau, error_type):
    with pytest.raises(error_type, match="tau"):
        concordance_time_dependent(
            np.array([[2.0], [1.0]]),
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.ones(2),
            tau=tau,
        )


@pytest.mark.parametrize(
    "working_memory",
    [0, -1, np.inf, np.nan, True, "large", "256", 1 + 0j, 10**400],
)
def test_working_memory_validation(working_memory):
    with pytest.raises(ValueError, match="working_memory"):
        concordance_time_dependent(
            np.array([[2.0], [1.0]]),
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.ones(2),
            working_memory=working_memory,
        )


@pytest.mark.parametrize(
    ("train_times", "train_indicators", "error_type", "message"),
    [
        (None, None, ValueError, "must be provided"),
        (np.array([]), np.array([]), ValueError, "must not be empty"),
        (np.array([1.0]), np.array([1, 0]), ValueError, "same shape"),
        (np.array([np.inf]), np.array([1]), ValueError, "finite"),
        (np.array([-1.0]), np.array([1]), ValueError, "non-negative"),
        (np.array(["1"]), np.array([1]), TypeError, "real numeric"),
        (np.array([1.0]), np.array([2]), ValueError, "only 0 and 1"),
        (np.array([1.0]), np.array([1j]), TypeError, "Boolean or real"),
    ],
)
def test_ipcw_training_validation(
    train_times,
    train_indicators,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        concordance_time_dependent(
            np.array([[2.0], [1.0]]),
            np.array([1.0]),
            np.array([1.0, 2.0]),
            np.ones(2),
            train_times,
            train_indicators,
            method="IPCW",
        )


def test_string_options_are_validated_cleanly():
    args = (
        np.array([[2.0], [1.0]]),
        np.array([1.0]),
        np.array([1.0, 2.0]),
        np.ones(2),
    )
    with pytest.raises(TypeError, match="method"):
        concordance_time_dependent(*args, method=True)
    with pytest.raises(ValueError, match="Unsupported"):
        concordance_time_dependent(*args, method="Harrell")
    with pytest.raises(TypeError, match="ties"):
        concordance_time_dependent(*args, ties=False)
    with pytest.raises(ValueError, match="handling ties"):
        concordance_time_dependent(*args, ties="Maybe")


@pytest.mark.parametrize("risks", ["Survival", "Hazard"])
@pytest.mark.parametrize("per_subject_grid", [False, True])
def test_evaluator_matches_precomputed_lower_level_risks(
    risks,
    per_subject_grid,
):
    shared_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    hazards = np.array([0.5, 0.4, 0.2, 0.1])
    if per_subject_grid:
        time_grid = np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [0.0, 0.8, 2.2, 3.4, 4.0],
                [0.0, 1.2, 2.4, 3.2, 4.0],
                [0.0, 0.5, 1.5, 3.5, 4.0],
            ]
        )
        pred_survs = np.exp(-hazards[:, None] * time_grid)
    else:
        time_grid = shared_grid
        pred_survs = np.exp(-hazards[:, None] * shared_grid)

    evaluator = SurvivalEvaluator(
        pred_survs=pred_survs,
        time_coordinates=time_grid,
        event_times=np.array([1.0, 1.0, 2.0, 3.0]),
        event_indicators=np.array([1, 1, 0, 1]),
    )
    risk_times = np.array([1.0])
    if risks == "Survival":
        risk_scores = -evaluator.predict_multi_probabilities_from_curve(risk_times)
    else:
        risk_scores = evaluator.predict_multi_hazards_from_curve(risk_times)

    expected = concordance_time_dependent(
        risk_scores,
        risk_times,
        evaluator.event_times,
        evaluator.event_indicators,
    )
    actual = evaluator.concordance_time_dependent(
        risks=risks,
        working_memory=1e-6,
    )
    np.testing.assert_allclose(actual, expected)


def test_evaluator_honors_explicit_ipcw_for_fully_observed_data():
    evaluator = SurvivalEvaluator(
        pred_survs=np.array(
            [
                [1.0, 0.2, 0.1, 0.05],
                [1.0, 0.7, 0.6, 0.4],
                [1.0, 0.8, 0.5, 0.3],
            ]
        ),
        time_coordinates=np.arange(4.0),
        event_times=np.array([1.0, 2.0, 3.0]),
        event_indicators=np.ones(3),
        train_event_times=np.array([1.5, 3.0]),
        train_event_indicators=np.array([0, 1]),
    )
    np.testing.assert_allclose(
        evaluator.concordance_time_dependent(method="Antolini"),
        (2 / 3, 2.0, 3.0),
    )
    np.testing.assert_allclose(
        evaluator.concordance_time_dependent(method="IPCW"),
        (1 / 3, 2.0, 6.0),
    )


def test_evaluator_preserves_float32_interpolation_precision():
    risk_gap = 1.2e-8
    evaluator = SurvivalEvaluator(
        pred_survs=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        time_coordinates=np.array(
            [
                [0.0, 1 / (0.5 + risk_gap)],
                [0.0, 2.0],
            ]
        ),
        event_times=np.array([1.0, 2.0]),
        event_indicators=np.array([1, 0]),
    )
    np.testing.assert_allclose(
        evaluator.concordance_time_dependent(risks="Survival"),
        (1.0, 1.0, 1.0),
    )


def test_hazard_evaluator_only_predicts_subject_prefixes():
    time_grids = np.array(
        [
            [0.0, 1.0, 4.0, 8.0, 12.0],
            [0.0, 0.5, 1.0, 1.5, 2.0],
            [0.0, 1.0, 4.0, 8.0, 12.0],
            [0.0, 1.0, 4.0, 8.0, 12.0],
        ]
    )
    hazards = np.array([0.5, 0.4, 0.3, 0.2])
    evaluator = SurvivalEvaluator(
        pred_survs=np.exp(-hazards[:, None] * time_grids),
        time_coordinates=time_grids,
        event_times=np.array([1.0, 2.0, 10.0, 12.0]),
        event_indicators=np.array([1, 0, 1, 0]),
    )
    with pytest.raises(ValueError, match="largest time coordinate"):
        evaluator.predict_multi_hazards_from_curve(np.array([1.0, 10.0]))
    np.testing.assert_allclose(
        evaluator.concordance_time_dependent(
            risks="Hazard",
            working_memory=1e-6,
        ),
        (1.0, 4.0, 4.0),
    )


def test_evaluator_tau_excludes_boundary_anchor_but_keeps_later_candidates():
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    hazards = np.array([0.5, 0.3, 0.1, 0.05])
    evaluator = SurvivalEvaluator(
        pred_survs=np.exp(-hazards[:, None] * grid),
        time_coordinates=grid,
        event_times=np.array([1.0, 2.0, 10.0, 12.0]),
        event_indicators=np.array([1, 1, 1, 0]),
        train_event_times=np.array([1.0, 2.0, 3.0, 4.0]),
        train_event_indicators=np.ones(4),
    )
    np.testing.assert_allclose(
        evaluator.concordance_time_dependent(
            method="IPCW",
            risks="Hazard",
            tau=2.0,
        ),
        (1.0, 3.0, 3.0),
    )


def test_evaluator_streams_only_compact_prefixes_with_bounded_rows(monkeypatch):
    n_samples = 80
    time_grid = np.linspace(0.0, 10.0, 12)
    hazards = np.linspace(0.5, 0.05, n_samples)
    evaluator = SurvivalEvaluator(
        pred_survs=np.exp(-hazards[:, None] * time_grid),
        time_coordinates=time_grid,
        event_times=np.linspace(0.1, 9.9, n_samples),
        event_indicators=np.ones(n_samples),
    )
    calls = []
    original = evaluator._fill_multi_curve_predictions

    def record_fill(
        output,
        sample_indices,
        survival_curves,
        time_grids,
        target_times,
        predictor,
        target_counts=None,
    ):
        calls.append(
            (
                output.shape,
                output.nbytes,
                target_times.size,
                target_counts.copy(),
                output.flags.f_contiguous,
            )
        )
        return original(
            output,
            sample_indices,
            survival_curves,
            time_grids,
            target_times,
            predictor,
            target_counts,
        )

    monkeypatch.setattr(evaluator, "_fill_multi_curve_predictions", record_fill)
    working_memory = 0.005
    budget = int(working_memory * (1 << 20))
    result = evaluator.concordance_time_dependent(working_memory=working_memory)

    assert len(calls) > 1
    assert all(nbytes <= budget for _, nbytes, _, _, _ in calls)
    assert all(shape != (n_samples, n_samples - 1) for shape, *_ in calls)
    assert all(fortran_order for *_, fortran_order in calls)
    assert all(np.all(counts[:-1] <= counts[1:]) for *_, counts, _ in calls)
    pair_count = n_samples * (n_samples - 1) / 2
    np.testing.assert_allclose(result, (1.0, pair_count, pair_count))


def test_one_complete_compact_row_is_the_minimum_buffer(monkeypatch):
    event_times = np.arange(1, 10, dtype=float)
    risk_times = event_times[:-1]
    risk_scores = np.broadcast_to(
        np.arange(9, 0, -1, dtype=float)[:, None],
        (9, risk_times.size),
    ).copy()
    output_shapes = []
    original = td_concordance._fill_matrix_risk_batch

    def record_fill(*args):
        output_shapes.append(args[-1].shape)
        return original(*args)

    monkeypatch.setattr(td_concordance, "_fill_matrix_risk_batch", record_fill)
    result = concordance_time_dependent(
        risk_scores,
        risk_times,
        event_times,
        np.ones(9),
        working_memory=1 / (1 << 20),
    )
    assert output_shapes
    assert all(rows == 1 for rows, _ in output_shapes)
    assert max(columns for _, columns in output_shapes) == risk_times.size
    np.testing.assert_allclose(result, (1.0, 36.0, 36.0))


def test_interleaved_extra_coordinates_do_not_inflate_compact_buffer(monkeypatch):
    n_samples = 32
    event_times = np.arange(1, n_samples + 1, dtype=float)
    risk_times = np.arange(0.5, n_samples + 0.5, 0.5)
    risk_scores = np.empty((n_samples, risk_times.size))
    risk_scores[:, ::2] = 1000.0
    risk_scores[:, 1::2] = np.arange(n_samples, 0, -1)[:, None]
    seen_required_columns = []
    seen_output_columns = []
    original = td_concordance._fill_matrix_risk_batch

    def record_fill(scores, required_columns, indices, prefixes, output):
        seen_required_columns.append(required_columns.copy())
        seen_output_columns.append(output.shape[1])
        return original(scores, required_columns, indices, prefixes, output)

    monkeypatch.setattr(td_concordance, "_fill_matrix_risk_batch", record_fill)
    result = concordance_time_dependent(
        risk_scores,
        risk_times,
        event_times,
        np.ones(n_samples),
        working_memory=0.001,
    )

    expected_columns = np.arange(1, risk_times.size - 1, 2)
    assert seen_required_columns
    assert all(
        np.array_equal(columns, expected_columns) for columns in seen_required_columns
    )
    assert max(seen_output_columns) == n_samples - 1
    assert max(seen_output_columns) < risk_times.size
    pair_count = n_samples * (n_samples - 1) / 2
    np.testing.assert_allclose(result, (1.0, pair_count, pair_count))


def test_matrix_filler_uses_unbuffered_take_mode(monkeypatch):
    calls = []
    original_take = td_concordance.np.take

    def record_take(array, indices, *, out, mode):
        calls.append((array.ndim, out.ndim, mode, out.size))
        return original_take(array, indices, out=out, mode=mode)

    monkeypatch.setattr(td_concordance.np, "take", record_take)
    concordance_time_dependent(
        np.array([[3.0, 0.0], [2.0, 2.0], [1.0, 1.0]]),
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        np.ones(3),
        working_memory=1e-6,
    )
    assert calls
    assert all(
        input_ndim == output_ndim == 1 for input_ndim, output_ndim, _, _ in calls
    )
    assert all(mode == "clip" for _, _, mode, _ in calls)


def test_matrix_path_peak_memory_excludes_second_risk_batch():
    n_samples = 512
    event_times = np.arange(1, n_samples + 1, dtype=float)
    risk_times = event_times[:-1]
    risk_scores = np.broadcast_to(
        np.arange(n_samples, 0, -1, dtype=float)[:, None],
        (n_samples, risk_times.size),
    ).copy()
    working_memory = 0.25
    budget = int(working_memory * (1 << 20))

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        result = concordance_time_dependent(
            risk_scores,
            risk_times,
            event_times,
            np.ones(n_samples),
            working_memory=working_memory,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    # The compact risk buffer is one budget; O(n) layout and 1-D workspaces fit
    # well below the allowance that a second output-sized buffer would consume.
    assert peak - baseline < 1.6 * budget
    pair_count = n_samples * (n_samples - 1) / 2
    np.testing.assert_allclose(result, (1.0, pair_count, pair_count))
