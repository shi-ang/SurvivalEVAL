import numpy as np
import pytest

from SurvivalEVAL import PointEvaluator
from SurvivalEVAL.Evaluations.Concordance import (
    ConcordanceCounts,
    _finalize_counts,
    _get_comparable_ic,
    _margin_counts,
    _right_censored_risk_counts,
    concordance,
)


def _add_directed_pair(counts, risks, anchor, candidate, weight=1.0, tied_tol=1e-8):
    risk_diff = risks[candidate] - risks[anchor]
    if abs(risk_diff) <= tied_tol:
        counts.risk_tie_pairs += weight
    elif risk_diff < 0:
        counts.concordant += weight
    else:
        counts.discordant += weight


def _brute_harrell_counts(event_indicators, event_times, risks, sample_weights=None):
    if sample_weights is None:
        sample_weights = np.ones(event_times.shape[0], dtype=float)
    counts = ConcordanceCounts()

    for i in range(event_times.shape[0]):
        for j in range(i + 1, event_times.shape[0]):
            if (
                event_indicators[i]
                and event_indicators[j]
                and event_times[i] == event_times[j]
            ):
                counts.time_tie_pairs += sample_weights[i] * sample_weights[j]

    for i in range(event_times.shape[0]):
        if not event_indicators[i]:
            continue
        for j in range(event_times.shape[0]):
            if i == j:
                continue
            is_later = event_times[j] > event_times[i]
            is_same_time_censored = (
                event_times[j] == event_times[i] and not event_indicators[j]
            )
            if is_later or is_same_time_censored:
                _add_directed_pair(
                    counts,
                    risks,
                    anchor=i,
                    candidate=j,
                    weight=sample_weights[i] * sample_weights[j],
                )

    return counts


def _brute_margin_counts(
    event_indicators, event_times, risks, bg_event_times, partial_weights
):
    counts = _brute_harrell_counts(
        np.ones_like(event_indicators, dtype=bool),
        bg_event_times,
        risks,
        sample_weights=partial_weights,
    )

    for i in range(event_times.shape[0]):
        if not event_indicators[i]:
            continue
        for j in range(event_times.shape[0]):
            if i == j:
                continue
            is_later = event_times[j] > event_times[i]
            is_same_time_censored = (
                event_times[j] == event_times[i] and not event_indicators[j]
            )
            if not (is_later or is_same_time_censored):
                continue

            baseline_weight = partial_weights[i] * partial_weights[j]
            if bg_event_times[i] < bg_event_times[j]:
                _add_directed_pair(counts, risks, i, j, weight=-baseline_weight)
            elif bg_event_times[i] > bg_event_times[j]:
                _add_directed_pair(counts, risks, j, i, weight=-baseline_weight)
            else:
                counts.time_tie_pairs -= baseline_weight

            _add_directed_pair(counts, risks, i, j, weight=1.0)

    return counts


def _assert_counts_close(actual, expected):
    assert np.isclose(actual.concordant, expected.concordant)
    assert np.isclose(actual.discordant, expected.discordant)
    assert np.isclose(actual.risk_tie_pairs, expected.risk_tie_pairs)
    assert np.isclose(actual.time_tie_pairs, expected.time_tie_pairs)


def test_concordance_default_gives_half_credit_to_tied_risk_predictions():
    predicted_times = np.array([1.0, 1.0, 3.0])
    event_times = np.array([1.0, 2.0, 3.0])
    event_indicators = np.array([1, 1, 1])

    c_default, concordant_default, total_default = concordance(
        predicted_times,
        event_times,
        event_indicators,
        method="Harrell",
    )
    c_no_ties, concordant_no_ties, total_no_ties = concordance(
        predicted_times,
        event_times,
        event_indicators,
        method="Harrell",
        ties="None",
    )

    assert np.isclose(c_default, 5.0 / 6.0)
    assert np.isclose(concordant_default, 2.5)
    assert np.isclose(total_default, 3.0)
    assert np.isclose(c_no_ties, 1.0)
    assert np.isclose(concordant_no_ties, 2.0)
    assert np.isclose(total_no_ties, 2.0)


def test_point_evaluator_concordance_default_uses_standard_harrell_tie_handling():
    evaluator = PointEvaluator(
        pred_times=np.array([1.0, 1.0, 3.0]),
        event_times=np.array([1.0, 2.0, 3.0]),
        event_indicators=np.array([1, 1, 1]),
    )

    c_default, concordant_default, total_default = evaluator.concordance()
    c_risk, concordant_risk, total_risk = evaluator.concordance(ties="Risk")

    assert np.isclose(c_default, c_risk)
    assert np.isclose(concordant_default, concordant_risk)
    assert np.isclose(total_default, total_risk)
    assert np.isclose(c_default, 5.0 / 6.0)


def test_concordance_tie_modes_separate_risk_time_and_censoring_ties():
    # This example has 8 concordant pairs, 3 discordant pairs, 5 comparable
    # risk ties, and 2 event-event time ties. It also includes event-censored
    # same-time pairs and censored-censored same-time pairs, which must not be
    # counted as time ties.
    predicted_times = np.array([1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 4.0, 2.0])
    event_times = np.array([1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 3.0])
    event_indicators = np.array([1, 1, 1, 0, 1, 0, 0, 1])

    c_none, concordant_none, total_none = concordance(
        predicted_times,
        event_times,
        event_indicators,
        ties="None",
    )
    c_risk, concordant_risk, total_risk = concordance(
        predicted_times,
        event_times,
        event_indicators,
        ties="Risk",
    )
    c_time, concordant_time, total_time = concordance(
        predicted_times,
        event_times,
        event_indicators,
        ties="Time",
    )
    c_all, concordant_all, total_all = concordance(
        predicted_times,
        event_times,
        event_indicators,
        ties="All",
    )

    assert np.isclose(c_none, 8.0 / 11.0)
    assert np.isclose(concordant_none, 8.0)
    assert np.isclose(total_none, 11.0)

    assert np.isclose(c_risk, 10.5 / 16.0)
    assert np.isclose(concordant_risk, 10.5)
    assert np.isclose(total_risk, 16.0)

    assert np.isclose(c_time, 9.0 / 13.0)
    assert np.isclose(concordant_time, 9.0)
    assert np.isclose(total_time, 13.0)

    assert np.isclose(c_all, 11.5 / 18.0)
    assert np.isclose(concordant_all, 11.5)
    assert np.isclose(total_all, 18.0)


@pytest.mark.parametrize("ties", ["None", "Risk", "Time", "All"])
def test_harrell_concordance_matches_brute_force_edge_cases(ties):
    cases = [
        (
            np.array([1.0, 1.0, 1.0, 3.0, 0.0, 1.0, 4.0, 2.0]),
            np.array([1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 2.0, 3.0]),
            np.array([1, 1, 1, 0, 1, 0, 0, 1]),
        ),
        (
            np.array([2.0, 2.0, 1.0, 4.0, 3.0]),
            np.array([1.0, 2.0, 2.0, 2.0, 4.0]),
            np.array([1, 1, 0, 1, 0]),
        ),
    ]

    for predicted_times, event_times, event_indicators in cases:
        risks = -predicted_times
        expected_counts = _brute_harrell_counts(event_indicators, event_times, risks)
        expected = _finalize_counts(expected_counts, ties)
        actual = concordance(
            predicted_times,
            event_times,
            event_indicators,
            method="Harrell",
            ties=ties,
        )
        np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_private_right_censored_risk_counts_match_brute_force_for_random_small_inputs():
    rng = np.random.default_rng(0)

    for n_samples in range(2, 10):
        for _ in range(50):
            event_times = rng.integers(1, 5, size=n_samples).astype(float)
            event_indicators = rng.random(n_samples) < 0.65
            risks = rng.integers(-2, 3, size=n_samples).astype(float)

            actual = _right_censored_risk_counts(event_indicators, event_times, risks)
            expected = _brute_harrell_counts(event_indicators, event_times, risks)

            _assert_counts_close(actual, expected)


def test_margin_concordance_sorts_pairs_by_best_guess_times():
    event_indicators = np.array([True, False, True])
    event_times = np.array([1.0, 2.0, 3.0])
    bg_event_times = np.array([1.0, 4.0, 3.0])
    partial_weights = np.array([1.0, 0.5, 1.0])
    risks = np.array([3.0, 1.0, 2.0])

    counts = _margin_counts(
        event_indicators,
        event_times,
        estimate=risks,
        bg_event_time=bg_event_times,
        partial_weights=partial_weights,
    )
    c_index, concordant, _ = _finalize_counts(counts, "Risk")

    assert np.isclose(c_index, 1.0)
    assert np.isclose(concordant, 2.5)
    assert np.isclose(counts.discordant, 0.0)
    assert np.isclose(counts.risk_tie_pairs, 0.0)
    assert np.isclose(counts.time_tie_pairs, 0.0)


def test_margin_counts_match_brute_force_when_known_pairs_override_best_guess_order():
    event_indicators = np.array([True, False, False, True])
    event_times = np.array([2.0, 2.0, 4.0, 5.0])
    bg_event_times = np.array([2.0, 2.0, 1.0, 5.0])
    partial_weights = np.array([1.0, 0.25, 0.5, 1.0])
    risks = np.array([3.0, 2.0, 1.0, 2.0])

    actual = _margin_counts(
        event_indicators,
        event_times,
        estimate=risks,
        bg_event_time=bg_event_times,
        partial_weights=partial_weights,
    )
    expected = _brute_margin_counts(
        event_indicators,
        event_times,
        risks,
        bg_event_times,
        partial_weights,
    )

    _assert_counts_close(actual, expected)
    assert np.isclose(actual.concordant, 3.0)
    assert np.isclose(actual.discordant, 0.625)
    assert np.isclose(actual.time_tie_pairs, 0.0)


def test_margin_known_pairs_full_weight_when_best_guess_times_tie():
    event_indicators = np.array([True, False])
    event_times = np.array([1.0, 1.0])
    bg_event_times = np.array([1.0, 1.0])
    partial_weights = np.array([1.0, 0.25])
    risks = np.array([2.0, 1.0])

    actual = _margin_counts(
        event_indicators,
        event_times,
        estimate=risks,
        bg_event_time=bg_event_times,
        partial_weights=partial_weights,
    )

    assert np.isclose(actual.concordant, 1.0)
    assert np.isclose(actual.discordant, 0.0)
    assert np.isclose(actual.risk_tie_pairs, 0.0)
    assert np.isclose(actual.time_tie_pairs, 0.0)


def test_margin_concordance_preserves_known_pairs_when_best_guess_times_tie():
    event_indicators = np.array([True, False])
    event_times = np.array([1.0, 1.0])
    bg_event_times = np.array([1.0, 1.0])
    partial_weights = np.array([1.0, 0.25])
    risks = np.array([2.0, 1.0])

    counts = _margin_counts(
        event_indicators,
        event_times,
        estimate=risks,
        bg_event_time=bg_event_times,
        partial_weights=partial_weights,
    )
    c_index, concordant, _ = _finalize_counts(counts, "Risk")

    assert np.isclose(c_index, 1.0)
    assert np.isclose(concordant, 1.0)
    assert np.isclose(counts.discordant, 0.0)
    assert np.isclose(counts.risk_tie_pairs, 0.0)
    assert np.isclose(counts.time_tie_pairs, 0.0)


def test_get_comparable_ic_returns_directed_precedence_relation():
    left = np.array([0.0, 2.0, 4.0])
    right = np.array([1.0, 3.0, 5.0])

    comparable = _get_comparable_ic(left, right)

    expected = np.array(
        [
            [False, True, True],
            [False, False, True],
            [False, False, False],
        ]
    )
    np.testing.assert_array_equal(comparable, expected)


def test_get_comparable_ic_respects_touching_endpoint_inclusivity():
    # (0, 1] precedes (1, 2], but overlaps the exact event [1, 1].
    left = np.array([0.0, 1.0, 1.0])
    right = np.array([1.0, 2.0, 1.0])

    comparable = _get_comparable_ic(left, right)

    assert comparable[0, 1]
    assert not comparable[1, 0]
    assert not comparable[0, 2]
    assert not comparable[2, 0]
