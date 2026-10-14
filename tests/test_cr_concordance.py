import numpy as np
import pytest

from SurvivalEVAL import concordance, concordance_cr, concordance_cr_per_event
from SurvivalEVAL.Evaluations.Discrimination._concordance_utils import (
    _ConcordanceCounts,
    _finalize_counts,
)


def _direct_counts(risks, times, events, cause, weights, tau):
    """Evaluate the Wolbers pair definitions directly, without sorted counting."""
    counts = _ConcordanceCounts()
    for i in range(times.size):
        if events[i] != cause or (tau is not None and times[i] >= tau):
            continue
        for j in range(times.size):
            if i == j:
                continue
            if events[j] == cause and times[i] == times[j] and i < j:
                counts.time_tie_pairs += weights[i] * weights[j]
            type_a = times[i] < times[j] or (times[i] == times[j] and events[j] == 0)
            type_b = events[j] not in (0, cause) and times[j] <= times[i]
            if not (type_a or type_b):
                continue
            weight = weights[i] ** 2 if type_a else weights[i] * weights[j]
            if abs(risks[i] - risks[j]) <= 1e-8:
                counts.risk_tie_pairs += weight
            elif risks[i] > risks[j]:
                counts.concordant += weight
            else:
                counts.discordant += weight
    return counts


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["Naive", "IPCW"])
@pytest.mark.parametrize("ties", ["None", "Risk", "Time", "All"])
@pytest.mark.parametrize("tau", [None, 4.0])
def test_matches_direct_pair_definitions(dtype, method, ties, tau):
    rng = np.random.default_rng(73)
    risks = rng.integers(-3, 4, 60).astype(dtype)
    times = rng.integers(0, 8, 60).astype(dtype)
    events = rng.choice([0, 1, 2, 7], 60)
    train_times = np.array([1, 2, 2, 4, 6, 10], dtype=dtype)
    train_events = np.array([0, 1, 0, 7, 0, 2])
    weights = np.ones(times.shape)
    if method == "IPCW":
        # Independent reverse risk-set product, excluding each query's update.
        for i, query in enumerate(times):
            for time in np.unique(
                train_times[(train_events == 0) & (train_times < query)]
            ):
                censorings = np.count_nonzero(
                    (train_times == time) & (train_events == 0)
                )
                risk_set = np.count_nonzero(train_times > time) + censorings
                weights[i] /= 1 - censorings / risk_set
    expected = _finalize_counts(
        _direct_counts(risks, times, events, 2, weights, tau), ties
    )
    for values in (risks, times, events, train_times, train_events):
        values.flags.writeable = False
    actual = concordance_cr_per_event(
        risks,
        times,
        events,
        train_times,
        train_events,
        method=method,
        ties=ties,
        tau=tau,
        event_of_interest=2,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_hand_computed_ipcw_distinguishes_type_a_and_b_weights():
    risks = [0.7, 0.8, 0.9, 0.2]
    times = [4, 2, 8, 6]
    events = [1, 2, 1, 0]
    # G(2)=1, G(4)=2/3, and G(6)=G(8)=1/3.
    train_times, train_events = [1, 3, 5, 10], [2, 0, 0, 1]
    unweighted = concordance_cr_per_event(risks, times, events)
    weighted = concordance_cr_per_event(
        risks, times, events, train_times, train_events, method="IPCW"
    )
    truncated = concordance_cr_per_event(
        risks, times, events, train_times, train_events, method="IPCW", tau=8
    )
    # Anchor at 4: discordant A pair (9/4), concordant A pair (9/4),
    # and discordant B pair (3/2). Anchor at 8: concordant B pair (3).
    np.testing.assert_allclose(unweighted, (0.5, 2, 4))
    np.testing.assert_allclose(weighted, (7 / 12, 21 / 4, 9))
    np.testing.assert_allclose(truncated, (3 / 8, 9 / 4, 6))


@pytest.mark.parametrize("method", ["Naive", "IPCW"])
def test_tau_only_truncates_anchors_and_keeps_later_comparators(method):
    actual = concordance_cr_per_event(
        [3, 1, 2, 4],
        [1, 2, 3, 4],
        [1, 2, 0, 1],
        [1, 2, 3, 4],
        [1, 2, 3, 1],
        method=method,
        tau=4,
    )
    np.testing.assert_allclose(actual, (2 / 3, 2, 3))
    # With tau=None, the final selected cause still has the earlier competitor.
    full = concordance_cr_per_event(
        [3, 1, 2, 4],
        [1, 2, 3, 4],
        [1, 2, 0, 1],
        [1, 2, 3, 4],
        [1, 2, 3, 1],
        method=method,
    )
    np.testing.assert_allclose(full, (0.75, 3, 4))


@pytest.mark.parametrize(
    "ties, expected",
    [
        ("Risk", (1, 4, 4)),
        ("None", (1, 4, 4)),
        ("Time", (0.9, 4.5, 5)),
        ("All", (0.9, 4.5, 5)),
    ],
)
def test_competing_event_time_ties_are_counted_once(ties, expected):
    actual = concordance_cr_per_event([4, 3, 2, 1], [1] * 4, [1, 1, 2, 0], ties=ties)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ties", ["Risk", "All", "None", "Time"])
def test_prediction_ties_receive_half_credit_or_are_excluded(ties):
    result = concordance_cr_per_event([0.5] * 4, [1, 2, 3, 4], [2, 1, 0, 1], ties=ties)
    if ties in ("Risk", "All"):
        np.testing.assert_allclose(result, (0.5, 2, 4))
    else:
        assert np.isnan(result[0])
        assert result[1:] == (0, 0)


def test_risk_tolerance_applies_to_earlier_competing_events():
    np.testing.assert_allclose(
        concordance_cr_per_event([0, 1e-8], [1, 2], [2, 1]), (0.5, 0.5, 1)
    )
    np.testing.assert_allclose(
        concordance_cr_per_event([0, 2e-8], [1, 2], [2, 1]), (1, 1, 1)
    )


@pytest.mark.parametrize("events", [[1, 0], [2, 1]])
def test_float32_risk_rounding_does_not_widen_tie_tolerance(events):
    lower = np.float32(0.2)
    higher = np.nextafter(lower, np.float32(1))
    assert higher - lower > 1e-8
    risks = np.array([higher, lower] if events[0] == 1 else [lower, higher])
    risks.flags.writeable = False
    np.testing.assert_allclose(
        concordance_cr_per_event(risks, [1, 2], events), (1, 1, 1)
    )


@pytest.mark.parametrize("method", ["Naive", "IPCW"])
@pytest.mark.parametrize("ties", ["None", "Risk", "Time", "All"])
def test_no_competing_events_reduces_to_existing_concordance(method, ties):
    risks = np.array([2, 1, 3, 0, 4], dtype=float)
    times = np.array([1, 2, 2, 3, 4], dtype=float)
    events = np.array([1, 1, 1, 0, 1])
    train_times = np.array([1, 2, 5], dtype=float)
    train_events = np.array([0, 1, 1])
    expected = concordance(
        -risks,
        times,
        events,
        train_times,
        train_events,
        method=method,
        ties=ties,
        tau=3,
    )
    actual = concordance_cr_per_event(
        risks, times, events, train_times, train_events, method=method, ties=ties, tau=3
    )
    np.testing.assert_allclose(actual, expected)


def test_preserves_competing_labels_instead_of_treating_them_as_censoring():
    np.testing.assert_allclose(
        concordance_cr_per_event([0.2, 0.8], [2, 4], [7, 3], event_of_interest=3),
        (1, 1, 1),
    )
    with pytest.raises(ValueError, match="no comparable pairs"):
        concordance(-np.array([0.2, 0.8]), np.array([2, 4]), np.array([0, 1]))


def test_unused_zero_censoring_survival_is_ignored():
    np.testing.assert_allclose(
        concordance_cr_per_event(
            [3, 2, 1], [1, 3, 4], [1, 1, 2], [1, 3], [1, 0], method="IPCW", tau=2
        ),
        (1, 2, 2),
    )
    # A final selected-cause anchor with no comparators needs no G at its time.
    np.testing.assert_allclose(
        concordance_cr_per_event(
            [3, 2, 1], [1, 2, 3], [1, 1, 1], [1, 3], [1, 0], method="IPCW"
        ),
        (1, 3, 3),
    )


def test_final_anchor_with_earlier_competitor_requires_positive_censoring_survival():
    with pytest.raises(ValueError, match="Censoring survival"):
        concordance_cr_per_event(
            [0.2, 0.8], [1, 4], [2, 1], [1, 3], [1, 0], method="IPCW"
        )


@pytest.mark.parametrize("ties", ["Risk", "None", "Time", "All"])
def test_zero_survival_for_ignored_final_selected_cause_time_ties(ties):
    kwargs = dict(
        predicted_risks=[3, 2, 1],
        event_times=[1, 3, 3],
        event_indicators=[1, 1, 1],
        train_event_times=[1, 2],
        train_event_indicators=[1, 0],
        method="IPCW",
        ties=ties,
    )
    if ties in ("Time", "All"):
        with pytest.raises(ValueError, match="Censoring survival"):
            concordance_cr_per_event(**kwargs)
    else:
        np.testing.assert_allclose(concordance_cr_per_event(**kwargs), (1, 2, 2))


@pytest.mark.parametrize(
    "events, tau", [([0, 0], None), ([2, 2], None), ([1, 1], 0), ([1, 0], 1)]
)
def test_no_eligible_pairs(events, tau):
    with pytest.raises(ValueError, match="no comparable pairs"):
        concordance_cr_per_event([0.8, 0.2], [1, 2], events, tau=tau)


def test_training_outcomes_are_required():
    with pytest.raises(ValueError, match="Training"):
        concordance_cr_per_event([0.8, 0.2], [1, 2], [1, 2], method="IPCW")


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("predicted_risks", [[0.8, 0.2]], "one entry"),
        ("predicted_risks", [0.8], "one entry"),
        ("predicted_risks", [np.inf, 0.2], "finite scores"),
        ("event_times", [-1, 2], "non-negative"),
        ("event_times", [np.inf, 2], "finite"),
        ("event_times", [1], "same shape"),
        ("event_indicators", [1, 0.5], "integer labels"),
        ("event_indicators", [1, -1], "integer labels"),
        ("event_indicators", [1, np.inf], "integer labels"),
        ("event_of_interest", 0, "positive integer"),
        ("event_of_interest", True, "positive integer"),
        ("event_of_interest", 1.5, "positive integer"),
        ("method", "Margin", "method"),
        ("method", "Harrell", "method"),
        ("method", "Uno", "method"),
        ("ties", "skip", "handling ties"),
        ("tau", -1, "non-negative"),
        ("tau", np.nan, "finite"),
        ("tau", np.inf, "finite"),
    ],
)
def test_invalid_inputs(name, value, message):
    kwargs = dict(
        predicted_risks=[0.8, 0.2], event_times=[1, 2], event_indicators=[1, 2]
    )
    kwargs[name] = value
    with pytest.raises(ValueError, match=message):
        concordance_cr_per_event(**kwargs)


@pytest.mark.parametrize(
    "name, value",
    [
        ("train_event_times", [-1, 3]),
        ("train_event_times", [1]),
        ("train_event_indicators", [0.5, 1]),
    ],
)
def test_invalid_training_outcomes(name, value):
    kwargs = dict(
        predicted_risks=[0.8, 0.2],
        event_times=[1, 2],
        event_indicators=[1, 2],
        train_event_times=[1, 3],
        train_event_indicators=[0, 2],
        method="IPCW",
    )
    kwargs[name] = value
    with pytest.raises(ValueError):
        concordance_cr_per_event(**kwargs)


def test_noncontiguous_readonly_scores_and_monotone_transformations():
    storage = np.array([3, 0, 1, 0, 2, 0, 4, 0], dtype=float)
    scores = storage[::2]
    scores.flags.writeable = False
    times, events = [1, 2, 3, 4], [1, 2, 0, 1]
    np.testing.assert_allclose(
        concordance_cr_per_event(scores, times, events),
        concordance_cr_per_event(np.exp(scores), times, events),
    )
    np.testing.assert_array_equal(storage, [3, 0, 1, 0, 2, 0, 4, 0])


@pytest.mark.parametrize(
    "method, reduction, expected",
    [
        ("Naive", "mean", 1 / 4),
        ("Naive", "pooled", 2 / 7),
        ("IPCW", "mean", 7 / 24),
        ("IPCW", "pooled", 7 / 16),
    ],
)
def test_hand_computed_aggregate_scores(method, reduction, expected):
    risks = [[0.7, 0.3], [0.8, 0.1], [0.9, 0.4], [0.2, 0.2]]
    score, details = concordance_cr(
        risks,
        [4, 2, 8, 6],
        [1, 2, 1, 0],
        [1, 3, 5, 10],
        [2, 0, 0, 1],
        method=method,
        reduction=reduction,
        return_details=True,
    )
    assert score == pytest.approx(expected)
    assert details["per_event_scores"] == pytest.approx(
        {1: 0.5 if method == "Naive" else 7 / 12, 2: 0}
    )
    assert details["per_event_concordant_pairs"] == pytest.approx(
        {1: 2 if method == "Naive" else 21 / 4, 2: 0}
    )
    assert details["per_event_total_pairs"] == pytest.approx(
        {1: 4 if method == "Naive" else 9, 2: 3}
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["Naive", "IPCW"])
@pytest.mark.parametrize("ties", ["None", "Risk", "Time", "All"])
@pytest.mark.parametrize("tau", [None, 4])
@pytest.mark.parametrize("reduction", ["mean", "pooled"])
def test_aggregate_matches_per_event_results(dtype, method, ties, tau, reduction):
    rng = np.random.default_rng(73)
    # Use non-contiguous read-only columns to cover NumPy views and float32 risks.
    risks = rng.integers(-3, 4, (60, 6)).astype(dtype)[:, ::2]
    times = rng.integers(0, 8, 60).astype(dtype)
    events = rng.integers(0, 4, 60)
    train_times = [1, 2, 2, 4, 6, 10]
    train_events = [0, 1, 0, 3, 0, 2]
    risks.flags.writeable = False
    times.flags.writeable = False
    events.flags.writeable = False
    results = [
        concordance_cr_per_event(
            risks[:, k],
            times,
            events,
            train_times,
            train_events,
            method=method,
            ties=ties,
            tau=tau,
            event_of_interest=k + 1,
        )
        for k in range(3)
    ]
    expected = (
        np.mean([item[0] for item in results])
        if reduction == "mean"
        else sum(item[1] for item in results) / sum(item[2] for item in results)
    )
    actual = concordance_cr(
        risks,
        times,
        events,
        train_times,
        train_events,
        method=method,
        ties=ties,
        tau=tau,
        reduction=reduction,
    )
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("events, tau", [([1, 0, 1], None), ([1, 2, 1], 2)])
def test_undefined_causes_are_reported_without_silently_omitting_them(events, tau):
    risks = [[0.9, 0.3], [0.4, 0.2], [0.1, 0.1]]
    mean, details = concordance_cr(
        risks, [1, 2, 3], events, tau=tau, return_details=True
    )
    assert np.isnan(mean)
    assert details["per_event_scores"][1] == 1
    assert np.isnan(details["per_event_scores"][2])
    assert details["per_event_concordant_pairs"][2] == 0
    assert details["per_event_total_pairs"][2] == 0
    assert concordance_cr(risks, [1, 2, 3], events, tau=tau, reduction="pooled") == 1


@pytest.mark.parametrize("reduction", ["mean", "pooled"])
def test_aggregate_is_undefined_when_all_causes_have_no_pairs(reduction):
    score, details = concordance_cr(
        [[0.9, 0.1], [0.1, 0.9]],
        [1, 2],
        [0, 0],
        reduction=reduction,
        return_details=True,
    )
    assert np.isnan(score)
    assert all(np.isnan(value) for value in details["per_event_scores"].values())
    assert details["per_event_total_pairs"] == {1: 0, 2: 0}


def test_aggregate_handles_causes_with_only_excluded_prediction_ties():
    score, details = concordance_cr(
        [[0.5, 0.1], [0.5, 0.9], [0.5, 0.2]],
        [1, 2, 3],
        [1, 2, 0],
        ties="None",
        return_details=True,
    )
    assert np.isnan(score)
    assert np.isnan(details["per_event_scores"][1])
    assert details["per_event_total_pairs"][1] == 0
    assert details["per_event_scores"][2] == 1
    assert (
        concordance_cr(
            [[0.5, 0.1], [0.5, 0.9], [0.5, 0.2]],
            [1, 2, 3],
            [1, 2, 0],
            ties="None",
            reduction="pooled",
        )
        == 1
    )


@pytest.mark.parametrize("reduction", ["mean", "pooled"])
def test_aggregate_with_one_cause_matches_per_event(reduction):
    result = concordance_cr_per_event([3, 1, 2], [1, 2, 3], [1, 0, 1])
    assert (
        concordance_cr([[3], [1], [2]], [1, 2, 3], [1, 0, 1], reduction=reduction)
        == result[0]
    )


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("predicted_risks", [0.8, 0.2], "shape"),
        ("predicted_risks", [[0.8], [0.2], [0.1]], "shape"),
        ("predicted_risks", [[np.inf, 0.2], [0.8, 0.1]], "finite scores"),
        ("event_indicators", [1, 3], "number of predicted causes"),
        ("event_indicators", [1, -1], "integer labels"),
        ("event_times", [-1, 2], "non-negative"),
        ("method", "Uno", "method"),
        ("method", "Harrell", "method"),
        ("method", "IPCW", "Training"),
        ("ties", "skip", "handling ties"),
        ("tau", np.nan, "finite"),
        ("tau", -1, "non-negative"),
        ("reduction", "sum", "reduction"),
    ],
)
def test_invalid_aggregate_inputs(name, value, message):
    kwargs = dict(
        predicted_risks=[[0.8, 0.2], [0.2, 0.8]],
        event_times=[1, 2],
        event_indicators=[1, 2],
    )
    kwargs[name] = value
    with pytest.raises(ValueError, match=message):
        concordance_cr(**kwargs)


def test_aggregate_does_not_hide_invalid_ipcw_probabilities():
    with pytest.raises(ValueError, match="Censoring survival"):
        concordance_cr(
            [[0.2, 0.8], [0.8, 0.2]], [1, 4], [2, 1], [1, 3], [1, 0], method="IPCW"
        )
