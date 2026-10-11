import numpy as np
import pytest

from SurvivalEVAL.NonparametricEstimator.CompetingRisks import (
    AalenJohansenCompetingRisks,
)
from SurvivalEVAL.NonparametricEstimator.MultiState import AalenJohansenMultiState


def test_multistate_product_with_return_transitions():
    transitions = np.zeros((2, 3, 3))
    transitions[0, 0, 1:] = [4, 2]
    transitions[1, 0, 1] = 1
    transitions[1, 1, [0, 2]] = [1, 2]
    estimator = AalenJohansenMultiState(3).fit(
        [1, 2], [[10, 0, 0], [4, 4, 2]], transitions
    )
    expected = [
        [[0.4, 0.4, 0.2], [0, 1, 0], [0, 0, 1]],
        [[0.4, 0.2, 0.4], [0.25, 0.25, 0.5], [0, 0, 1]],
    ]
    np.testing.assert_allclose(estimator.P_, expected)
    np.testing.assert_allclose(estimator.P_.sum(axis=2), 1)
    assert np.all(estimator.P_ >= 0)
    np.testing.assert_allclose(estimator.predict_P(0), np.eye(3))
    np.testing.assert_allclose(estimator.predict_P(np.nextafter(1.0, 0.0)), np.eye(3))
    np.testing.assert_allclose(estimator.predict_P([1, 1.5]), [expected[0]] * 2)
    np.testing.assert_allclose(estimator.predict_P([2, np.inf]), [expected[1]] * 2)


def test_chronological_product_does_not_reverse_or_merge_updates():
    transitions = np.zeros((2, 3, 3))
    transitions[0, 0, 1] = 2
    transitions[1, 1, 2] = 1
    estimator = AalenJohansenMultiState(3).fit(
        [0, 1], [[4, 0, 0], [2, 2, 0]], transitions
    )
    np.testing.assert_allclose(estimator.predict_P(0)[0], [0.5, 0.5, 0])
    np.testing.assert_allclose(estimator.predict_P(1)[0], [0.5, 0.25, 0.25])


def test_competing_risks_is_a_special_case():
    times = np.array([0, 1, 1, 1, 2, 3, 3, 4])
    events = np.array([1, 1, 2, 0, 2, 1, 0, 0])
    event_times = np.unique(times[events > 0])
    risk_sets = np.zeros((event_times.size, 3))
    transitions = np.zeros((event_times.size, 3, 3))
    for j, t in enumerate(event_times):
        risk_sets[j, 0] = np.count_nonzero(times >= t)
        for cause in (1, 2):
            transitions[j, 0, cause] = np.count_nonzero(
                (times == t) & (events == cause)
            )

    competing = AalenJohansenCompetingRisks().fit(times, events)
    multistate = AalenJohansenMultiState(3).fit(event_times, risk_sets, transitions)
    query = [0, 0.5, 1, 2, 3, 5, np.inf]
    np.testing.assert_allclose(multistate.predict_P(query), competing.predict_P(query))


def test_zero_departures_and_zero_risk_sets_give_identity():
    estimator = AalenJohansenMultiState(np.int64(2)).fit(
        [1, 2], [[0, 0], [3, 0]], np.zeros((2, 2, 2))
    )
    np.testing.assert_array_equal(
        estimator.predict_P([0, 1, 2]), np.tile(np.eye(2), (3, 1, 1))
    )


def test_single_state_and_empty_schedule():
    estimator = AalenJohansenMultiState(1).fit([0], [[5]], [[[0]]])
    np.testing.assert_array_equal(estimator.predict_P(np.inf), [[1]])
    empty = AalenJohansenMultiState(2).fit([], np.empty((0, 2)), np.empty((0, 2, 2)))
    np.testing.assert_array_equal(empty.predict_P(0), np.eye(2))
    np.testing.assert_array_equal(empty.predict_P(np.inf), np.eye(2))


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (0,), (2, 0)])
def test_prediction_shapes(shape):
    estimator = AalenJohansenMultiState(2).fit([1], [[2, 0]], [[[0, 1], [0, 0]]])
    assert estimator.predict_P(np.full(shape, 1.0)).shape == shape + (2, 2)


def test_fitted_model_does_not_alias_input_arrays():
    times = np.array([1.0])
    risks = np.array([[2.0, 0]])
    transitions = np.array([[[0.0, 1], [0, 0]]])
    estimator = AalenJohansenMultiState(2).fit(times, risks, transitions)
    times[0], risks[0, 0], transitions[0, 0, 1] = 5, 10, 0
    np.testing.assert_allclose(estimator.predict_P(1)[0], [0.5, 0.5])


def test_prediction_requires_fit_and_valid_times():
    estimator = AalenJohansenMultiState(2)
    with pytest.raises(RuntimeError, match="fit"):
        estimator.predict_P(0)
    estimator.fit([1], [[2, 0]], [[[0, 1], [0, 0]]])
    for t in (-1, -np.inf, np.nan, [0, np.nan]):
        with pytest.raises(ValueError, match="Prediction times"):
            estimator.predict_P(t)


@pytest.mark.parametrize("n_states", [0, -1, 1.5, 2.0, True, np.bool_(True), np.nan])
def test_invalid_state_count(n_states):
    with pytest.raises(ValueError, match="positive integer"):
        AalenJohansenMultiState(n_states)


@pytest.mark.parametrize(
    "times, risks, transitions, message",
    [
        ([[1]], [[2, 0]], [[[0, 1], [0, 0]]], "event_times"),
        ([-1], [[2, 0]], [[[0, 1], [0, 0]]], "event_times"),
        ([np.nan], [[2, 0]], [[[0, 1], [0, 0]]], "event_times"),
        ([np.inf], [[2, 0]], [[[0, 1], [0, 0]]], "event_times"),
        ([2, 1], [[2, 0]] * 2, [[[0, 1], [0, 0]]] * 2, "event_times"),
        ([1, 1], [[2, 0]] * 2, [[[0, 1], [0, 0]]] * 2, "event_times"),
        ([1], [[2]], [[[0, 1], [0, 0]]], "risk_sets.*shape"),
        ([1], [[2, 0]], [[0, 1], [0, 0]], "transitions.*shape"),
        ([1], [[-1, 0]], [[[0, 0], [0, 0]]], "risk_sets.*non-negative"),
        ([1], [[np.nan, 0]], [[[0, 0], [0, 0]]], "risk_sets.*finite"),
        ([1], [[np.inf, 0]], [[[0, 0], [0, 0]]], "risk_sets.*finite"),
        ([1], [[2, 0]], [[[0, -1], [0, 0]]], "transitions.*non-negative"),
        ([1], [[2, 0]], [[[0, np.nan], [0, 0]]], "transitions.*finite"),
        ([1], [[2, 0]], [[[0, np.inf], [0, 0]]], "transitions.*finite"),
        ([1], [[2, 0]], [[[1, 1], [0, 0]]], "zero diagonal"),
        ([1], [[2, 0]], [[[-1, 1], [0, 0]]], "transitions.*non-negative"),
        ([1], [[2, 0]], [[[0, 3], [0, 0]]], "departures"),
        ([1], [[0, 0]], [[[0, 1], [0, 0]]], "departures"),
    ],
)
def test_invalid_aggregated_data(times, risks, transitions, message):
    with pytest.raises(ValueError, match=message):
        AalenJohansenMultiState(2).fit(times, risks, transitions)
