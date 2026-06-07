import numpy as np
import pytest

from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    CopulaGraphic,
    KaplanMeier,
    KaplanMeierArea,
    NelsonAalen,
    TurnbullEstimator,
)


def test_kaplan_meier_returns_one_before_first_observation():
    estimator = KaplanMeier(
        event_times=np.array([2.0, 3.0]),
        event_indicators=np.array([1, 1]),
    )

    np.testing.assert_allclose(estimator.survival_times, [0.0, 2.0, 3.0])
    np.testing.assert_allclose(estimator.population_count, [2, 2, 1])
    np.testing.assert_allclose(estimator.events, [0, 1, 1])
    np.testing.assert_allclose(estimator.survival_probabilities, [1.0, 0.5, 0.0])
    np.testing.assert_allclose(estimator.cumulative_dens, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(estimator.probability_dens, [0.5, 0.5, 0.0])
    assert estimator.predict(0.0) == 1.0
    np.testing.assert_allclose(
        estimator.predict(np.array([0.0, 1.0, 2.0])),
        np.array([1.0, 1.0, 0.5]),
    )


def test_kaplan_meier_does_not_duplicate_observed_time_zero():
    estimator = KaplanMeier(
        event_times=np.array([0.0, 2.0]),
        event_indicators=np.array([1, 1]),
    )

    np.testing.assert_allclose(estimator.survival_times, [0.0, 2.0])
    assert estimator.predict(0.0) == 0.5


def test_kaplan_meier_area_reuses_estimator_baseline():
    estimator = KaplanMeierArea(
        event_times=np.array([2.0, 3.0]),
        event_indicators=np.array([1, 1]),
    )

    assert np.count_nonzero(estimator.area_times == 0.0) == 1
    assert estimator.area_probabilities[0] == 1.0


@pytest.mark.parametrize("copula_type", ["Clayton", "Gumbel", "Frank"])
def test_copula_graphic_adds_pre_event_baseline(copula_type):
    estimator = CopulaGraphic(
        event_times=np.array([2.0, 3.0]),
        event_indicators=np.array([1, 1]),
        alpha=2.0,
        type=copula_type,
    )

    assert estimator.survival_times[0] == 0.0
    assert estimator.population_count[0] == 2
    assert estimator.events[0] == 0
    assert estimator.survival_probabilities[0] == 1.0
    assert estimator.cumulative_dens[0] == 0.0
    assert estimator.predict(0.0) == 1.0


def test_copula_graphic_does_not_duplicate_observed_time_zero():
    estimator = CopulaGraphic(
        event_times=np.array([0.0, 2.0]),
        event_indicators=np.array([1, 1]),
        alpha=2.0,
        type="Clayton",
    )

    np.testing.assert_allclose(estimator.survival_times, [0.0, 2.0])


def test_turnbull_estimator_keeps_pre_event_baseline():
    estimator = TurnbullEstimator().fit(
        left=np.array([1.0, 2.0]),
        right=np.array([2.0, 3.0]),
    )

    assert estimator.survival_times_[0] == 0.0
    assert estimator.survival_probabilities_[0] == 1.0
    assert estimator.predict(0.0) == 1.0


def test_nelson_aalen_returns_baseline_before_first_observation():
    estimator = NelsonAalen(
        event_times=np.array([2.0, 3.0]),
        event_indicators=np.array([1, 1]),
    )

    np.testing.assert_allclose(estimator.survival_times, [0.0, 2.0, 3.0])
    np.testing.assert_allclose(estimator.population_count, [2, 2, 1])
    np.testing.assert_allclose(estimator.events, [0, 1, 1])
    np.testing.assert_allclose(estimator.hazard, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(estimator.cumulative_hazard, [0.0, 0.5, 1.5])
    np.testing.assert_allclose(
        estimator.survival_probabilities, np.exp(-np.array([0.0, 0.5, 1.5]))
    )
    assert estimator.predict(0.0) == 0.0
    assert estimator.predict_survival(0.0) == 1.0
    np.testing.assert_allclose(
        estimator.predict(np.array([0.0, 1.0, 2.0])),
        np.array([0.0, 0.0, 0.5]),
    )
    np.testing.assert_allclose(
        estimator.predict_survival(np.array([0.0, 1.0, 2.0])),
        np.exp(-np.array([0.0, 0.0, 0.5])),
    )


def test_nelson_aalen_does_not_duplicate_observed_time_zero():
    estimator = NelsonAalen(
        event_times=np.array([0.0, 2.0]),
        event_indicators=np.array([1, 1]),
    )

    np.testing.assert_allclose(estimator.survival_times, [0.0, 2.0])
    assert estimator.predict(0.0) == 0.5


def test_nelson_aalen_rejects_negative_prediction_times():
    estimator = NelsonAalen(
        event_times=np.array([2.0, 3.0]),
        event_indicators=np.array([1, 1]),
    )

    with pytest.raises(ValueError, match="non-negative"):
        estimator.predict(np.array([0.0, -1.0]))
