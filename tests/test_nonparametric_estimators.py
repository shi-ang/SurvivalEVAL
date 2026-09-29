import numpy as np
import pytest

from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    CopulaGraphic,
    KaplanMeier,
    KaplanMeierArea,
    NelsonAalen,
    TurnbullEstimator,
)


@pytest.mark.parametrize(
    "estimator_class, kwargs",
    [
        (KaplanMeier, {}),
        (NelsonAalen, {}),
        (CopulaGraphic, {"alpha": 2.0, "type": "Clayton"}),
        (CopulaGraphic, {"alpha": 2.0, "type": "Gumbel"}),
        (CopulaGraphic, {"alpha": 2.0, "type": "Frank"}),
    ],
)
@pytest.mark.parametrize("indicator_dtype", [bool, np.int8, np.float32, np.float64])
def test_estimators_group_tied_times_without_changing_risk_sets(
    estimator_class, kwargs, indicator_dtype
):
    rng = np.random.default_rng(12)
    for origin in (0.0, 2.0):
        times = rng.integers(0, 40, size=300).astype(float) + origin
        indicators = (rng.random(300) < 0.7).astype(indicator_dtype)
        unique_times = np.unique(times)
        expected_population = np.array([(times >= t).sum() for t in unique_times])
        expected_events = np.array([indicators[times == t].sum() for t in unique_times])
        if origin > 0:
            unique_times = np.r_[0.0, unique_times]
            expected_population = np.r_[times.size, expected_population]
            expected_events = np.r_[0, expected_events]

        times.flags.writeable = False
        indicators.flags.writeable = False
        estimator = estimator_class(times, indicators, **kwargs)
        np.testing.assert_array_equal(estimator.survival_times, unique_times)
        np.testing.assert_array_equal(estimator.population_count, expected_population)
        np.testing.assert_array_equal(estimator.events, expected_events)
        if estimator_class is KaplanMeier:
            np.testing.assert_allclose(
                estimator.survival_probabilities,
                np.cumprod(1 - expected_events / expected_population),
            )
        elif estimator_class is NelsonAalen:
            np.testing.assert_allclose(
                estimator.cumulative_hazard,
                np.cumsum(expected_events / expected_population),
            )


@pytest.mark.parametrize("indicators", [[0], [1], [0, 0, 0], [1, 1, 1], [1, 0, 1]])
@pytest.mark.parametrize("estimator_class", [KaplanMeier, NelsonAalen])
def test_estimators_with_one_observation_time(estimator_class, indicators):
    indicators = np.asarray(indicators, dtype=bool)
    estimator = estimator_class(np.full(indicators.size, 2.0), indicators)
    np.testing.assert_array_equal(estimator.survival_times, [0.0, 2.0])
    np.testing.assert_array_equal(
        estimator.population_count, [indicators.size, indicators.size]
    )
    np.testing.assert_array_equal(estimator.events, [0, indicators.sum()])


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


def test_turnbull_estimator_includes_exact_observations():
    estimator = TurnbullEstimator().fit(
        left=np.array([1.0, 2.0]),
        right=np.array([1.0, 3.0]),
    )

    np.testing.assert_allclose(estimator.probability_dens_, [0.0, 0.5, 0.5])
    np.testing.assert_allclose(estimator.survival_times_, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        estimator.survival_probabilities_,
        [1.0, 0.5, 0.5, 0.0],
    )
    assert estimator.predict(1.0) == 0.5


def test_turnbull_estimator_handles_only_exact_observations():
    estimator = TurnbullEstimator().fit(
        left=np.array([1.0, 2.0]),
        right=np.array([1.0, 2.0]),
    )

    np.testing.assert_allclose(estimator.survival_times_, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        estimator.survival_probabilities_,
        [1.0, 0.5, 0.0],
    )


def test_turnbull_estimator_uses_left_open_interval_bounds():
    estimator = TurnbullEstimator().fit(
        left=np.array([0.0, 1.0]),
        right=np.array([1.0, 2.0]),
    )

    np.testing.assert_allclose(estimator.probability_dens_, [0.5, 0.5])
    np.testing.assert_allclose(estimator.survival_times_, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        estimator.survival_probabilities_,
        [1.0, 0.5, 0.0],
    )


def test_turnbull_estimator_keeps_infinite_tail_support():
    estimator = TurnbullEstimator().fit(
        left=np.array([1.0, 2.0]),
        right=np.array([1.0, np.inf]),
    )

    assert np.isinf(estimator.tau_[-1])
    np.testing.assert_allclose(estimator.probability_dens_, [0.0, 0.5, 0.5])
    np.testing.assert_allclose(estimator.survival_times_, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(estimator.survival_probabilities_, [1.0, 0.5, 0.5])


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
