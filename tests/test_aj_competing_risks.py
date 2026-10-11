import numpy as np
import pytest
from lifelines import AalenJohansenFitter, KaplanMeierFitter

from SurvivalEVAL.NonparametricEstimator.CompetingRisks import (
    AalenJohansenCompetingRisks,
)


@pytest.fixture
def no_cross_ties():
    """Disjoint cause timelines allow a deterministic comparison with lifelines."""
    rng = np.random.default_rng(12345)
    times = np.r_[
        rng.choice([1.0, 2.0, 3.0, 4.0], 66),
        rng.choice([1.5, 2.5, 3.5, 4.5], 66),
        rng.uniform(0.5, 5.0, 68),
    ]
    events = np.repeat([1, 2, 0], [66, 66, 68])
    order = rng.permutation(times.size)
    return times[order], events[order]


@pytest.mark.parametrize("cause", [1, 2])
def test_aj_matches_lifelines_no_cross_ties(no_cross_ties, cause):
    times, events = no_cross_ties
    reference = AalenJohansenFitter(jitter_level=0.0, calculate_variance=False)
    reference.fit(times, events, event_of_interest=cause)
    timeline = reference.cumulative_density_.index.to_numpy()
    estimator = AalenJohansenCompetingRisks().fit(times, events)

    np.testing.assert_allclose(
        estimator.predict_cif(timeline)[:, cause - 1],
        reference.cumulative_density_.iloc[:, 0],
        rtol=1e-12,
        atol=1e-12,
    )
    km = KaplanMeierFitter().fit(times, events > 0)
    np.testing.assert_allclose(estimator.predict_surv(timeline), km.predict(timeline))


def test_cross_cause_and_censoring_ties():
    # At time 1 all five subjects are at risk, including the tied censoring.
    estimator = AalenJohansenCompetingRisks().fit([1, 1, 1, 2, 3], [1, 2, 0, 2, 0])
    query = [0, np.nextafter(1.0, 0.0), 1, 1.5, 2, np.inf]
    np.testing.assert_allclose(
        estimator.predict_surv(query), [1, 1, 0.6, 0.6, 0.3, 0.3]
    )
    np.testing.assert_allclose(
        estimator.predict_cif(query),
        [[0, 0], [0, 0], [0.2, 0.2], [0.2, 0.2], [0.2, 0.5], [0.2, 0.5]],
    )

    matrices = estimator.predict_P(query)
    np.testing.assert_allclose(matrices[0], np.eye(3))
    np.testing.assert_allclose(matrices.sum(axis=-1), 1)
    np.testing.assert_allclose(matrices[:, 1:, :], np.tile(np.eye(3)[1:], (6, 1, 1)))


def test_events_at_zero_are_included():
    estimator = AalenJohansenCompetingRisks().fit([0, 0, 1], [1, 0, 2])
    assert estimator.predict_surv(0) == pytest.approx(2 / 3)
    np.testing.assert_allclose(estimator.predict_cif(0), [1 / 3, 0])
    np.testing.assert_allclose(estimator.predict_cif(1), [1 / 3, 2 / 3])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grouped_estimates_match_direct_recursion(seed):
    rng = np.random.default_rng(seed)
    times = rng.integers(0, 30, 150)
    events = rng.integers(0, 4, times.size)
    event_times = np.unique(times[events > 0])
    survival = 1.0
    cif = np.zeros(4)
    expected = []
    for t in event_times:
        risk = np.count_nonzero(times >= t)
        for cause in range(1, 5):
            cif[cause - 1] += (
                survival * np.count_nonzero((times == t) & (events == cause)) / risk
            )
        survival *= 1 - np.count_nonzero((times == t) & (events > 0)) / risk
        expected.append(np.r_[survival, cif])

    times.flags.writeable = False
    events.flags.writeable = False
    estimator = AalenJohansenCompetingRisks(n_causes=4).fit(times, events)
    actual = estimator.predict_P(event_times)[:, 0, :]
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=1), 1)
    assert np.all(np.diff(estimator.surv_) <= 0)
    assert np.all(np.diff(estimator.cif_, axis=1) >= 0)
    np.testing.assert_array_equal(estimator.cif_[3], 0)


def test_one_cause_reduces_to_kaplan_meier():
    times, events = [1, 1, 2, 3, 4], [1, 0, 1, 1, 0]
    estimator = AalenJohansenCompetingRisks().fit(times, events)
    km = KaplanMeierFitter().fit(times, events)
    query = [0, 1, 1.5, 2, 3, 4, np.inf]
    np.testing.assert_allclose(estimator.predict_surv(query), km.predict(query))
    np.testing.assert_allclose(
        estimator.predict_surv(query) + estimator.predict_cif(query)[:, 0], 1
    )


def test_censoring_only_requires_known_causes():
    with pytest.raises(ValueError, match="Specify n_causes"):
        AalenJohansenCompetingRisks().fit([1, 2], [0, 0])
    estimator = AalenJohansenCompetingRisks(n_causes=2).fit([1, 2], [0, 0])
    assert estimator.unique_times_.size == 0
    np.testing.assert_array_equal(estimator.predict_surv([0, 3, np.inf]), 1)
    np.testing.assert_array_equal(estimator.predict_cif([0, 3, np.inf]), 0)
    np.testing.assert_array_equal(estimator.predict_P(3), np.eye(3))


def test_refit_infers_causes_from_current_data():
    estimator = AalenJohansenCompetingRisks().fit([1, 2], [1, 0])
    estimator.fit([1, 2], [2, 0])
    assert estimator.n_causes_ == 2
    np.testing.assert_allclose(estimator.predict_cif(1), [0, 0.5])
    estimator.fit([1, 2], [1, 0])
    assert estimator.n_causes_ == 1
    assert estimator.predict_P(1).shape == (2, 2)


@pytest.mark.parametrize("shape", [(), (3,), (2, 3), (0,), (2, 0)])
def test_prediction_shapes(shape):
    estimator = AalenJohansenCompetingRisks(n_causes=2).fit([1, 2], [1, 2])
    query = np.full(shape, 1.0)
    assert np.shape(estimator.predict_surv(query)) == shape
    assert estimator.predict_cif(query).shape == shape + (2,)
    assert estimator.predict_P(query).shape == shape + (3, 3)
    if query.size:
        np.testing.assert_allclose(
            estimator.predict_cif(query), np.tile([0.5, 0], shape + (1,))
        )


@pytest.mark.parametrize("method", ["predict_surv", "predict_cif", "predict_P"])
def test_prediction_requires_fit_and_valid_times(method):
    estimator = AalenJohansenCompetingRisks()
    with pytest.raises(RuntimeError, match="fit"):
        getattr(estimator, method)(0)
    estimator.fit([1, 2], [1, 2])
    for t in (-1, -np.inf, np.nan, [0, np.nan]):
        with pytest.raises(ValueError, match="Prediction times"):
            getattr(estimator, method)(t)


@pytest.mark.parametrize("n_causes", [0, -1, 1.5, 2.0, True, np.bool_(True), np.nan])
def test_invalid_cause_count(n_causes):
    with pytest.raises(ValueError, match="positive integer"):
        AalenJohansenCompetingRisks(n_causes)


@pytest.mark.parametrize(
    "times, events",
    [
        ([], []),
        ([[1, 2]], [[1, 2]]),
        ([1, 2], [1]),
        ([1, 2], [[1, 2]]),
        ([-1, 2], [1, 2]),
        ([np.nan, 2], [1, 2]),
        ([np.inf, 2], [1, 2]),
        ([1, 2], [-1, 2]),
        ([1, 2], [0.5, 2]),
        ([1, 2], [np.nan, 2]),
        ([1, 2], [np.inf, 2]),
        ([1, 2], [1, 3]),
    ],
)
def test_invalid_observations(times, events):
    with pytest.raises(ValueError):
        AalenJohansenCompetingRisks(n_causes=2).fit(times, events)
