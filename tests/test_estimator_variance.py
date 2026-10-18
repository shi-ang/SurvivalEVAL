import warnings
from unittest.mock import patch

import numpy as np
import pytest

from SurvivalEVAL.NonparametricEstimator.CompetingRisks import (
    AalenJohansenCompetingRisks,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier


def test_km_greenwood_variance_shapes_sides_and_step_tail() -> None:
    km = KaplanMeier(np.array([1, 1, 2, 4]), np.array([1, 0, 1, 0]))
    times = np.array([[np.inf, 0, 2], [1, 10, 2]])
    expected = [[21 / 256, 0, 21 / 256], [3 / 64, 21 / 256, 21 / 256]]
    np.testing.assert_allclose(km.predict_variance(times), expected)
    assert isinstance(km.predict_variance(2), float)
    assert km.predict_variance(2) == pytest.approx(21 / 256)
    assert km.predict_variance(2, side="left") == pytest.approx(3 / 64)
    assert km.predict_variance(1, side="left") == 0
    assert km.predict_variance(np.array([])).shape == (0,)


def test_reverse_km_variance_uses_events_first_risk_sets() -> None:
    km = KaplanMeier(np.array([1, 1, 2, 4]), np.array([1, 0, 1, 0]), reverse=True)
    # At time 1, an original event leaves before censoring: Y_censor = 3.
    np.testing.assert_allclose(
        km.predict_variance([0, 1, 2, 4, np.inf]), [0, 2 / 27, 2 / 27, 0, 0]
    )
    assert km.predict_variance(4, side="left") == pytest.approx(2 / 27)
    # A terminal original-event-only group has zero reverse risk and no update.
    terminal_event = KaplanMeier(np.array([1, 2]), np.array([0, 1]), reverse=True)
    assert terminal_event.predict_variance(2) == pytest.approx(0.125)


def test_aj_variances_match_r_survival_for_every_cause() -> None:
    aj = AalenJohansenCompetingRisks(3).fit([1, 1, 2, 4], [2, 0, 1, 1])
    # R survival 3.8.12: survfit(Surv(time, factor(cause, levels=0:3)) ~ 1).
    expected = np.array(
        [[0, 0, 0], [0, 3 / 64, 0], [21 / 256, 3 / 64, 0], [3 / 64, 3 / 64, 0]]
    )
    times = np.array([[0, 1], [2, 4]])
    result = aj.predict_cif_variance(times)
    assert result.shape == (2, 2, 3)
    np.testing.assert_allclose(result, expected.reshape(2, 2, 3))
    np.testing.assert_allclose(aj.predict_cif_variance(100), expected[-1])
    np.testing.assert_allclose(aj.predict_cif_variance(np.inf), expected[-1])
    assert aj.predict_cif_variance(2).shape == (3,)
    assert aj.predict_cif_variance(np.array([])).shape == (0, 3)


@pytest.mark.parametrize("events", [[2, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
def test_aj_variance_batch_preserves_repeated_steps_and_query_order(
    events: list,
) -> None:
    aj = AalenJohansenCompetingRisks(3).fit([1, 1, 2, 4], events)
    queries = np.array(
        [[[4, 1, 0], [2, 1.5, np.inf]], [[0.5, 4.5, 2.5], [2, 4, 1]]]
    )
    expected = np.array([aj.predict_cif_variance(t) for t in queries.ravel()])
    np.testing.assert_array_equal(
        aj.predict_cif_variance(queries), expected.reshape(queries.shape + (3,))
    )


@pytest.mark.parametrize("events", [[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
def test_one_cause_aj_variance_reduces_to_km(events: list) -> None:
    times = np.array([0, 1, 1, 2])
    queries = np.array([[0, 1, 2], [3, np.inf, 1]])
    km = KaplanMeier(times, np.array(events))
    aj = AalenJohansenCompetingRisks(1).fit(times, events)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = aj.predict_cif_variance(queries)[..., 0]
        expected = km.predict_variance(queries)
    np.testing.assert_allclose(actual, expected, atol=1e-16)


@pytest.mark.parametrize("events", [[0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 1, 2]])
def test_aj_terminal_ties_and_all_censored_data(events: list) -> None:
    aj = AalenJohansenCompetingRisks(2).fit([1, 1, 1, 1], events)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = aj.predict_cif_variance([0, 1, 2])
    probabilities = np.array([events.count(1), events.count(2)]) / 4
    expected = probabilities * (1 - probabilities) / 4
    np.testing.assert_allclose(result, [[0, 0], expected, expected])


def test_aj_refitting_replaces_variance_inputs() -> None:
    aj = AalenJohansenCompetingRisks().fit([1, 1, 2, 4], [2, 0, 1, 1])
    assert aj.predict_cif_variance(2).shape == (2,)
    aj.fit([1, 2, 3, 4], [1, 0, 0, 0])
    np.testing.assert_allclose(
        aj.predict_cif_variance([0, 1, 4]), [[0], [3 / 64], [3 / 64]]
    )


@pytest.mark.parametrize("n_samples", [5, 7, 11])
def test_terminal_single_cause_variance_is_exactly_zero(n_samples: int) -> None:
    times, events = np.arange(n_samples), np.ones(n_samples)
    aj = AalenJohansenCompetingRisks(2).fit(times, events)
    km = KaplanMeier(times, events)
    np.testing.assert_array_equal(aj.predict_cif_variance(n_samples), [0, 0])
    assert km.predict_variance(n_samples) == 0


def test_variance_queries_reuse_fitted_counts() -> None:
    times = np.array([1, 1, 2, 4])
    events = np.array([1, 0, 1, 0])
    km = KaplanMeier(times, events)
    aj = AalenJohansenCompetingRisks(2).fit(times, events)
    with (
        patch(
            "SurvivalEVAL.NonparametricEstimator.SingleEvent.KaplanMeier._compute_event_counts",
            side_effect=AssertionError("recount"),
        ),
        patch(
            "SurvivalEVAL.NonparametricEstimator.CompetingRisks.AalenJohansen.np.sort",
            side_effect=AssertionError("resort"),
        ),
    ):
        np.testing.assert_allclose(km.predict_variance([1, 2]), [3 / 64, 21 / 256])
        np.testing.assert_allclose(
            aj.predict_cif_variance([1, 2]), [[3 / 64, 0], [21 / 256, 0]]
        )


def test_readonly_inputs_and_repeated_queries_are_unchanged() -> None:
    times = np.repeat([1.0, 1, 2, 4], 2)[::2]
    events = np.repeat([2, 0, 1, 1], 2)[::2]
    times.flags.writeable = events.flags.writeable = False
    aj = AalenJohansenCompetingRisks(2).fit(times, events)
    queries = np.repeat([2.0, 1, 2, np.inf], 2)[::2]
    queries.flags.writeable = False
    cifs, survival = aj.cif_.copy(), aj.surv_.copy()
    first = aj.predict_cif_variance(queries)
    np.testing.assert_array_equal(aj.predict_cif_variance(queries), first)
    np.testing.assert_array_equal(aj.cif_, cifs)
    np.testing.assert_array_equal(aj.surv_, survival)
    np.testing.assert_array_equal(times, [1, 1, 2, 4])
    np.testing.assert_array_equal(events, [2, 0, 1, 1])
    np.testing.assert_array_equal(queries, [2, 1, 2, np.inf])


@pytest.mark.parametrize("target", [-1, np.nan, [0, -1], [0, np.nan]])
def test_variance_rejects_invalid_times(target: object) -> None:
    km = KaplanMeier(np.array([1, 2]), np.array([1, 0]))
    aj = AalenJohansenCompetingRisks().fit([1, 2], [1, 0])
    with pytest.raises(ValueError):
        km.predict_variance(target)
    with pytest.raises(ValueError):
        aj.predict_cif_variance(target)


def test_km_variance_rejects_invalid_side() -> None:
    km = KaplanMeier(np.array([1, 2]), np.array([1, 0]))
    with pytest.raises(ValueError, match="side"):
        km.predict_variance(1, side="invalid")


def test_aj_variance_requires_fit() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        AalenJohansenCompetingRisks().predict_cif_variance(1)
