import numpy as np

from SurvivalEVAL import PointEvaluator
from SurvivalEVAL.Evaluations.Concordance import _get_comparable_ic, concordance


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
