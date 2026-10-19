from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import trapezoid

from SurvivalEVAL import (
    CompetingRisksEvaluator,
    SurvivalEvaluator,
    aj_calibration,
    aj_calibration_per_event,
)


@pytest.fixture
def calibration_data():
    # At t=1, two events and a censoring share a risk set of five. At t=2,
    # survival before the update is .6, with two subjects still at risk.
    times = np.array([0, 1, 2, 3], dtype=float)
    durations = np.array([1, 1, 1, 2, 3], dtype=float)
    events = np.array([1, 2, 0, 2, 0])
    cifs = np.tile([[0, 0.1, 0.15, 0.2], [0, 0.1, 0.25, 0.3]], (5, 1, 1))
    return cifs, times, durations, events


@pytest.mark.parametrize(
    "alpha, expected", [(1, [0.05, 0.15]), (2, [1 / 240, 37 / 1200])]
)
def test_hand_computed_scores_with_tied_events_and_censoring(
    calibration_data,
    alpha,
    expected,
):
    cifs, times, durations, events = calibration_data
    score, details = aj_calibration(*calibration_data, alpha=alpha, return_details=True)
    np.testing.assert_allclose(
        details["aj_cifs"], [[0, 0.2, 0.2, 0.2], [0, 0.2, 0.5, 0.5]]
    )
    np.testing.assert_allclose(details["mean_pred_cifs"], cifs.mean(axis=0))
    np.testing.assert_array_equal(details["time_coordinates"], times)
    np.testing.assert_allclose(list(details["per_event_scores"].values()), expected)
    assert score == pytest.approx(np.mean(expected))
    for k in (1, 2):
        per_score, per_details = aj_calibration_per_event(
            cifs[:, k - 1],
            times,
            durations,
            events,
            k,
            alpha=alpha,
            return_details=True,
        )
        assert per_score == pytest.approx(expected[k - 1])
        np.testing.assert_allclose(per_details["aj_cifs"], details["aj_cifs"][k - 1])


def _reference_cifs(durations, events, times, n_causes):
    """Compute the AJ recurrence explicitly, including all competing events."""
    cumulative = np.zeros(n_causes)
    survival = 1.0
    updates = []
    for t in np.unique(durations[events > 0]):
        risk = np.count_nonzero(durations >= t)
        counts = np.array(
            [
                np.count_nonzero((durations == t) & (events == k))
                for k in range(1, n_causes + 1)
            ]
        )
        cumulative += survival * counts / risk
        survival *= 1 - counts.sum() / risk
        updates.append((t, cumulative.copy()))
    return np.array(
        [
            next((cif for t, cif in reversed(updates) if t <= time), np.zeros(n_causes))
            for time in times
        ]
    ).T


@pytest.mark.parametrize("alpha", [1, 2, 3.5])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_direct_reference_on_irregular_grid(alpha, dtype):
    rng = np.random.default_rng(102)
    times = np.array([0, 0.2, 1, 1.3, 2.2, 4, 7], dtype=dtype)
    durations = rng.integers(0, 8, 80).astype(dtype)
    events = rng.integers(0, 4, 80)
    increments = rng.uniform(0, 0.02, (80, 3, times.size)).astype(dtype)
    increments[:, :, 0] = 0
    cifs = np.cumsum(increments, axis=2)
    cifs.flags.writeable = False
    times.flags.writeable = False
    durations.flags.writeable = False
    events.flags.writeable = False
    score, details = aj_calibration(
        cifs,
        times,
        durations,
        events,
        alpha=alpha,
        min_prop_at_risk=0,
        return_details=True,
    )
    reference = _reference_cifs(durations, events, times, 3)
    expected = (
        trapezoid(
            np.abs(cifs.mean(axis=0, dtype=float) - reference) ** alpha, times, axis=1
        )
        / times[-1]
    )
    np.testing.assert_allclose(details["aj_cifs"], reference)
    np.testing.assert_allclose(
        list(details["per_event_scores"].values()),
        expected,
        rtol=2e-6 if dtype == np.float32 else 1e-12,
    )
    assert score == pytest.approx(expected.mean(), rel=2e-6)


@pytest.mark.parametrize("reduction", ["mean", "sum", "max"])
def test_aggregation(calibration_data, reduction):
    scores = np.array([1 / 240, 37 / 1200])
    expected = {"mean": np.mean, "sum": np.sum, "max": np.max}[reduction.lower()](
        scores
    )
    assert aj_calibration(*calibration_data, reduction=reduction) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "threshold, grid, expected",
    [
        (0.4, [0, 1, 2], [0.005625, 0.020625]),
        (0.6, [0, 1], [0.005, 0.005]),
        (1, [0, 1], [0.005, 0.005]),
    ],
)
def test_at_risk_truncation_includes_ties_and_threshold_equality(
    calibration_data,
    threshold,
    grid,
    expected,
):
    score, details = aj_calibration(
        *calibration_data, min_prop_at_risk=threshold, return_details=True
    )
    np.testing.assert_array_equal(details["time_coordinates"], grid)
    np.testing.assert_allclose(list(details["per_event_scores"].values()), expected)
    assert score == pytest.approx(np.mean(expected))


def test_missing_zero_is_prepended_without_mutating_inputs(calibration_data):
    cifs, times, durations, events = calibration_data
    cifs, times = cifs[:, :, 1:], times[1:]
    before = cifs.copy()
    cifs.flags.writeable = False
    times.flags.writeable = False
    assert aj_calibration(cifs, times, durations, events) == pytest.approx(
        aj_calibration(*calibration_data)
    )
    np.testing.assert_array_equal(cifs, before)
    np.testing.assert_array_equal(times, [1, 2, 3])


def test_all_censored_and_unobserved_causes_are_scored():
    cifs = np.tile([[0, 0.2, 0.4], [0, 0.1, 0.2]], (4, 1, 1))
    score, details = aj_calibration(
        cifs, [0, 1, 2], [2] * 4, [0] * 4, return_details=True
    )
    np.testing.assert_array_equal(details["aj_cifs"], np.zeros((2, 3)))
    assert details["per_event_scores"] == pytest.approx({1: 0.06, 2: 0.015})
    assert score == pytest.approx(0.0375)
    # The selected cause can be absent even when a differently numbered cause occurs.
    assert aj_calibration_per_event(
        cifs[:, 0], [0, 1, 2], [2] * 4, [3] * 4, 1
    ) == pytest.approx(0.06)


def test_unobserved_predicted_cause_is_not_dropped(calibration_data):
    cifs, times, durations, events = calibration_data
    extra = np.broadcast_to([0, 0.05, 0.1, 0.15], (5, 1, 4))
    score, details = aj_calibration(
        np.concatenate((cifs, extra), axis=1),
        times,
        durations,
        events,
        return_details=True,
    )
    assert set(details["per_event_scores"]) == {1, 2, 3}
    np.testing.assert_array_equal(details["aj_cifs"][2], np.zeros(4))
    assert details["per_event_scores"][3] == pytest.approx(19 / 2400)
    assert score == pytest.approx(np.mean(list(details["per_event_scores"].values())))


def test_predictions_matching_aj_have_zero_score_even_with_events_at_zero():
    cifs = np.tile([[1 / 3, 1 / 3, 1 / 3], [0, 1 / 3, 1 / 3]], (3, 1, 1))
    assert aj_calibration(cifs, [0, 1, 2], [0, 1, 2], [1, 2, 0]) == pytest.approx(
        0, abs=1e-30
    )


def test_time_units_and_individual_variation_do_not_change_score(calibration_data):
    cifs, times, durations, events = calibration_data
    expected = aj_calibration(*calibration_data)
    assert aj_calibration(cifs, times * 365, durations * 365, events) == pytest.approx(
        expected
    )
    cifs[0] *= 0.5
    cifs[1] *= 1.5
    assert aj_calibration(cifs, times, durations, events) == pytest.approx(expected)


def test_list_inputs(calibration_data):
    assert aj_calibration(
        *(item.tolist() for item in calibration_data)
    ) == pytest.approx(aj_calibration(*calibration_data))


@pytest.mark.parametrize("alpha", [0, -1, np.nan, np.inf, [2]])
def test_invalid_alpha(calibration_data, alpha):
    with pytest.raises(ValueError, match="alpha"):
        aj_calibration(*calibration_data, alpha=alpha)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, np.inf, [0.05]])
def test_invalid_at_risk_threshold(calibration_data, threshold):
    with pytest.raises(ValueError, match="min_prop_at_risk"):
        aj_calibration(*calibration_data, min_prop_at_risk=threshold)


@pytest.mark.parametrize("cause", [None, True, 0, -1, 1.5])
def test_invalid_selected_cause(calibration_data, cause):
    cifs, times, durations, events = calibration_data
    with pytest.raises(ValueError, match="event_of_interest"):
        aj_calibration_per_event(cifs[:, 0], times, durations, events, cause)


@pytest.mark.parametrize(
    "index, value, message",
    [
        (0, np.zeros((5, 4)), "shape"),
        (0, np.zeros((4, 2, 4)), "sample"),
        (0, np.zeros((5, 2, 3)), "time axis"),
        (0, np.full((5, 2, 4), np.inf), "finite probabilities"),
        (0, np.full((5, 2, 4), -0.1), "probabilities"),
        (0, np.full((5, 2, 4), 1.1), "probabilities"),
        (0, np.tile([[0, 0.1, 0.05, 0.2]] * 2, (5, 1, 1)), "nondecreasing"),
        (0, np.full((5, 2, 4), 0.6), "sum to at most one"),
        (1, [[0, 1, 2, 3]], "1D"),
        (1, [0], "at least two"),
        (1, [0, 2, 1, 3], "strictly increasing"),
        (1, [0, 1, 1, 3], "strictly increasing"),
        (1, [-1, 1, 2, 3], "non-negative"),
        (1, [0, 1, 2, np.inf], "finite"),
        (2, [1, 1, 1, 2], "same shape"),
        (2, [-1, 1, 1, 2, 3], "non-negative"),
        (2, [np.inf, 1, 1, 2, 3], "finite"),
        (3, [1, 2, 0, 2, 3], "n_causes"),
        (3, [1, 2, 0, 2, -1], "integer labels"),
        (3, [1, 2, 0, 2, 0.5], "integer labels"),
        (3, [1, 2, 0, 2, np.inf], "integer labels"),
    ],
)
def test_invalid_inputs(calibration_data, index, value, message):
    args = list(calibration_data)
    args[index] = value
    with pytest.raises(ValueError, match=message):
        aj_calibration(*args)


def test_invalid_reduction(calibration_data):
    with pytest.raises(ValueError, match="reduction"):
        aj_calibration(*calibration_data, reduction="median")


def test_no_positive_integration_interval_after_truncation():
    with pytest.raises(ValueError, match="At least two"):
        aj_calibration(np.zeros((3, 1, 3)), [0, 1, 2], [0] * 3, [0] * 3)


def test_per_event_requires_one_cause_matrix(calibration_data):
    with pytest.raises(ValueError, match="Per-event CIFs"):
        aj_calibration_per_event(*calibration_data, event_of_interest=1)


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("alpha", [1, 2])
@pytest.mark.parametrize("reduction", ["mean", "sum", "max"])
def test_evaluator_matches_cause_scores_and_aggregation(
    calibration_data: tuple,
    interpolation: str,
    alpha: float,
    reduction: str,
) -> None:
    evaluator = CompetingRisksEvaluator(*calibration_data, interpolation=interpolation)
    expected = [0.05, 0.15] if alpha == 1 else [1 / 240, 37 / 1200]
    score, details = evaluator.aj_calibration(
        alpha=alpha, reduction=reduction, return_details=True
    )
    aggregate = {"mean": np.mean, "sum": np.sum, "max": np.max}[reduction](expected)
    assert score == pytest.approx(aggregate)
    assert evaluator.aj_calibration(alpha=alpha, reduction=reduction) == pytest.approx(
        score
    )
    assert details["per_event_scores"] == pytest.approx(dict(enumerate(expected, 1)))
    np.testing.assert_array_equal(details["time_coordinates"], calibration_data[1])
    for cause, value in enumerate(expected, 1):
        assert evaluator.aj_calibration_per_event(cause, alpha=alpha) == pytest.approx(
            value
        )


def test_evaluator_plot_matches_the_retained_cause_curves(
    calibration_data: tuple,
) -> None:
    evaluator = CompetingRisksEvaluator(*calibration_data)
    with patch("matplotlib.pyplot.show") as show:
        score, (fig, ax) = evaluator.aj_calibration_per_event(
            2, min_prop_at_risk=0.4, draw_figure=True
        )
    try:
        assert score == pytest.approx(0.020625)
        for line in ax.lines:
            np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), [0, 0.1, 0.25])
        np.testing.assert_allclose(ax.lines[1].get_ydata(), [0, 0.2, 0.5])
        assert ax.get_xlabel() == "Time"
        assert ax.get_ylabel() == "Cumulative incidence"
        show.assert_not_called()
    finally:
        plt.close(fig)


def test_one_cause_evaluator_matches_km_on_the_same_event_grid() -> None:
    grid = np.array([0, 1, 2, 4])
    cifs = np.tile([0, 0.1, 0.3, 0.6], (4, 1, 1))
    times, events = np.array([1, 2, 3, 4]), np.array([1, 1, 0, 1])
    evaluator = CompetingRisksEvaluator(cifs, grid, times, events)
    survival = SurvivalEvaluator(1 - cifs[:, 0], grid, times, events)
    expected = survival.km_calibration()
    assert evaluator.aj_calibration() == pytest.approx(expected)
    assert evaluator.aj_calibration_per_event(1) == pytest.approx(expected)


@pytest.mark.parametrize("cause", [None, True, np.bool_(True), 0, -1, 1.5, 3])
def test_evaluator_rejects_invalid_cause(
    calibration_data: tuple,
    cause: object,
) -> None:
    evaluator = CompetingRisksEvaluator(*calibration_data)
    with pytest.raises(ValueError, match="event_of_interest"):
        evaluator.aj_calibration_per_event(cause)


@pytest.mark.parametrize("cause", [1, 2])
def test_single_cause_pooling_preserves_incidence_for_sparse_labels(
    calibration_data: tuple,
    cause: int,
) -> None:
    cifs, times, durations, events = calibration_data
    labels = np.where(events == cause, 1000000, np.where(events > 0, 500, 0))
    expected, expected_details = aj_calibration_per_event(
        cifs[:, cause - 1], times, durations, events, cause, return_details=True
    )
    actual, details = aj_calibration_per_event(
        cifs[:, cause - 1], times, durations, labels, 1000000, return_details=True
    )
    assert actual == pytest.approx(expected)
    np.testing.assert_allclose(details["aj_cifs"], expected_details["aj_cifs"])
