import numpy as np
import pytest

from SurvivalEVAL.Evaluations.AreaUnderPRCurve import auprc_uncensored_grid
from SurvivalEVAL.Evaluations.util import (
    align_curve_and_time_coordinates,
    check_monotonicity,
    make_monotonic,
    survival_to_quantile,
)


@pytest.mark.parametrize(
    ("curves", "time_coordinates", "expected_curves", "expected_times"),
    [
        (
            np.array([1.0, 0.5, 0.0]),
            np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]),
            np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]]),
            np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]),
        ),
        (
            np.array([[1.0, 0.7, 0.2], [1.0, 0.4, 0.1]]),
            np.array([0.0, 1.0, 2.0]),
            np.array([[1.0, 0.7, 0.2], [1.0, 0.4, 0.1]]),
            np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
        ),
    ],
)
def test_align_curve_and_time_coordinates_broadcasts_shared_inputs(
    curves, time_coordinates, expected_curves, expected_times
):
    aligned_curves, aligned_times = align_curve_and_time_coordinates(
        curves, time_coordinates
    )

    np.testing.assert_allclose(aligned_curves, expected_curves)
    np.testing.assert_allclose(aligned_times, expected_times)


def test_align_curve_and_time_coordinates_accepts_external_sample_count():
    curves, times = align_curve_and_time_coordinates(
        curves=np.array([1.0, 0.5, 0.0]),
        time_coordinates=np.array([0.0, 1.0, 2.0]),
        n_samples=2,
    )

    assert curves.shape == times.shape == (2, 3)


@pytest.mark.parametrize(
    ("values", "direction", "expected"),
    [
        ([1.0, 2.0, 2.0], "increasing", True),
        ([1.0, 2.0, 2.0], "decreasing", False),
        ([2.0, 2.0, 1.0], "increasing", False),
        ([2.0, 2.0, 1.0], "decreasing", True),
    ],
)
def test_check_monotonicity_respects_direction(values, direction, expected):
    assert check_monotonicity(values, direction=direction) is expected


def test_check_monotonicity_checks_each_row_and_preserves_legacy_default():
    increasing = np.array([[0.0, 0.5, 1.0], [1.0, 1.0, 2.0]])

    assert check_monotonicity(increasing)
    assert check_monotonicity(increasing, direction="increasing")
    assert not check_monotonicity(increasing, direction="decreasing")


def test_check_monotonicity_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        check_monotonicity([0.0, 1.0], direction="sideways")


def test_make_monotonic_corrects_increasing_survival_curve():
    survival_curves = np.array([[0.2, 0.5, 0.8]])

    result = make_monotonic(
        survival_curves, np.array([0.0, 1.0, 2.0]), method="floor"
    )

    assert check_monotonicity(result, direction="decreasing")
    np.testing.assert_allclose(result, [[0.2, 0.2, 0.2]])


def test_survival_to_quantile_rejects_increasing_survival_probabilities():
    with pytest.raises(ValueError, match="nonincreasing"):
        survival_to_quantile(
            surv_prob=[[0.2, 0.5, 0.8]],
            time_coordinates=[[0.0, 1.0, 2.0]],
            quantile_levels=[0.25, 0.5],
        )


def test_survival_to_quantile_rejects_decreasing_time_coordinates():
    with pytest.raises(ValueError, match="strictly increasing"):
        survival_to_quantile(
            surv_prob=[[1.0, 0.8, 0.5]],
            time_coordinates=[[2.0, 1.0, 0.0]],
            quantile_levels=[0.25, 0.5],
        )


def test_survival_to_quantile_rejects_decreasing_quantile_levels():
    with pytest.raises(ValueError, match="increasing order"):
        survival_to_quantile(
            surv_prob=[[1.0, 0.8, 0.5]],
            time_coordinates=[[0.0, 1.0, 2.0]],
            quantile_levels=[0.5, 0.25],
        )


def test_auprc_rejects_decreasing_cdf():
    with pytest.raises(AssertionError, match="non-decreasing"):
        auprc_uncensored_grid(
            pred_cdf=np.array([[0.8, 0.5, 0.2]]),
            time_grid=np.array([0.0, 1.0, 2.0]),
            event_times=np.array([1.0]),
        )


def test_auprc_rejects_decreasing_time_grid():
    with pytest.raises(AssertionError, match="time_grid"):
        auprc_uncensored_grid(
            pred_cdf=np.array([[0.2, 0.5, 0.8]]),
            time_grid=np.array([2.0, 1.0, 0.0]),
            event_times=np.array([1.0]),
        )
