import numpy as np
import pytest

from SurvivalEVAL.Evaluations.AreaUnderPRCurve import (
    auprc_ic,
    auprc_right_censored_grid,
    auprc_uncensored_grid,
)
from SurvivalEVAL.Evaluations.util import (
    align_curve_and_time_coordinates,
    check_monotonicity,
    make_monotonic,
    predict_multi_probs_from_curve,
    predict_prob_from_curve,
    survival_to_quantile,
    zero_padding,
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


def test_zero_padding_supports_1d_curve_and_1d_time_coordinates():
    curve = np.array([0.8, 0.5, 0.1])
    time_coordinates = np.array([1.0, 2.0, 3.0])

    with pytest.warns(UserWarning, match="first time coordinate"):
        padded_curve, padded_times = zero_padding(curve, time_coordinates)

    np.testing.assert_allclose(padded_curve, [1.0, 0.8, 0.5, 0.1])
    np.testing.assert_allclose(padded_times, [0.0, 1.0, 2.0, 3.0])
    assert isinstance(padded_curve, np.ndarray)
    assert isinstance(padded_times, np.ndarray)
    assert padded_curve.ndim == 1


@pytest.mark.parametrize(
    ("pred_survs", "time_coordinates", "message"),
    [
        ([0.8, 0.5, 0.1], np.array([1.0, 2.0, 3.0]), "pred_survs"),
        (np.array([0.8, 0.5, 0.1]), [1.0, 2.0, 3.0], "time_coordinates"),
        (
            np.array(["0.8", "0.5", "0.1"]),
            np.array([1.0, 2.0, 3.0]),
            "pred_survs",
        ),
        (
            np.array([0.8, 0.5, 0.1]),
            np.array(["1.0", "2.0", "3.0"]),
            "time_coordinates",
        ),
    ],
)
def test_zero_padding_rejects_invalid_input_datatypes(
    pred_survs, time_coordinates, message
):
    with pytest.raises(TypeError, match=message):
        zero_padding(pred_survs, time_coordinates)


def test_zero_padding_rejects_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        zero_padding(np.array([]), np.array([]))


def test_zero_padding_rejects_negative_time_coordinates():
    with pytest.raises(ValueError, match="non-negative"):
        zero_padding(np.array([1.0, 0.8, 0.5]), np.array([-1.0, 1.0, 2.0]))


def test_zero_padding_preserves_1d_curve_that_already_starts_at_zero():
    curve = np.array([0.8, 0.5, 0.1])
    time_coordinates = np.array([0.0, 2.0, 3.0])

    padded_curve, padded_times = zero_padding(curve, time_coordinates)

    np.testing.assert_allclose(padded_curve, curve)
    np.testing.assert_allclose(padded_times, time_coordinates)


@pytest.mark.parametrize(
    "curves",
    [
        np.array([1.0, 0.6, 0.2]),
        np.array([[1.0, 0.6, 0.2], [1.0, 0.5, 0.1]]),
    ],
)
def test_zero_padding_uses_2d_time_coordinates(curves):
    time_coordinates = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]])

    with pytest.warns(UserWarning, match="first time coordinate"):
        padded_curves, padded_times = zero_padding(curves, time_coordinates)

    np.testing.assert_allclose(padded_times[:, 0], 0.0)
    np.testing.assert_allclose(padded_times[:, 1:], time_coordinates)
    if curves.ndim == 1:
        np.testing.assert_allclose(padded_curves[0], 1.0)
        np.testing.assert_allclose(padded_curves[1:], curves)
    else:
        np.testing.assert_allclose(padded_curves[:, 0], 1.0)
        np.testing.assert_allclose(padded_curves[:, 1:], curves)


def test_zero_padding_does_not_duplicate_existing_2d_zero_coordinates():
    curves = np.array([[0.9, 0.5, 0.1], [0.8, 0.4, 0.1]])
    time_coordinates = np.array([[0.0, 1.0, 2.0], [0.0, 1.5, 3.0]])

    padded_curves, padded_times = zero_padding(curves, time_coordinates)

    np.testing.assert_allclose(padded_curves, curves)
    np.testing.assert_allclose(padded_times, time_coordinates)
    assert np.all(np.diff(padded_times, axis=1) > 0)


def test_zero_padding_rejects_mixed_2d_grid_starts():
    with pytest.raises(ValueError, match="All rows"):
        zero_padding(
            np.array([[1.0, 0.5, 0.1], [1.0, 0.4, 0.1]]),
            np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        )


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


def test_check_monotonicity_handles_repeated_infinities():
    increasing = np.array([[0.0, np.inf, np.inf], [1.0, 2.0, np.inf]])
    decreasing = np.array([[np.inf, np.inf, 2.0], [np.inf, 1.0, 1.0]])

    assert check_monotonicity(increasing, direction="increasing")
    assert check_monotonicity(decreasing, direction="decreasing")
    assert not check_monotonicity(increasing, direction="decreasing")


def test_check_monotonicity_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        check_monotonicity([0.0, 1.0], direction="sideways")


def test_make_monotonic_corrects_increasing_survival_curve():
    survival_curves = np.array([[0.2, 0.5, 0.8]])

    result = make_monotonic(survival_curves, np.array([0.0, 1.0, 2.0]), method="floor")

    assert check_monotonicity(result, direction="decreasing")
    np.testing.assert_allclose(result, [[0.2, 0.2, 0.2]])


def test_make_monotonic_isotonic_is_l2_optimal_for_increasing_curve():
    result = make_monotonic(
        np.array([0.45, 0.13, 0.40]),
        np.array([0.0, 1.0, 2.0]),
        method="isotonic",
        direction="increasing",
    )

    assert result.ndim == 1
    np.testing.assert_allclose(result, [0.29, 0.29, 0.40])


def test_make_monotonic_isotonic_supports_2d_decreasing_curves():
    curves = np.array(
        [
            [0.60, 0.20, 0.40],
            [0.90, 0.50, 0.10],
        ]
    )

    result = make_monotonic(
        curves,
        np.array([0.0, 1.0, 2.0]),
        method="isotonic",
    )

    assert result.shape == curves.shape
    np.testing.assert_allclose(
        result,
        [
            [0.60, 0.30, 0.30],
            [0.90, 0.50, 0.10],
        ],
    )


def test_make_monotonic_ceil_preserves_legacy_default_behavior():
    result = make_monotonic(
        np.array([[0.90, 0.70, 0.80]]),
        np.array([0.0, 1.0, 2.0]),
    )

    np.testing.assert_allclose(result, [[0.90, 0.80, 0.80]])


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("ceil", [0.10, 0.50, 0.50]),
        ("floor", [0.10, 0.30, 0.30]),
    ],
)
def test_make_monotonic_accumulators_support_increasing_curves(method, expected):
    result = make_monotonic(
        np.array([0.10, 0.50, 0.30]),
        np.array([0.0, 1.0, 2.0]),
        method=method,
        direction="increasing",
    )

    np.testing.assert_allclose(result, expected)


def test_make_monotonic_bootstrap_supports_increasing_cdf():
    result = make_monotonic(
        np.array([0.20, 0.60, 0.40]),
        np.array([0.0, 1.0, 2.0]),
        method="bootstrap",
        direction="increasing",
        seed=0,
        num_bs=100,
    )

    assert result.ndim == 1
    assert check_monotonicity(result, direction="increasing")


@pytest.mark.parametrize(
    ("curves", "times", "kwargs", "message"),
    [
        (
            np.array([0.9, 0.5, 0.1]),
            np.array([0.0, 1.0, 2.0]),
            {"method": "unknown"},
            "method",
        ),
        (
            np.array([0.9, 0.5, 0.1]),
            np.array([0.0, 1.0, 2.0]),
            {"direction": "sideways"},
            "direction",
        ),
        (
            np.ones((1, 1, 3)),
            np.array([0.0, 1.0, 2.0]),
            {},
            "1-D or 2-D",
        ),
        (
            np.array([0.9, 0.5, 0.1]),
            np.array([[0.0, 1.0, 2.0]]),
            {},
            "times_coordinate must be a 1-D",
        ),
        (
            np.array([0.9, 0.5, 0.1]),
            np.array([0.0, 1.0]),
            {},
            "same number of time points",
        ),
        (
            np.array([0.9, 0.5, 0.1]),
            np.array([0.0, 2.0, 1.0]),
            {},
            "sorted in ascending order",
        ),
    ],
)
def test_make_monotonic_rejects_invalid_inputs(curves, times, kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_monotonic(curves, times, **kwargs)


def test_predict_prob_from_curve_pads_time_grid_starting_after_zero():
    with pytest.warns(UserWarning, match="first time coordinate"):
        prob = predict_prob_from_curve(
            survival_curve=np.array([0.8, 0.5, 0.2]),
            times_coordinate=np.array([1.0, 2.0, 3.0]),
            target_time=0.5,
        )

    assert prob == pytest.approx(0.9)


def test_predict_prob_from_curve_accepts_curve_below_one_at_origin():
    prob = predict_prob_from_curve(
        survival_curve=np.array([0.9, 0.8, 0.5]),
        times_coordinate=np.array([0.0, 1.0, 2.0]),
        target_time=1.0,
    )

    assert prob == pytest.approx(0.8)


def test_predict_prob_from_curve_rejects_negative_target_time():
    with pytest.raises(ValueError, match="non-negative"):
        predict_prob_from_curve(
            survival_curve=np.array([1.0, 0.8, 0.5]),
            times_coordinate=np.array([0.0, 1.0, 2.0]),
            target_time=-0.1,
        )


def test_predict_prob_from_curve_rejects_negative_time_coordinates():
    with pytest.raises(ValueError, match="non-negative"):
        predict_prob_from_curve(
            survival_curve=np.array([1.0, 0.8, 0.5]),
            times_coordinate=np.array([-1.0, 1.0, 2.0]),
            target_time=1.0,
        )


def test_predict_multi_probs_from_curve_pads_time_grid_starting_after_zero():
    with pytest.warns(UserWarning, match="first time coordinate"):
        probs = predict_multi_probs_from_curve(
            survival_curve=np.array([0.8, 0.5, 0.2]),
            times_coordinate=np.array([1.0, 2.0, 3.0]),
            target_times=np.array([0.0, 0.5, 1.0]),
        )

    np.testing.assert_allclose(probs, [1.0, 0.9, 0.8])


def test_predict_multi_probs_from_curve_rejects_negative_target_times():
    with pytest.raises(ValueError, match="non-negative"):
        predict_multi_probs_from_curve(
            survival_curve=np.array([1.0, 0.8, 0.5]),
            times_coordinate=np.array([0.0, 1.0, 2.0]),
            target_times=np.array([0.5, -0.1]),
        )


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


def test_survival_to_quantile_rejects_duplicate_time_coordinates():
    with pytest.raises(ValueError, match="strictly increasing"):
        survival_to_quantile(
            surv_prob=[[1.0, 0.8, 0.5]],
            time_coordinates=[[0.0, 1.0, 1.0]],
            quantile_levels=[0.25, 0.5],
        )


def test_survival_to_quantile_rejects_decreasing_quantile_levels():
    with pytest.raises(ValueError, match="increasing order"):
        survival_to_quantile(
            surv_prob=[[1.0, 0.8, 0.5]],
            time_coordinates=[[0.0, 1.0, 2.0]],
            quantile_levels=[0.5, 0.25],
        )


def test_survival_to_quantile_handles_all_one_survival_curve_with_pchip():
    result = survival_to_quantile(
        surv_prob=[[1.0, 1.0, 1.0]],
        time_coordinates=[[0.0, 1.0, 2.0]],
        quantile_levels=[0.0, 0.25, 0.5],
    )

    np.testing.assert_allclose(result, [[0.0, np.inf, np.inf]])


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


def test_auprc_right_censored_rejects_duplicate_time_grid():
    with pytest.raises(AssertionError, match="strictly increasing"):
        auprc_right_censored_grid(
            pred_cdf=np.array([[0.2, 0.5, 0.8]]),
            time_grid=np.array([0.0, 1.0, 1.0]),
            censor_times=np.array([1.0]),
        )


def test_auprc_ic_left_censored_uses_zero_at_origin_by_default():
    pred_cdf = np.array([[0.2, 0.6, 1.0]])
    time_grid = np.array([1.0, 2.0, 3.0])
    left = np.array([0.0])
    right = np.array([np.inf])

    default_score = auprc_ic(pred_cdf, time_grid, left, right)
    overridden_score = auprc_ic(
        pred_cdf,
        time_grid,
        left,
        right,
        left_extrapolation_value=0.2,
    )

    assert default_score == pytest.approx(1.0)
    assert overridden_score == pytest.approx(0.8)
