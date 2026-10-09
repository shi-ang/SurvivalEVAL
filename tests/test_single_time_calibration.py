import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from SurvivalEVAL import IntervalCenEvaluator, SingleTimeEvaluator, SurvivalEvaluator
from SurvivalEVAL.Evaluations.SingleTimeCalibration import (
    integrated_calibration_index,
    one_cal_ic,
    one_calibration,
)


@pytest.fixture(params=["DN", "Uncensored", "MidPoint", "Turnbull"])
def calibrate(request):
    def evaluate(predictions, times, **kwargs):
        if request.param in {"DN", "Uncensored"}:
            return one_calibration(
                predictions,
                times,
                np.ones(times.size),
                2.0,
                method=request.param,
                **kwargs,
            )
        left = right = times
        if request.param == "Turnbull":
            centers = times + np.linspace(-0.3, 0.3, times.size)
            left, right = centers - 0.01, centers + 0.01
        return one_cal_ic(
            predictions,
            left,
            right,
            2.0,
            method=request.param,
            **kwargs,
        )

    return evaluate


@pytest.mark.parametrize("probability", [0.0, 1.0])
@pytest.mark.parametrize("matches", [False, True])
def test_one_calibration_boundary_contributions(calibrate, probability, matches):
    observed = probability if matches else 1 - probability
    times = np.full(12, 1.0 if observed else 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p_value, statistic, obs, exp = calibrate(
            np.full(12, probability),
            times,
            num_bins=3,
        )
    assert statistic == (0.0 if matches else np.inf)
    assert p_value == (1.0 if matches else 0.0)
    np.testing.assert_allclose(obs, observed)
    np.testing.assert_allclose(exp, probability)


def test_one_calibration_combines_boundary_and_interior_bins(calibrate):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p_value, statistic, observed, _ = calibrate(
            np.repeat([0.0, 0.5, 1.0], 4),
            np.array([3.0] * 4 + [0.5, 0.5, 3.0, 3.0] + [0.5] * 4),
            num_bins=3,
            binning_strategy="H",
        )
    assert np.isfinite(statistic)
    assert 0 < p_value <= 1
    assert observed[0] == 0.0
    assert observed[-1] == 1.0


def test_one_calibration_insufficient_bins_returns_undefined_p_value(calibrate):
    p_value, statistic, observed, expected = calibrate(
        np.zeros(12),
        np.ones(12),
        num_bins=3,
        binning_strategy="H",
    )
    assert np.isnan(p_value)
    assert statistic == np.inf
    assert observed == [1.0]
    assert expected == [0.0]


@pytest.mark.parametrize("probability", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ici_constant_predictions_use_kaplan_meier(probability, dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary, (fig, _) = integrated_calibration_index(
            np.full(6, probability, dtype=dtype),
            np.arange(1.0, 7.0),
            np.array([1, 0, 1, 1, 0, 1]),
            3.0,
            draw_figure=True,
        )
    try:
        for metric in ["ICI", "E50", "E90", "E_max"]:
            assert summary[metric] == pytest.approx(abs(probability - 0.375))
        np.testing.assert_allclose(summary["curve"]["grid"], probability)
        np.testing.assert_allclose(summary["curve"]["cal_pred"], 0.375)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("levels", [None, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ici_mixed_boundary_predictions_are_finite(levels, dtype):
    rng = np.random.default_rng(17)
    predictions = rng.uniform(0.05, 0.95, 120)
    if levels is not None:
        predictions = np.floor(predictions * levels) / (levels - 1)
    predictions[:2] = [0.0, 1.0]
    predictions = predictions.astype(dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = integrated_calibration_index(
            predictions,
            rng.exponential(5.0, 120),
            rng.integers(0, 2, 120),
            3.0,
        )
    for metric in ["ICI", "E50", "E90", "E_max"]:
        assert 0 <= summary[metric] <= 1
    assert np.all(np.isfinite(summary["curve"]["cal_pred"]))


def test_survival_evaluator_all_one_curves_calibrate_without_arithmetic_errors():
    evaluator = SurvivalEvaluator(
        np.ones((12, 3)),
        np.array([0.0, 1.0, 2.0]),
        np.arange(1.0, 13.0),
        np.ones(12),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summary = evaluator.integrated_calibration_index(3.0, draw_figure=False)
        p_value, _, expected = evaluator.one_calibration(3.0, num_bins=3)
    assert summary["ICI"] == pytest.approx(0.25)
    assert p_value == 0.0
    np.testing.assert_array_equal(expected, 0.0)


@pytest.mark.parametrize("kind", ["survival", "single-time", "interval"])
@pytest.mark.parametrize("strategy", ["C", "H"])
def test_constant_calibration_details_handle_tied_or_single_bins(kind, strategy):
    times = np.arange(1.0, 13.0)
    curves = np.ones((12, 3))
    grid = np.array([0.0, 1.0, 2.0])
    if kind == "survival":
        evaluator = SurvivalEvaluator(curves, grid, times, np.ones(12))
    elif kind == "single-time":
        evaluator = SingleTimeEvaluator(
            np.ones(12), times, np.ones(12), target_time=3.0
        )
    else:
        evaluator = IntervalCenEvaluator(curves, grid, times, times + 0.1)

    kwargs = {"num_bins": 3, "binning_strategy": strategy, "return_details": True}
    if kind != "single-time":
        kwargs["target_time"] = 3.0
    if kind == "interval":
        kwargs["method"] = "MidPoint"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p_value, details = evaluator.one_calibration(**kwargs)
    try:
        assert p_value == 0.0 if strategy == "C" else np.isnan(p_value)
        assert details["statistics"] == np.inf
        assert np.isnan(details["max_local_deviation"])
    finally:
        plt.close(details["histogram_plot"][0])
        plt.close(details["pp_plot"][0])
