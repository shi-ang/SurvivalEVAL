import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from SurvivalEVAL import IntervalCenEvaluator
from SurvivalEVAL.Evaluations.DistributionCalibration import (
    coverage_ic,
    create_censor_hist,
    create_interval_c_hist,
)
from SurvivalEVAL.Evaluations.util import (
    predict_multi_probs_from_curve,
    predict_prob_from_curve,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    KaplanMeier,
    TurnbullEstimatorLifelines,
)


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("flat", [False, True])
def test_survival_predictions_at_infinity_are_zero(interpolation, flat):
    curve = np.ones(3) if flat else np.array([1.0, 0.8, 0.6])
    times = np.array([0.0, 1.0, 2.0])
    targets = np.array([np.inf, 1.0, 4.0, np.inf])
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        result = predict_multi_probs_from_curve(curve, times, targets, interpolation)
        scalar = predict_prob_from_curve(curve, times, np.inf, interpolation)
    np.testing.assert_allclose(
        result, [0.0, 1.0 if flat else 0.8, 1.0 if flat else 0.2, 0.0]
    )
    assert scalar == 0.0


def test_kaplan_meier_flat_tail_handles_infinity():
    km = KaplanMeier(np.array([1.0, 2.0]), np.zeros(2))
    with np.errstate(divide="raise", invalid="raise"):
        probabilities = km.predict(np.array([0.0, 2.0, 100.0, np.inf]))
    np.testing.assert_array_equal(probabilities, [1.0, 1.0, 1.0, 0.0])


@pytest.mark.parametrize("right", [1.0, 3.0, np.inf])
def test_turnbull_common_support_avoids_degenerate_fitting(right):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        tb = TurnbullEstimatorLifelines(np.ones(3), np.full(3, right))
        probabilities = tb.predict(np.array([0.0, 1.0, 3.0, np.inf]))
    expected = [1.0, float(right > 1.0), float(right > 3.0), 0.0]
    np.testing.assert_array_equal(probabilities, expected)


@pytest.mark.parametrize("probability, index", [(0.0, 9), (0.55, 4), (1.0, 0)])
def test_interval_histogram_equal_endpoints_are_point_masses(probability, index):
    with np.errstate(divide="raise", invalid="raise"):
        hist = create_interval_c_hist(probability, probability, 10)
    expected = np.zeros(10)
    expected[index] = 1.0
    np.testing.assert_array_equal(hist, expected)


def test_interval_histogram_distributes_overlap_mass():
    expected = np.array([1 / 6, 5 / 12, 5 / 12, 0.0])
    np.testing.assert_allclose(create_interval_c_hist(0.85, 0.25, 4), expected)
    np.testing.assert_allclose(create_interval_c_hist(0.25, 0.85, 4), expected)


@pytest.mark.parametrize("probability", [0.0, 0.2, 0.55, 1.0, 1e-300])
def test_interval_histogram_matches_right_censored_histogram(probability):
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        hist = create_interval_c_hist(probability, 0.0, 10)
    np.testing.assert_allclose(hist, create_censor_hist(probability, 10))
    assert hist.sum() == pytest.approx(1.0)


def test_linear_coverage_handles_finite_exact_and_unbounded_intervals():
    with np.errstate(divide="raise", invalid="raise"):
        observed, gap, width = coverage_ic(
            np.array([np.inf, 1.0, 3.0, 1.0, 3.0, 0.0, 3.0]),
            np.array([np.inf, 2.0, np.inf, np.inf, 4.0, np.inf, 3.0]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0]),
            np.array([np.inf, 3.0, np.inf, np.inf, np.inf, 5.0, 3.0]),
            cov_level=0.8,
            method="linear",
        )
    assert observed == pytest.approx(4.5 / 7)
    assert gap == pytest.approx(observed - 0.8)
    assert width == np.inf


@pytest.mark.parametrize("censoring", ["finite", "right", "mixed"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_interval_evaluator_all_one_curves(censoring, dtype):
    left = np.arange(1.0, 13.0)
    right = left + 0.1
    if censoring == "right":
        right[:] = np.inf
    elif censoring == "mixed":
        right[::2] = np.inf
    evaluator = IntervalCenEvaluator(
        np.ones((12, 3), dtype=dtype),
        np.array([0.0, 1.0, 2.0], dtype=dtype),
        left,
        right,
        left,
        left + 0.1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p_value, details = evaluator.d_calibration(return_details=True)
        ks_p_value, ks_statistic = evaluator.ksd_calibration()
        coverages = [
            evaluator.coverage(cov_level=0.8, method=method)
            for method in ["linear", "Turnbull"]
        ]
        errors = [evaluator.mae(), evaluator.mse(), evaluator.rmse()]
    try:
        censored_count = np.count_nonzero(np.isinf(right))
        expected_hist = np.full(10, censored_count / 10)
        expected_hist[0] += left.size - censored_count
        np.testing.assert_allclose(details["histogram"], expected_hist, rtol=1e-6)
        assert np.isfinite(p_value)
        assert np.isfinite(ks_p_value)
        assert ks_statistic == 1.0
        for coverage in coverages:
            np.testing.assert_array_equal(coverage, [0.0, -0.8, np.inf])
        np.testing.assert_array_equal(errors, 0.0 if censoring == "right" else np.inf)
    finally:
        plt.close(details["histogram_plot"][0])
        plt.close(details["pp_plot"][0])


def test_interval_evaluator_mixed_curves_preserve_finite_predictions():
    times = np.array([0.0, 1.0, 2.0])
    evaluator = IntervalCenEvaluator(
        np.array([[1.0, 1.0, 1.0], [1.0, 0.8, 0.6], [1.0, 0.6, 0.2]]),
        times,
        np.ones(3),
        np.array([2.0, 2.0, np.inf]),
        np.arange(1.0, 7.0),
        np.arange(1.1, 7.1),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        np.testing.assert_allclose(
            evaluator.predict_probability_from_curve(1.0), [1.0, 0.8, 0.6]
        )
        np.testing.assert_allclose(
            evaluator.predict_probability_from_curve(evaluator.right_limits),
            [1.0, 0.6, 0.0],
        )
        _, hist = evaluator.d_calibration()
        _, ks_statistic = evaluator.ksd_calibration()
        coverage, _, width = evaluator.coverage(cov_level=0.8)
    assert np.all(np.isfinite(hist))
    assert hist.sum() == pytest.approx(3.0)
    assert np.isfinite(ks_statistic)
    assert np.isfinite(coverage)
    assert width == np.inf
