import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from SurvivalEVAL import ScikitSurvivalEvaluator, SurvivalEvaluator
from SurvivalEVAL.Evaluations.SingleTimeCalibration import one_calibration


class _StepFunction:
    def __init__(self, times, probabilities):
        self.x = np.asarray(times)
        self.y = np.asarray(probabilities)


def _sample_interval_data(rng: np.random.Generator, n_samples: int):
    for _ in range(100):
        rates = rng.uniform(0.05, 0.2, size=n_samples)
        event_times = rng.exponential(scale=1.0 / rates)
        censor_times = rng.exponential(scale=12.0, size=n_samples)
        indicators = event_times <= censor_times
        if indicators.any() and (~indicators).any():
            observed = np.minimum(event_times, censor_times)
            return rates, observed, indicators
    raise RuntimeError(
        "Failed to generate interval data with both events and censoring."
    )


@pytest.fixture
def evaluator_data():
    rng = np.random.default_rng(42)
    n_test = 80
    n_train = 120
    time_grid = np.linspace(0.0, 25.0, 80)

    train_rates, train_times, train_indicators = _sample_interval_data(rng, n_train)
    test_rates, test_times, test_indicators = _sample_interval_data(rng, n_test)

    pred_rates = test_rates * rng.uniform(0.8, 1.2, size=n_test)
    surv_curves = np.exp(-np.outer(pred_rates, time_grid))
    surv_curves = np.clip(surv_curves, 1e-8, 1.0)

    evaluator = SurvivalEvaluator(
        pred_survs=surv_curves,
        time_coordinates=time_grid,
        event_times=test_times,
        event_indicators=test_indicators,
        train_event_times=train_times,
        train_event_indicators=train_indicators,
        predict_time_method="Median",
    )

    return {
        "evaluator": evaluator,
        "time_grid": time_grid,
        "n_test": n_test,
        "train_times": train_times,
        "train_indicators": train_indicators,
        "test_times": test_times,
        "test_indicators": test_indicators,
    }


def test_prediction_utilities(evaluator_data):
    evaluator = evaluator_data["evaluator"]
    n_test = evaluator_data["n_test"]

    single_prob = evaluator.predict_probability_from_curve(8.0)
    assert single_prob.shape == (n_test,)

    per_sample_times = np.linspace(4.0, 12.0, n_test)
    per_sample_prob = evaluator.predict_probability_from_curve(per_sample_times)
    assert per_sample_prob.shape == (n_test,)

    target_grid = np.array([2.0, 6.0, 10.0])
    multi_probs = evaluator.predict_multi_probabilities_from_curve(target_grid)
    assert multi_probs.shape == (n_test, target_grid.size)

    intervals = evaluator.predict_interval(cov_level=0.8)
    assert intervals.shape == (n_test, 2)
    assert np.all(intervals[:, 0] <= intervals[:, 1])

    quantile_intervals = evaluator.predict_interval(quantile_range=(0.1, 0.9))
    np.testing.assert_allclose(quantile_intervals, intervals)


def test_survival_evaluator_rejects_prediction_row_count_mismatch():
    with pytest.raises(ValueError, match="prediction rows"):
        SurvivalEvaluator(
            pred_survs=np.ones((3, 3)),
            time_coordinates=np.array([0.0, 1.0, 2.0]),
            event_times=np.array([1.0, 2.0]),
            event_indicators=np.array([1, 0]),
        )


def test_survival_evaluator_rejects_time_coordinate_row_count_mismatch():
    with pytest.raises(ValueError, match="prediction rows"):
        SurvivalEvaluator(
            pred_survs=np.array([1.0, 0.8, 0.5]),
            time_coordinates=np.array(
                [
                    [0.0, 1.0, 2.0],
                    [0.0, 1.5, 3.0],
                    [0.0, 2.0, 4.0],
                ]
            ),
            event_times=np.array([1.0, 2.0]),
            event_indicators=np.array([1, 0]),
        )


def test_predict_interval_accepts_matching_quantile_range_and_coverage_level(
    evaluator_data,
):
    evaluator = evaluator_data["evaluator"]

    combined_intervals = evaluator.predict_interval(
        quantile_range=(0.1, 0.9), cov_level=0.8
    )
    quantile_intervals = evaluator.predict_interval(quantile_range=(0.1, 0.9))

    np.testing.assert_allclose(combined_intervals, quantile_intervals)


def test_predict_interval_rejects_mismatched_quantile_range_and_coverage_level(
    evaluator_data,
):
    evaluator = evaluator_data["evaluator"]

    with pytest.raises(ValueError, match="must equal"):
        evaluator.predict_interval(quantile_range=(0.1, 0.8), cov_level=0.8)


def test_scikit_survival_evaluator_handles_all_one_curves():
    curves = [
        _StepFunction([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
        _StepFunction([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
    ]

    evaluator = ScikitSurvivalEvaluator(
        surv=curves,
        event_times=np.array([1.0, 2.0]),
        event_indicators=np.array([1, 1]),
    )

    np.testing.assert_allclose(evaluator.pred_survs[:, -1], 0.99)
    assert np.all(np.isfinite(evaluator.predicted_event_times))


def test_scikit_survival_evaluator_preserves_mixed_batch_repair():
    curves = [
        _StepFunction([1.0, 2.0, 3.0], [1.0, 0.9, 0.8]),
        _StepFunction([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]),
    ]

    evaluator = ScikitSurvivalEvaluator(
        surv=curves,
        event_times=np.array([1.0, 2.0]),
        event_indicators=np.array([1, 1]),
    )

    assert np.isclose(evaluator.pred_survs[0, -1], 0.8)
    assert np.isclose(evaluator.pred_survs[1, -1], 0.99)


def test_pchip_with_2d_zero_based_grids_does_not_get_duplicate_zeros():
    time_coordinates = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.5, 2.5, 4.0],
        ]
    )
    pred_survs = np.array(
        [
            [0.9, 0.7, 0.4, 0.1],
            [0.8, 0.6, 0.3, 0.1],
        ]
    )

    evaluator = SurvivalEvaluator(
        pred_survs=pred_survs,
        time_coordinates=time_coordinates,
        event_times=np.array([1.0, 2.0]),
        event_indicators=np.array([1, 1]),
        predict_time_method="Mean",
        interpolation="Pchip",
    )

    np.testing.assert_allclose(evaluator.time_coordinates, time_coordinates)
    assert np.all(np.diff(evaluator.time_coordinates, axis=1) > 0)
    assert np.all(np.isfinite(evaluator.predicted_event_times))


def test_concordance_variants(evaluator_data):
    evaluator = evaluator_data["evaluator"]

    c_harrell, concordant_h, total_h = evaluator.concordance(method="Harrell")
    assert 0.0 <= c_harrell <= 1.0
    assert total_h > 0
    assert concordant_h <= total_h

    c_margin, concordant_m, total_m = evaluator.concordance(method="Margin")
    assert 0.0 <= c_margin <= 1.0
    assert total_m > 0
    assert concordant_m <= total_m


def test_auc_alias(evaluator_data):
    evaluator = evaluator_data["evaluator"]
    auc = evaluator.auc(target_time=9.0)
    auroc = evaluator.auroc(target_time=9.0)
    assert np.isclose(auc, auroc)
    assert 0.0 <= auc <= 1.0


def test_brier_scores(evaluator_data):
    evaluator = evaluator_data["evaluator"]
    brier = evaluator.brier_score(target_time=8.0, IPCW_weighted=True)
    assert np.isfinite(brier)

    times = np.array([4.0, 8.0, 12.0])
    brier_mp = evaluator.brier_score_multiple_points(
        target_times=times, IPCW_weighted=False
    )
    assert brier_mp.shape == times.shape
    assert np.all(np.isfinite(brier_mp))


def test_default_brier_score_without_training_data():
    time_grid = np.array([0.0, 1.0, 2.0, 4.0])
    pred_survs = np.array(
        [
            [1.0, 0.8, 0.5, 0.2],
            [1.0, 0.7, 0.4, 0.1],
            [1.0, 0.9, 0.6, 0.3],
        ]
    )
    event_times = np.array([1.0, 2.0, 3.0])

    censored_evaluator = SurvivalEvaluator(
        pred_survs=pred_survs,
        time_coordinates=time_grid,
        event_times=event_times,
        event_indicators=np.array([1, 0, 1]),
    )
    uncensored_evaluator = SurvivalEvaluator(
        pred_survs=pred_survs,
        time_coordinates=time_grid,
        event_times=event_times,
        event_indicators=np.ones(event_times.shape[0]),
    )

    assert np.isfinite(
        censored_evaluator.brier_score(target_time=None, IPCW_weighted=False)
    )
    assert np.isfinite(uncensored_evaluator.brier_score(target_time=None))
    with pytest.raises(TypeError, match="Train set information is missing"):
        censored_evaluator.brier_score(target_time=None, IPCW_weighted=True)


def test_integrated_brier_score(evaluator_data):
    evaluator = evaluator_data["evaluator"]

    ibs_ipcw = evaluator.integrated_brier_score(num_points=10, IPCW_weighted=True)
    assert np.isfinite(ibs_ipcw)

    explicit_grid = np.linspace(2.0, 14.0, 6)
    ibs_naive = evaluator.integrated_brier_score(
        target_times=explicit_grid, IPCW_weighted=False
    )
    assert np.isfinite(ibs_naive)

    with pytest.raises(ValueError):
        evaluator.integrated_brier_score(
            num_points=5, target_times=np.array([1.0, 2.0, 3.0])
        )


def test_error_metrics(evaluator_data):
    evaluator = evaluator_data["evaluator"]

    mae_hinge = evaluator.mae(method="Hinge", weighted=False)
    mae_margin = evaluator.mae(method="Margin")
    mse_ipcwt = evaluator.mse(method="IPCW-T")
    mse_pseudo = evaluator.mse(method="Pseudo_obs")
    rmse_pseudo = evaluator.rmse(method="Pseudo_obs")

    for value in [mae_hinge, mae_margin, mse_ipcwt, mse_pseudo, rmse_pseudo]:
        assert np.isfinite(value)
        assert value >= 0

    assert np.isclose(rmse_pseudo, np.sqrt(mse_pseudo), atol=1e-6)


def test_calibration_metrics(evaluator_data):
    evaluator = evaluator_data["evaluator"]
    target_time = 8.0

    one_p, one_details = evaluator.one_calibration(
        target_time=target_time, num_bins=5, return_details=True
    )
    assert np.isfinite(one_p)
    assert "observed_probabilities" in one_details
    hist_fig, _ = one_details["histogram_plot"]
    pp_fig, _ = one_details["pp_plot"]
    plt.close(hist_fig)
    plt.close(pp_fig)

    ici_summary = evaluator.integrated_calibration_index(
        target_time=target_time, draw_figure=False
    )
    assert "ICI" in ici_summary

    d_p, d_details = evaluator.d_calibration(num_bins=6, return_details=True)
    assert np.isfinite(d_p)
    assert "histogram" in d_details
    d_hist_fig, _ = d_details["histogram_plot"]
    d_pp_fig, _ = d_details["pp_plot"]
    plt.close(d_hist_fig)
    plt.close(d_pp_fig)

    ksd_p, ksd_details = evaluator.ksd_calibration(return_details=True)
    assert np.isfinite(ksd_p)
    assert "figure" in ksd_details
    ksd_fig, _ = ksd_details["figure"]
    plt.close(ksd_fig)


def test_uncensored_one_calibration_uses_filtered_bin_size():
    p_value, statistic, observed, expected = one_calibration(
        preds=np.array([0.9, 0.8, 0.6, 0.5, 0.3, 0.2]),
        event_time=np.array([1.0, 0.5, 3.0, 4.0, 5.0, 6.0]),
        event_indicator=np.array([1, 0, 1, 1, 1, 1]),
        target_time=2.0,
        num_bins=3,
        method="Uncensored",
    )

    assert np.isfinite(p_value)
    assert np.isclose(statistic, 29 / 9)
    np.testing.assert_allclose(observed, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(expected, [0.9, 0.55, 0.25])


def test_h_statistic_one_calibration_includes_prediction_one():
    p_value, statistic, observed, expected = one_calibration(
        preds=np.array([0.1, 0.4, 0.7, 1.0]),
        event_time=np.full(4, 2.0),
        event_indicator=np.ones(4, dtype=int),
        target_time=1.0,
        num_bins=3,
        binning_strategy="H",
        method="Uncensored",
    )

    assert np.isfinite(p_value)
    assert np.isfinite(statistic)
    np.testing.assert_allclose(observed, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(expected, [0.1, 0.4, 0.85])


def test_residuals_and_km_calibration(evaluator_data):
    evaluator = evaluator_data["evaluator"]

    residuals = evaluator.residuals(method="CoxSnell")
    assert residuals.shape[0] == evaluator_data["n_test"]

    km_cal = evaluator.km_calibration(draw_figure=False)
    assert np.isfinite(km_cal)


def test_calibration_metrics_support_shared_curve_with_per_sample_grids():
    shared_curve = np.array([1.0, 0.8, 0.45, 0.1])
    per_sample_grids = np.array(
        [
            [0.0, 1.0, 2.0, 4.0],
            [0.0, 0.5, 2.5, 5.0],
            [0.0, 1.5, 3.0, 6.0],
            [0.0, 0.75, 1.75, 3.5],
        ]
    )
    event_times = np.array([1.2, 2.0, 3.5, 2.7])
    event_indicators = np.array([1, 0, 1, 1])

    shared_curve_evaluator = SurvivalEvaluator(
        pred_survs=shared_curve,
        time_coordinates=per_sample_grids,
        event_times=event_times,
        event_indicators=event_indicators,
    )
    expanded_curve_evaluator = SurvivalEvaluator(
        pred_survs=np.tile(shared_curve, (event_times.size, 1)),
        time_coordinates=per_sample_grids,
        event_times=event_times,
        event_indicators=event_indicators,
    )

    shared_km_cal = shared_curve_evaluator.km_calibration()
    shared_auprc = shared_curve_evaluator.auprc(n_quad=64)

    assert np.isfinite(shared_km_cal)
    assert 0.0 <= shared_auprc <= 1.0
    assert np.isclose(shared_km_cal, expanded_curve_evaluator.km_calibration())
    assert np.isclose(shared_auprc, expanded_curve_evaluator.auprc(n_quad=64))


def test_log_rank_and_auprc(evaluator_data):
    evaluator = evaluator_data["evaluator"]

    p_value, statistic = evaluator.log_rank()
    assert np.isfinite(p_value)
    assert np.isfinite(statistic)

    auprc_value = evaluator.auprc(n_quad=64)
    assert 0.0 <= auprc_value <= 1.0


def test_pred_survs_setter_resets_cache(evaluator_data):
    evaluator = evaluator_data["evaluator"]
    original_times = evaluator.predicted_event_times.copy()
    updated_curves = evaluator.pred_survs * 0.95 + 0.05
    evaluator.pred_survs = updated_curves
    refreshed_times = evaluator.predicted_event_times
    assert not np.allclose(refreshed_times, original_times)
