import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from SurvivalEVAL import SurvivalEvaluator


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


def test_residuals_and_km_calibration(evaluator_data):
    evaluator = evaluator_data["evaluator"]

    residuals = evaluator.residuals(method="CoxSnell")
    assert residuals.shape[0] == evaluator_data["n_test"]

    km_cal = evaluator.km_calibration(draw_figure=False)
    assert np.isfinite(km_cal)


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
