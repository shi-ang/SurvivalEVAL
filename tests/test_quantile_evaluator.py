import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from SurvivalEVAL import QuantileRegEvaluator


def _generate_dataset(rng: np.random.Generator, n_samples: int):
    for _ in range(100):
        rates = rng.uniform(0.05, 0.2, size=n_samples)
        event = rng.exponential(scale=1.0 / rates) + 0.1
        censor = rng.exponential(scale=8.0, size=n_samples) + 0.1
        indicators = event <= censor
        if indicators.any() and (~indicators).any():
            observed = np.minimum(event, censor)
            return rates, observed, indicators
    raise RuntimeError("Unable to generate dataset with mixed censoring status.")


@pytest.fixture
def quantile_evaluator():
    rng = np.random.default_rng(123)
    n_test = 60
    n_train = 90

    quantile_levels = np.linspace(0.05, 0.95, 10)

    train_rates, train_times, train_indicators = _generate_dataset(rng, n_train)
    test_rates, test_times, test_indicators = _generate_dataset(rng, n_test)

    rate_scale = rng.lognormal(mean=0.0, sigma=0.1, size=n_test)
    adjusted_rates = np.clip(test_rates * rate_scale, 0.02, 0.4)
    predicted_quantiles = -np.log(1.0 - quantile_levels) / adjusted_rates[:, None]

    evaluator = QuantileRegEvaluator(
        pred_regs=predicted_quantiles,
        quantile_levels=quantile_levels,
        event_times=test_times,
        event_indicators=test_indicators,
        train_event_times=train_times,
        train_event_indicators=train_indicators,
        predict_time_method="Median",
    )

    return {
        "evaluator": evaluator,
        "quantile_levels": quantile_levels,
        "n_test": n_test,
        "train_times": train_times,
        "train_indicators": train_indicators,
        "test_times": test_times,
        "test_indicators": test_indicators,
    }


def test_quantile_prediction_utilities(quantile_evaluator):
    evaluator = quantile_evaluator["evaluator"]
    n_test = quantile_evaluator["n_test"]

    prob_at_time = evaluator.predict_probability_from_curve(6.0)
    assert prob_at_time.shape == (n_test,)
    assert np.all((prob_at_time >= 0.0) & (prob_at_time <= 1.0))

    per_sample_times = np.linspace(3.0, 9.0, n_test)
    sample_probs = evaluator.predict_probability_from_curve(per_sample_times)
    assert sample_probs.shape == (n_test,)

    grid_times = np.array([2.0, 5.0, 9.0, 12.0])
    multi_probs = evaluator.predict_multi_probabilities_from_curve(grid_times)
    assert multi_probs.shape == (n_test, grid_times.size)

    intervals = evaluator.predict_interval(cov_level=0.8)
    assert intervals.shape == (n_test, 2)
    assert np.all(intervals[:, 0] <= intervals[:, 1])


def test_quantile_concordance_and_auc(quantile_evaluator):
    evaluator = quantile_evaluator["evaluator"]

    c_harrell, concordant_h, total_h = evaluator.concordance(method="Harrell")
    assert 0.0 <= c_harrell <= 1.0
    assert concordant_h <= total_h

    c_margin, concordant_m, total_m = evaluator.concordance(method="Margin")
    assert 0.0 <= c_margin <= 1.0
    assert concordant_m <= total_m

    auc = evaluator.auc(target_time=7.0)
    assert 0.0 <= auc <= 1.0
    assert np.isclose(auc, evaluator.auroc(target_time=7.0))


def test_quantile_brier_metrics(quantile_evaluator):
    evaluator = quantile_evaluator["evaluator"]

    brier_single = evaluator.brier_score(target_time=6.0, IPCW_weighted=True)
    assert np.isfinite(brier_single)

    target_grid = np.array([4.0, 8.0, 12.0])
    brier_multi = evaluator.brier_score_multiple_points(
        target_times=target_grid, IPCW_weighted=False
    )
    assert brier_multi.shape == target_grid.shape
    assert np.all(np.isfinite(brier_multi))

    ibs_ipcw = evaluator.integrated_brier_score(num_points=9, IPCW_weighted=True)
    ibs_naive = evaluator.integrated_brier_score(
        target_times=np.linspace(2.0, 14.0, 7), IPCW_weighted=False
    )
    assert np.isfinite(ibs_ipcw)
    assert np.isfinite(ibs_naive)


def test_quantile_error_scores(quantile_evaluator):
    evaluator = quantile_evaluator["evaluator"]

    mae_hinge = evaluator.mae(method="Hinge", weighted=False)
    mae_margin = evaluator.mae(method="Margin")
    mse_ipcwt = evaluator.mse(method="IPCW-T")
    mse_pseudo = evaluator.mse(method="Pseudo_obs")
    rmse_margin = evaluator.rmse(method="Margin")

    for value in [mae_hinge, mae_margin, mse_ipcwt, mse_pseudo, rmse_margin]:
        assert np.isfinite(value)
        assert value >= 0.0


def test_quantile_calibration_tools(quantile_evaluator):
    evaluator = quantile_evaluator["evaluator"]
    target_time = 6.0

    one_p, one_details = evaluator.one_calibration(
        target_time=target_time, num_bins=5, return_details=True
    )
    assert np.isfinite(one_p)
    assert "observed_probabilities" in one_details
    fig_hist, _ = one_details["histogram_plot"]
    fig_pp, _ = one_details["pp_plot"]
    plt.close(fig_hist)
    plt.close(fig_pp)

    ici_summary = evaluator.integrated_calibration_index(
        target_time=target_time, draw_figure=False
    )
    assert "ICI" in ici_summary

    d_p, d_details = evaluator.d_calibration(num_bins=6, return_details=True)
    assert np.isfinite(d_p)
    assert "histogram" in d_details
    plt.close(d_details["histogram_plot"][0])
    plt.close(d_details["pp_plot"][0])

    ksd_p, ksd_details = evaluator.ksd_calibration(return_details=True)
    assert np.isfinite(ksd_p)
    plt.close(ksd_details["figure"][0])


def test_quantile_residuals_and_km_calibration(quantile_evaluator):
    evaluator = quantile_evaluator["evaluator"]
    n_test = quantile_evaluator["n_test"]

    residuals = evaluator.residuals(method="CoxSnell")
    assert residuals.shape == (n_test,)

    km_cal = evaluator.km_calibration(draw_figure=False)
    assert np.isfinite(km_cal)


def test_quantile_log_rank(quantile_evaluator):
    evaluator = quantile_evaluator["evaluator"]

    p_value, statistic = evaluator.log_rank()
    assert np.isfinite(p_value)
    assert np.isfinite(statistic)
