from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from lifelines import WeibullAFTFitter

from SurvivalEVAL.Evaluations.MeanError import inclusion_rate, mean_error_ic
from SurvivalEVAL.Evaluations.SingleTimeCalibration import one_cal_ic
from SurvivalEVAL.IntervalCenEvaluator import IntervalCenEvaluator
from SurvivalEVAL.NonparametricEstimator.SingleEvent import TurnbullEstimator


def _load_breast_data():
    data_path = Path(__file__).resolve().parents[1] / "data" / "breast.csv"
    df = pd.read_csv(data_path)
    df["right"] = df["right"].fillna(np.inf)
    return df


def _midpoint_times(interval_df: pd.DataFrame):
    left = interval_df["left"].to_numpy(float)
    right = interval_df["right"].to_numpy(float)
    is_finite = np.isfinite(right)
    times = np.where(is_finite, (left + right) / 2.0, left)
    indicators = is_finite.astype(int)
    return times, indicators


def test_h_statistic_interval_one_calibration_includes_prediction_one():
    p_value, statistic, observed, expected = one_cal_ic(
        preds=np.array([0.1, 0.4, 0.7, 1.0]),
        left_limits=np.full(4, 2.0),
        right_limits=np.full(4, 2.0),
        target_time=1.0,
        num_bins=3,
        binning_strategy="H",
        method="MidPoint",
    )

    assert np.isfinite(p_value)
    assert np.isfinite(statistic)
    np.testing.assert_allclose(observed, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(expected, [0.1, 0.4, 0.85])


@pytest.fixture(scope="module")
def interval_evaluator():
    df = _load_breast_data()
    train_df = df.iloc[:60].copy()
    test_df = df.iloc[60:].copy()

    aft_model = WeibullAFTFitter()
    aft_model.fit_interval_censoring(train_df, "left", "right")

    prediction = aft_model.predict_survival_function(test_df, times=None)
    time_coords = prediction.index.values
    pred_survs = prediction.values.T  # (N, T)

    evaluator = IntervalCenEvaluator(
        pred_survs=pred_survs,
        time_coordinates=time_coords,
        left_limits=test_df["left"].values,
        right_limits=test_df["right"].values,
        train_left_limits=train_df["left"].values,
        train_right_limits=train_df["right"].values,
    )

    test_times, test_indicators = _midpoint_times(test_df)
    train_times, train_indicators = _midpoint_times(train_df)

    evaluator.event_times = test_times
    evaluator.event_indicators = test_indicators
    evaluator.train_event_times = train_times
    evaluator.train_event_indicators = train_indicators

    return evaluator, time_coords


def test_interval_cen_evaluator_methods(interval_evaluator):
    evaluator, time_coords = interval_evaluator

    predicted_times = evaluator.predicted_event_times
    assert predicted_times.shape[0] == evaluator.left_limits.shape[0]

    evaluator._clear_cache()
    assert "predicted_event_times" not in evaluator.__dict__
    assert evaluator.predicted_event_times.shape == predicted_times.shape

    c_prob, num_prob, den_prob = evaluator.concordance(method="probability")
    assert np.isfinite(c_prob)
    assert np.isfinite(num_prob)
    assert np.isfinite(den_prob)
    assert np.isscalar(num_prob)
    assert np.isscalar(den_prob)

    c_mid, num_mid, den_mid = evaluator.concordance(method="midpoint")
    assert np.isfinite(c_mid)
    assert np.isfinite(num_mid)
    assert np.isfinite(den_mid)
    assert np.isscalar(num_mid)
    assert np.isscalar(den_mid)

    target_time = float(time_coords[min(3, len(time_coords) - 1)])
    brier = evaluator.brier_score(
        target_time=target_time,
        method="Tsouprou-marginal",
    )
    assert np.isfinite(brier)

    point_times = time_coords[1:4] if len(time_coords) > 3 else time_coords[:2]
    brier_multi = evaluator.brier_score_multiple_points(
        target_times=point_times,
        method="Tsouprou-marginal",
    )
    assert brier_multi.shape[0] == len(point_times)
    assert np.all(np.isfinite(brier_multi))

    ibs = evaluator.integrated_brier_score(
        num_points=5,
        method="uncensored",
    )
    assert np.isfinite(ibs)

    crps = evaluator.crps(num_points=5)
    assert np.isfinite(crps)

    one_p, one_obs, one_exp = evaluator.one_calibration(
        target_time=target_time,
        num_bins=4,
    )
    assert np.isfinite(one_p)
    assert len(one_obs) == len(one_exp) == 4

    d_p, d_hist = evaluator.d_calibration(num_bins=5)
    assert np.isfinite(d_p)
    assert len(d_hist) == 5

    auprc = evaluator.auprc(n_quad=64)
    assert np.isfinite(auprc)

    inclusion = evaluator.inclusion_rate()
    assert np.isfinite(inclusion)

    mae = evaluator.mae()
    mse = evaluator.mse()
    rmse = evaluator.rmse()
    assert np.isfinite(mae)
    assert np.isfinite(mse)
    assert np.isfinite(rmse)

    coverage, cov_gap, avg_width = evaluator.coverage(cov_level=0.9, method="Turnbull")
    assert np.isfinite(coverage)
    assert np.isfinite(cov_gap)
    assert np.isfinite(avg_width)


def test_interval_evaluator_rejects_negative_left_limits():
    with pytest.raises(ValueError, match="finite and non-negative"):
        IntervalCenEvaluator(
            pred_survs=np.array(
                [
                    [1.0, 0.8, 0.4, 0.1],
                    [1.0, 0.5, 0.2, 0.1],
                ]
            ),
            time_coordinates=np.array([0.0, 1.0, 2.0, 3.0]),
            left_limits=np.array([-1.0, 1.0]),
            right_limits=np.array([1.0, np.inf]),
        )


def test_exact_intervals_treat_matching_predictions_as_inside():
    left = np.array([2.0, 2.0])
    right = np.array([2.0, 2.0])
    predicted = np.array([2.0, 3.0])

    assert inclusion_rate(left, right, predicted) == 0.5
    assert mean_error_ic(left, right, predicted, error_type="absolute") == 0.5
    assert mean_error_ic(left, right, predicted, error_type="squared") == 0.5


def test_exact_interval_matching_uses_numerical_tolerance():
    exact_time = np.array([2.0])
    predicted = np.array([2.0 + 1e-10])

    assert inclusion_rate(exact_time, exact_time, predicted) == 1.0
    assert (
        mean_error_ic(exact_time, exact_time, predicted, error_type="absolute") == 0.0
    )


def test_uncensored_brier_score_without_training_data():
    evaluator = IntervalCenEvaluator(
        pred_survs=np.array(
            [
                [1.0, 0.8, 0.4, 0.1],
                [1.0, 0.6, 0.3, 0.05],
                [1.0, 0.9, 0.5, 0.2],
            ]
        ),
        time_coordinates=np.array([0.0, 1.0, 2.0, 3.0]),
        left_limits=np.array([0.5, 1.0, 2.0]),
        right_limits=np.array([1.5, 2.0, np.inf]),
    )

    brier = evaluator.brier_score(target_time=None, method="uncensored")

    assert np.isfinite(brier)
    with pytest.raises(TypeError, match="Train set information is missing"):
        evaluator.brier_score(target_time=None, method="Tsouprou-marginal")
