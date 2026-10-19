import tracemalloc

import numpy as np
import pandas as pd
import pytest
from lifelines import WeibullAFTFitter

from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.evaluations.prediction_error import (
    brier_multiple_points,
    brier_multiple_points_ic,
    brier_score_ic,
)
from SurvivalEVAL.evaluations.prediction_error.brier_score import (
    _predict_survival_at_observed_times,
)
from SurvivalEVAL.nonparametric_estimator.single_event import KaplanMeier


def test_ipcw_brier_and_ibs_use_reverse_km_for_mixed_training_ties():
    grid = np.array([6.0, 13.0, 16.0])
    predictions = np.tile([0.8, 0.6, 0.4], (4, 1))
    times = np.array([5.0, 12.0, 15.0, 18.0])
    indicators = np.array([True, True, False, True])
    train_times = np.array([10.0, 10.0, 15.0, 20.0])
    train_indicators = np.array([True, False, True, True])
    expected = [0.19, 0.345, 0.235]

    multiple = brier_multiple_points(
        predictions, times, indicators, train_times, train_indicators, grid, ipcw=True
    )
    np.testing.assert_allclose(multiple, expected)

    evaluator = SurvivalEvaluator(
        pred_survs=np.column_stack((np.ones(4), predictions)),
        time_coordinates=np.r_[0, grid],
        event_times=times,
        event_indicators=indicators,
        train_event_times=train_times,
        train_event_indicators=train_indicators,
    )
    assert evaluator.integrated_brier_score(
        target_times=grid, IPCW_weighted=True
    ) == pytest.approx(0.27425)


@pytest.mark.parametrize("zero_survival", [False, True])
def test_ipcw_brier_uses_left_limit_even_when_right_survival_is_zero(zero_survival):
    train_times = np.array([10.0, 10.0] if zero_survival else [10, 10, 15, 20])
    train_indicators = np.array(
        [True, False] if zero_survival else [True, False, True, True]
    )
    args = {
        "event_times": np.array([10.0, 12.0]),
        "event_indicators": np.array([True, False]),
        "train_event_times": train_times,
        "train_event_indicators": train_indicators,
    }
    expected = 0.04
    with np.errstate(divide="raise", invalid="raise"):
        multiple = brier_multiple_points(
            pred_mat=np.array([[0.2], [0.8]]),
            target_times=np.array([10.0]),
            ipcw=True,
            **args,
        )
    np.testing.assert_allclose(multiple, [expected])


def test_brier_multiple_points_matches_float64_for_float32_inputs() -> None:
    preds = np.array([0.2, 0.4, 0.8], dtype=np.float32)
    event_times = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    event_indicators = np.array([True, False, True])

    result = brier_multiple_points(
        preds[:, None],
        event_times,
        event_indicators,
        train_event_times=event_times,
        train_event_indicators=event_indicators,
        target_times=np.array([2.0]),
        ipcw=False,
    )
    expected = brier_multiple_points(
        preds.astype(np.float64)[:, None],
        event_times.astype(np.float64),
        event_indicators,
        train_event_times=event_times.astype(np.float64),
        train_event_indicators=event_indicators,
        target_times=np.array([2.0]),
        ipcw=False,
    )

    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("ipcw", [True, False])
def test_brier_multiple_points_treats_censored_at_target_time_as_event_free(
    ipcw: bool,
) -> None:
    scores = brier_multiple_points(
        pred_mat=np.array([[0.2], [0.8], [0.9]]),
        event_times=np.array([1.0, 2.0, 3.0]),
        event_indicators=np.array([1, 0, 0]),
        train_event_times=np.array([4.0, 5.0, 6.0]),
        train_event_indicators=np.array([1, 1, 1]),
        target_times=np.array([2.0]),
        ipcw=ipcw,
    )

    np.testing.assert_allclose(scores, [0.03])


@pytest.mark.parametrize("ipcw", [True, False])
def test_brier_multiple_points_matches_hand_calculated_formula(ipcw: bool) -> None:
    predictions = np.array(
        [[0.85, 0.7, 0.55], [0.8, 0.6, 0.4], [0.65, 0.5, 0.35], [0.6, 0.45, 0.3]]
    )
    times = np.array([2.0, 5.0, 9.0, 12.0])
    events = np.array([1, 0, 1, 1], dtype=bool)
    train_times = np.array([1.0, 2.5, 4.5, 6.0, 8.0, 9.5, 11.0, 13.0])
    train_events = np.array([1, 1, 0, 1, 1, 0, 1, 1])
    targets = np.array([3.0, 6.0, 10.0])
    # Censoring jumps at 4.5 (risk set 6) and 9.5 (risk set 3).
    event_weights = np.array([1.0, 0.0, 6 / 5, 9 / 5]) if ipcw else np.ones(4)
    target_weights = np.array([1.0, 6 / 5, 9 / 5]) if ipcw else np.ones(3)
    expected = []
    for k, target in enumerate(targets):
        cases = events & (times <= target)
        controls = (times > target) | (~events & (times == target))
        event_loss = np.sum(predictions[cases, k] ** 2 * event_weights[cases])
        control_loss = np.sum((1 - predictions[controls, k]) ** 2) * target_weights[k]
        expected.append((event_loss + control_loss) / len(times))
    np.testing.assert_allclose(
        brier_multiple_points(
            predictions, times, events, train_times, train_events, targets, ipcw=ipcw
        ),
        expected,
    )


def test_brier_multiple_points_ic_matches_single_uncensored() -> None:
    pred_mat = np.array(
        [
            [0.90, 0.75, 0.50],
            [0.85, 0.65, 0.45],
            [0.70, 0.55, 0.35],
            [0.60, 0.40, 0.25],
        ]
    )
    left_limits = np.array([0.0, 2.0, 4.0, 6.0])
    right_limits = np.array([1.0, 3.5, np.inf, 8.5])
    target_times = np.array([0.5, 2.5, 7.0])

    multi_scores = brier_multiple_points_ic(
        pred_mat=pred_mat,
        left_limits=left_limits,
        right_limits=right_limits,
        target_times=target_times,
        method="uncensored",
    )

    single_scores = np.array(
        [
            brier_score_ic(
                preds=pred_mat[:, idx],
                left_limits=left_limits,
                right_limits=right_limits,
                target_time=target_times[idx],
                method="uncensored",
            )
            for idx in range(target_times.size)
        ]
    )

    np.testing.assert_allclose(
        multi_scores, single_scores, rtol=1e-6, atol=1e-8, equal_nan=True
    )


def test_brier_score_ic_tsouprou_treats_exact_event_time_as_dead() -> None:
    train_left_limits = np.array([0.5, 1.0, 2.0])
    train_right_limits = np.array([0.5, 1.0, 2.0])

    score = brier_score_ic(
        preds=np.array([0.0]),
        left_limits=np.array([1.0]),
        right_limits=np.array([1.0]),
        train_left_limits=train_left_limits,
        train_right_limits=train_right_limits,
        target_time=1.0,
        method="Tsouprou-marginal",
    )

    assert score == pytest.approx(0.0)


def test_brier_multiple_points_ic_tsouprou_uses_open_closed_interval() -> None:
    train_left_limits = np.array([0.5, 1.0, 2.0])
    train_right_limits = np.array([0.5, 1.0, 2.0])

    scores = brier_multiple_points_ic(
        pred_mat=np.array([[1.0, 0.0, 0.0]]),
        left_limits=np.array([1.0]),
        right_limits=np.array([1.0]),
        target_times=np.array([0.5, 1.0, 1.5]),
        train_left_limits=train_left_limits,
        train_right_limits=train_right_limits,
        method="Tsouprou-marginal",
    )

    np.testing.assert_allclose(scores, [0.0, 0.0, 0.0])


@pytest.mark.parametrize("repeats", [1, 80])
def test_brier_multiple_points_ic_conditional_matches_single(repeats) -> None:
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=30)
    latent_times = np.exp(
        1.0 - 0.3 * x_train + rng.normal(scale=0.25, size=x_train.shape[0])
    )
    train_left = np.floor(latent_times * 2) / 2
    train_right = train_left + 0.5

    x = np.array([-1.0, -0.2, 0.5, 1.2])
    left = np.array([1.0, 1.5, 2.0, 2.5])
    right = np.array([2.0, 2.5, 3.0, 4.0])
    target_times = np.array([1.25, 2.25, 3.25])
    pred_mat = np.array(
        [
            [0.90, 0.70, 0.50],
            [0.85, 0.60, 0.40],
            [0.80, 0.55, 0.30],
            [0.75, 0.45, 0.20],
        ]
    )
    x = np.tile(x, repeats)
    left = np.tile(left, repeats)
    right = np.tile(right, repeats)
    pred_mat = np.tile(pred_mat, (repeats, 1))

    multi_scores = brier_multiple_points_ic(
        pred_mat=pred_mat,
        left_limits=left,
        right_limits=right,
        target_times=target_times,
        train_left_limits=train_left,
        train_right_limits=train_right,
        x=x,
        x_train=x_train,
        method="Tsouprou-conditional",
    )
    single_scores = np.array(
        [
            brier_score_ic(
                preds=pred_mat[:, index],
                left_limits=left,
                right_limits=right,
                train_left_limits=train_left,
                train_right_limits=train_right,
                x=x,
                x_train=x_train,
                target_time=target_time,
                method="Tsouprou-conditional",
            )
            for index, target_time in enumerate(target_times)
        ]
    )

    np.testing.assert_allclose(multi_scores, single_scores, rtol=1e-6, atol=1e-8)


@pytest.fixture(scope="module", params=[1, 2])
def fitted_aft_model(request):
    rng = np.random.default_rng(93)
    features = pd.DataFrame(
        rng.normal(size=(60, request.param)),
        columns=[f"feature_{i}" for i in range(request.param)],
    )
    latent_times = np.exp(
        1.0 - 0.3 * features.iloc[:, 0] + rng.normal(scale=0.3, size=60)
    )
    training = features.assign(left=np.floor(latent_times * 2) / 2)
    training["right"] = training["left"] + 0.5
    model = WeibullAFTFitter().fit_interval_censoring(training, "left", "right")
    return model, features.columns


@pytest.mark.parametrize("n_samples", [1, 255, 256, 257, 700])
def test_paired_aft_predictions_match_full_diagonal(
    fitted_aft_model,
    n_samples,
    monkeypatch,
):
    model, columns = fitted_aft_model
    rng = np.random.default_rng(94)
    features = pd.DataFrame(
        rng.normal(size=(n_samples, len(columns))),
        columns=columns,
        index=rng.permutation(n_samples) + 10,
    )
    times = rng.integers(0, 12, size=n_samples).astype(float)
    times[0] = np.inf
    times.flags.writeable = False
    expected = model.predict_survival_function(features, times=times).to_numpy()
    expected = expected.diagonal().copy()

    original_predict = model.predict_survival_function
    requests = []

    def record_predict(batch_features, *, times):
        requests.append((len(batch_features), times.size))
        return original_predict(batch_features, times=times)

    monkeypatch.setattr(model, "predict_survival_function", record_predict)
    actual = _predict_survival_at_observed_times(model, features, times)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert sum(n_rows for n_rows, _ in requests) == n_samples
    assert all(n_rows == n_times <= 256 for n_rows, n_times in requests)


@pytest.mark.parametrize("metric", ["right", "interval"])
def test_multi_time_brier_uses_one_dense_float_workspace(metric: str) -> None:
    n_samples = 4_000
    n_times = 80
    predictions = np.full((n_samples, n_times), 0.5)
    target_times = np.linspace(0.0, 10.0, n_times)
    event_times = np.linspace(0.0, 10.0, n_samples)
    event_indicators = np.arange(n_samples) % 2
    left_limits = np.linspace(0.0, 9.0, n_samples)
    right_limits = left_limits + 2.0

    # Inputs are allocated before tracing. The optimized kernels should need
    # one N x T float workspace plus a boolean mask, not several float copies.
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    baseline_bytes, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        if metric == "right":
            brier_multiple_points(
                pred_mat=predictions,
                event_times=event_times,
                event_indicators=event_indicators,
                train_event_times=None,
                train_event_indicators=None,
                target_times=target_times,
                ipcw=False,
            )
        else:
            brier_multiple_points_ic(
                pred_mat=predictions,
                left_limits=left_limits,
                right_limits=right_limits,
                target_times=target_times,
                method="uncensored",
            )
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert peak_bytes - baseline_bytes < 1.6 * predictions.nbytes


def test_ipcw_brier_queries_censoring_model_with_vectors(monkeypatch) -> None:
    original_predict = KaplanMeier.predict
    query_shapes = []

    def recording_predict(self, prediction_times, *, side="right"):
        query_shapes.append(np.asarray(prediction_times).shape)
        assert side == "left"
        return original_predict(self, prediction_times, side=side)

    monkeypatch.setattr(KaplanMeier, "predict", recording_predict)

    brier_multiple_points(
        pred_mat=np.full((4, 3), 0.5),
        event_times=np.array([1.0, 2.0, 3.0, 4.0]),
        event_indicators=np.array([1, 0, 1, 1]),
        train_event_times=np.array([1.0, 2.0, 3.0, 4.0]),
        train_event_indicators=np.array([1, 0, 1, 1]),
        target_times=np.array([1.0, 2.0, 3.0]),
        ipcw=True,
    )

    assert query_shapes == [(2,), (3,)]
