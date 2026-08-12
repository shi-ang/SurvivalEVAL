import tracemalloc

import numpy as np
import pytest

from SurvivalEVAL.Evaluations.BrierScore import (
    brier_multiple_points,
    brier_multiple_points_ic,
    brier_score_ic,
    single_brier_score,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier


@pytest.mark.parametrize("ipcw", [True, False])
def test_single_brier_score_treats_censored_at_target_time_as_event_free(
    ipcw: bool,
) -> None:
    score = single_brier_score(
        preds=np.array([0.2, 0.8, 0.9]),
        event_times=np.array([1.0, 2.0, 3.0]),
        event_indicators=np.array([1, 0, 0]),
        train_event_times=np.array([4.0, 5.0, 6.0]),
        train_event_indicators=np.array([1, 1, 1]),
        target_time=2.0,
        ipcw=ipcw,
    )

    assert score == pytest.approx(0.03)


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
def test_brier_multiple_points_matches_single(ipcw: bool) -> None:
    pred_mat = np.array(
        [
            [0.85, 0.70, 0.55],
            [0.80, 0.60, 0.40],
            [0.65, 0.50, 0.35],
            [0.60, 0.45, 0.30],
        ]
    )
    event_times = np.array([2.0, 5.0, 9.0, 12.0])
    event_indicators = np.array([1, 0, 1, 1])
    train_event_times = np.array([1.0, 2.5, 4.5, 6.0, 8.0, 9.5, 11.0, 13.0])
    train_event_indicators = np.array([1, 1, 0, 1, 1, 0, 1, 1])
    target_times = np.array([3.0, 6.0, 10.0])

    multi_scores = brier_multiple_points(
        pred_mat=pred_mat,
        event_times=event_times,
        event_indicators=event_indicators,
        train_event_times=train_event_times,
        train_event_indicators=train_event_indicators,
        target_times=target_times,
        ipcw=ipcw,
    )

    single_scores = np.array(
        [
            single_brier_score(
                preds=pred_mat[:, idx],
                event_times=event_times,
                event_indicators=event_indicators,
                train_event_times=train_event_times,
                train_event_indicators=train_event_indicators,
                target_time=target_times[idx],
                ipcw=ipcw,
            )
            for idx in range(target_times.size)
        ]
    )

    np.testing.assert_allclose(multi_scores, single_scores, rtol=1e-6, atol=1e-8)


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


def test_brier_multiple_points_ic_conditional_matches_single() -> None:
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

    def recording_predict(self, prediction_times):
        query_shapes.append(np.asarray(prediction_times).shape)
        return original_predict(self, prediction_times)

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

    assert query_shapes == [(4,), (3,)]
