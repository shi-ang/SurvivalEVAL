import numpy as np
import pytest

from SurvivalEVAL.Evaluations.BrierScore import (
    brier_multiple_points,
    brier_multiple_points_ic,
    brier_score_ic,
    single_brier_score,
)


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
