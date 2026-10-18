"""Cross-metric checks for events-first IPCW and the shared extrapolated tail."""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from SurvivalEVAL import PointEvaluator, SingleTimeEvaluator, SurvivalEvaluator
from SurvivalEVAL.Evaluations.BrierScore import brier_multiple_points
from SurvivalEVAL.Evaluations.Discrimination import (
    auc,
    concordance,
    concordance_cr_per_event,
)
from SurvivalEVAL.Evaluations.Discrimination.Distributional.TimeDependentConcordance import (
    concordance_time_dependent,
)
from SurvivalEVAL.Evaluations.MeanError import mean_error
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier

METRICS = ["brier", "auc", "concordance", "dynamic", "competing", "error"]


def _score(metric: str, times: np.ndarray, target: float) -> float:
    events = np.array([1, 1, 0])
    probs = np.array([0.2, 0.6, 0.4])
    train_times = np.array([2.0, 2.0, 4.0, 6.0])
    train_events = np.array([1, 0, 0, 1])
    if metric == "brier":
        return float(
            brier_multiple_points(
                probs[:, None],
                times,
                events,
                train_times,
                train_events,
                np.asarray([target]),
                ipcw=True,
            )[0]
        )
    if metric == "auc":
        return auc(
            probs, times, events, target, train_times, train_events, method="IPCW"
        )
    if metric == "concordance":
        return concordance(
            probs, times, events, train_times, train_events, method="IPCW"
        )[0]
    if metric == "dynamic":
        risks = -np.tile(probs[:, None], (1, 2))
        return concordance_time_dependent(
            risks, times, events, train_times, train_events, method="IPCW"
        )[0]
    if metric == "competing":
        return concordance_cr_per_event(
            -probs, times, events, train_times, train_events, method="IPCW"
        )[0]
    return mean_error(
        times - 1, times, events, train_times, train_events, method="IPCW-D"
    )


@pytest.mark.parametrize(
    "metric, expected",
    [
        ("brier", 82 / 75),
        ("auc", 2 / 11),
        ("concordance", 8 / 89),
        ("dynamic", 8 / 89),
        ("competing", 8 / 89),
        ("error", 11 / 4),
    ],
)
def test_ipcw_metrics_share_positive_linear_extrapolation(
    metric: str,
    expected: float,
) -> None:
    # Reverse KM is 2/3 after 2, 1/3 after 4, and 1/3 at 6.
    # The existing line from (0, 1) to (6, 1/3) gives G(7)=2/9.
    assert _score(metric, np.array([2.0, 7.0, 8.0]), 7) == pytest.approx(expected)


@pytest.mark.parametrize("metric", METRICS)
def test_ipcw_metrics_reject_nonpositive_contributing_tail(metric: str) -> None:
    with pytest.raises(ValueError, match="Censoring survival"):
        _score(metric, np.array([2.0, 10.0, 12.0]), 10)


@pytest.mark.parametrize("metric", METRICS)
def test_ipcw_metrics_reject_nonfinite_contributing_probabilities(
    metric: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        KaplanMeier,
        "predict",
        lambda self, times, **kwargs: np.full(np.asarray(times).shape, np.nan),
    )
    with pytest.raises(ValueError, match="Censoring survival"):
        _score(metric, np.array([2.0, 4.0, 6.0]), 4)


def test_brier_vectors_use_only_contributing_event_times_and_horizons() -> None:
    times = np.array([2.0, 3.0, 4.0, 4.0, 6.0, 20.0])
    events = np.array([1, 1, 1, 0, 0, 0])
    probs = np.array([0.2, 0.3, 0.4, 0.8, 0.9, 0.7])
    grid = np.array([4.0, 2.0, 8.0, 6.0, 25.0])
    train_times = np.array([2.0, 2.0, 4.0, 6.0])
    train_events = np.array([1, 0, 0, 1])
    # Cases contribute .04 + .09/(2/3) + .16/(2/3) = .415 after 4.
    # At 25 there are no controls, so zero G(25) must not invalidate that score.
    expected = [5 / 48, 103 / 600, 49 / 240, 143 / 1200, 83 / 1200]
    predictions = np.tile(probs[:, None], (1, grid.size))
    arrays = (times, events, predictions, train_times, train_events, grid)
    before = [array.copy() for array in arrays]
    for array in arrays:
        array.flags.writeable = False
    actual = brier_multiple_points(
        predictions, times, events, train_times, train_events, grid, ipcw=True
    )
    np.testing.assert_allclose(actual, expected)
    for i, horizon in enumerate(grid):
        np.testing.assert_allclose(
            brier_multiple_points(
                predictions[:, i : i + 1],
                times,
                events,
                train_times,
                train_events,
                np.asarray([horizon]),
                ipcw=True,
            ),
            [expected[i]],
        )
    for array, original in zip(arrays, before):
        np.testing.assert_array_equal(array, original)


@pytest.mark.parametrize("ties", ["None", "Risk", "Time", "All"])
def test_terminal_censoring_jump_has_valid_left_weights_for_event_ties(
    ties: str,
) -> None:
    times, events = np.array([1.0, 2.0, 2.0]), np.ones(3, dtype=bool)
    train_times, train_events = np.array([1.0, 2.0]), np.array([1, 0])
    predictions = np.array([1.0, 2.0, 3.0])
    expected = (5 / 6, 2.5, 3) if ties in {"Time", "All"} else (1, 2, 2)
    np.testing.assert_allclose(
        concordance(
            predictions,
            times,
            events,
            train_times,
            train_events,
            method="Uno",
            ties=ties,
        ),
        expected,
    )
    np.testing.assert_allclose(
        concordance_cr_per_event(
            -predictions,
            times,
            events,
            train_times,
            train_events,
            method="IPCW",
            ties=ties,
        ),
        expected,
    )
    np.testing.assert_allclose(
        concordance_time_dependent(
            -np.tile(predictions[:, None], (1, 3)),
            times,
            events,
            train_times,
            train_events,
            method="IPCW",
            ties=ties,
        ),
        expected,
    )


def test_competing_pairs_use_left_weights_at_both_event_times() -> None:
    # Three concordant type A pairs at 2 have weight 1. The discordant type B
    # pair anchored at 6 with competing event at 4 has weight 3 * 3/2 = 9/2.
    np.testing.assert_allclose(
        concordance_cr_per_event(
            [3, 1, 2, 0],
            [2, 4, 4, 6],
            [1, 2, 0, 1],
            [2, 2, 4, 6],
            [1, 0, 0, 1],
            method="IPCW",
        ),
        (2 / 5, 3, 15 / 2),
    )


def test_brier_ipcw_is_default_for_low_level_function() -> None:
    probs, times, events = (
        np.array([[0.2], [0.8]]),
        np.array([1.0, 3.0]),
        np.array([1, 0]),
    )
    train_times, train_events = np.array([1.0, 2.0, 4.0, 6.0]), np.array([1, 0, 0, 1])
    grid = np.array([3.0])
    default = brier_multiple_points(
        probs, times, events, train_times, train_events, grid
    )
    # The event weight is 1; the control at 3 has weight 1/(2/3).
    np.testing.assert_allclose(default, [0.05])
    np.testing.assert_allclose(
        default,
        brier_multiple_points(
            probs, times, events, train_times, train_events, grid, ipcw=True
        ),
    )
    with pytest.raises(ValueError, match="Training"):
        brier_multiple_points(probs, times, events, None, None, grid)
    np.testing.assert_allclose(
        brier_multiple_points(probs, times, events, None, None, grid, ipcw=False),
        [0.04],
    )


def _evaluator(
    kind: str,
    training: bool,
    *,
    uncensored: bool = True,
) -> SurvivalEvaluator | SingleTimeEvaluator | PointEvaluator:
    outcomes = {
        "event_times": np.array([2.0, 4.0, 6.0]),
        "event_indicators": np.array([1, 1 if uncensored else 0, 1], dtype=bool),
    }
    if training:
        outcomes.update(
            train_event_times=np.array([2.0, 2.0, 4.0, 6.0]),
            train_event_indicators=np.array([1, 0, 0, 1]),
        )
    if kind == "single":
        return SingleTimeEvaluator(
            np.array([0.2, 0.6, 0.4]), target_time=4.0, **outcomes
        )
    if kind == "point":
        return PointEvaluator(np.array([1.0, 5.0, 3.0]), **outcomes)
    return SurvivalEvaluator(
        np.array([[1.0, 0.5, 0.2, 0.0], [1.0, 0.8, 0.6, 0.0], [1.0, 0.9, 0.4, 0.0]]),
        np.array([0.0, 2.0, 4.0, 8.0]),
        **outcomes,
    )


@pytest.mark.parametrize("kind", ["survival", "single"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("uncensored", [False, True])
def test_evaluator_brier_default_and_uncensored_shortcut(
    kind: str,
    training: bool,
    uncensored: bool,
) -> None:
    evaluator = _evaluator(kind, training, uncensored=uncensored)
    kwargs = {"target_time": 4} if kind == "survival" else {}
    plain = evaluator.brier_score(IPCW_weighted=False, **kwargs)
    if not training and not uncensored:
        with pytest.raises(TypeError, match="Train set information is missing"):
            evaluator.brier_score(**kwargs)
        with pytest.raises(TypeError, match="Train set information is missing"):
            evaluator.brier_score(IPCW_weighted=True, **kwargs)
        return
    default = evaluator.brier_score(**kwargs)
    assert isinstance(default, float)
    assert default == pytest.approx(evaluator.brier_score(IPCW_weighted=True, **kwargs))
    if uncensored:
        # Statuses at 4 are [0, 0, 1], with predictions [.2, .6, .4].
        assert default == pytest.approx(19 / 75)
        assert default == pytest.approx(plain)
    else:
        assert default != pytest.approx(plain)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("uncensored", [False, True])
def test_evaluator_brier_multiple_and_ibs_defaults(
    training: bool,
    uncensored: bool,
) -> None:
    grid = np.array([2.0, 4.0, 6.0])
    evaluator = _evaluator("survival", training, uncensored=uncensored)
    plain = evaluator.brier_score_multiple_points(grid, IPCW_weighted=False)
    expected_plain = trapezoid(plain, grid) / (grid[-1] - grid[0])
    assert evaluator.integrated_brier_score(
        target_times=grid, IPCW_weighted=False
    ) == pytest.approx(expected_plain)
    if not training and not uncensored:
        with pytest.raises(TypeError, match="Train set information is missing"):
            evaluator.brier_score_multiple_points(grid)
        with pytest.raises(TypeError, match="Train set information is missing"):
            evaluator.brier_score_multiple_points(grid, IPCW_weighted=True)
        with pytest.raises(TypeError, match="Train set information is missing"):
            evaluator.integrated_brier_score(target_times=grid)
        with pytest.raises(TypeError, match="Train set information is missing"):
            evaluator.integrated_brier_score(target_times=grid, IPCW_weighted=True)
        return
    default = evaluator.brier_score_multiple_points(grid)
    np.testing.assert_allclose(
        default, evaluator.brier_score_multiple_points(grid, IPCW_weighted=True)
    )
    expected_ibs = trapezoid(default, grid) / (grid[-1] - grid[0])
    assert evaluator.integrated_brier_score(target_times=grid) == pytest.approx(
        expected_ibs
    )
    assert evaluator.integrated_brier_score(
        target_times=grid, IPCW_weighted=True
    ) == pytest.approx(expected_ibs)
    if uncensored:
        np.testing.assert_allclose(default, plain)
    else:
        assert not np.allclose(default, plain)


@pytest.mark.parametrize("kind", ["survival", "point"])
@pytest.mark.parametrize("method", ["Harrell", "Naive", "Uno", "IPCW", "Margin"])
@pytest.mark.parametrize("training", [False, True])
def test_evaluator_static_concordance_uses_harrell_on_uncensored_data(
    kind: str,
    method: str,
    training: bool,
) -> None:
    evaluator = _evaluator(kind, training)
    # Two correctly ranked pairs and one incorrectly ranked pair.
    np.testing.assert_allclose(evaluator.concordance(method=method), (2 / 3, 2, 3))
    np.testing.assert_allclose(evaluator.concordance(method=method, tau=4), (1, 2, 2))


@pytest.mark.parametrize("kind", ["survival", "point"])
@pytest.mark.parametrize("metric", ["mae", "mse", "rmse"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize(
    "method", ["Uncensored", "Hinge", "Margin", "IPCW-T", "IPCW-D", "Pseudo_obs"]
)
def test_evaluator_error_methods_use_uncensored_scores(
    kind: str,
    metric: str,
    training: bool,
    method: str,
) -> None:
    evaluator = _evaluator(kind, training)
    predictions = (
        evaluator.predicted_event_times
        if kind == "survival"
        else np.array([1.0, 5.0, 3.0])
    )
    errors = predictions - evaluator.event_times
    expected = np.abs(errors).mean() if metric == "mae" else np.square(errors).mean()
    if metric == "rmse":
        expected = np.sqrt(expected)
    assert getattr(evaluator, metric)(method=method) == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["survival", "point"])
@pytest.mark.parametrize("metric", ["mae", "mse", "rmse"])
def test_evaluator_ipcw_d_still_weights_errors_when_test_set_contains_censoring(
    kind: str,
    metric: str,
) -> None:
    with pytest.raises(ValueError, match="training set values"):
        getattr(_evaluator(kind, False, uncensored=False), metric)(method="IPCW-D")
    evaluator = _evaluator(kind, True, uncensored=False)
    predictions = (
        evaluator.predicted_event_times
        if kind == "survival"
        else np.array([1.0, 5.0, 3.0])
    )
    errors = (
        predictions[evaluator.event_indicators]
        - evaluator.event_times[evaluator.event_indicators]
    )
    # Observed events at 2 and 6 have inverse censoring weights 1 and 3.
    losses = np.abs(errors) if metric == "mae" else np.square(errors)
    expected = np.mean(losses * np.array([1.0, 3.0]))
    if metric == "rmse":
        expected = np.sqrt(expected)
    assert getattr(evaluator, metric)(method="IPCW-D") == pytest.approx(expected)


@pytest.mark.parametrize("method", ["Antolini", "Naive", "IPCW", "iPcW"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("risks", ["Survival", "Hazard"])
def test_evaluator_dynamic_concordance_uses_antolini_on_uncensored_data(
    method: str,
    training: bool,
    risks: str,
) -> None:
    evaluator = _evaluator("survival", training)
    np.testing.assert_allclose(
        evaluator.concordance_time_dependent(method=method, risks=risks), (2 / 3, 2, 3)
    )
    np.testing.assert_allclose(
        evaluator.concordance_time_dependent(method=method, risks=risks, tau=4),
        (1, 2, 2),
    )


def test_evaluator_dynamic_ipcw_still_requires_training_when_test_set_has_censoring() -> (
    None
):
    evaluator = _evaluator("survival", False, uncensored=False)
    with pytest.raises(TypeError, match="Train set information is missing"):
        evaluator.concordance_time_dependent(method="IPCW")


def test_ipcw_brier_recovers_complete_data_with_independent_discrete_censoring() -> (
    None
):
    # The Cartesian product makes latent event times and censoring independent.
    true_times = np.repeat([1.0, 2.0, 3.0, 5.0, 6.0], 3)
    censor_times = np.tile([2.0, 3.0, 10.0], 5)
    observed = np.minimum(true_times, censor_times)
    events = true_times <= censor_times
    grid = np.array([2.0, 3.0, 5.0])
    hazards = np.repeat([0.1, 0.7, 1.2, 0.3, 0.8], 3)
    predictions = np.exp(-hazards[:, None] * grid)
    expected = np.mean((predictions - (true_times[:, None] > grid)) ** 2, axis=0)
    np.testing.assert_allclose(
        brier_multiple_points(
            predictions, observed, events, observed, events, grid, ipcw=True
        ),
        expected,
    )


@pytest.mark.parametrize("training", [False, True])
def test_brier_rejects_nonbinary_event_indicators(training: bool) -> None:
    test_events = np.array([1, 0] if training else [1, 2])
    train_events = np.array([1, 2] if training else [1, 0])
    with pytest.raises(ValueError, match="only 0 and 1"):
        brier_multiple_points(
            np.array([[0.2], [0.8]]),
            np.array([1.0, 3.0]),
            test_events,
            np.array([2.0, 4.0]),
            train_events,
            np.array([2.0]),
            ipcw=True,
        )
