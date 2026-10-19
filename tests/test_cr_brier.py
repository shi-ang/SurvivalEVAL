"""Competing-risk Brier scores checked against direct subject-wise calculations."""

import importlib
import tracemalloc
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import PchipInterpolator

from SurvivalEVAL import (
    CompetingRisksEvaluator,
    SurvivalEvaluator,
    brier_multiple_points,
    brier_multiple_points_cr,
    brier_multiple_points_cr_per_event,
)


def _direct_brier(
    probabilities: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    targets: np.ndarray,
    train_times: np.ndarray,
    train_events: np.ndarray,
    cause: int,
    ipcw: bool,
) -> np.ndarray:
    """Enumerate subjects and form reverse-KM products without package helpers."""

    def censoring_survival(query: float) -> float:
        survival = 1.0
        for time in np.unique(train_times[(train_events == 0) & (train_times < query)]):
            censorings = np.count_nonzero((train_times == time) & (train_events == 0))
            risk_set = np.count_nonzero(train_times > time) + censorings
            survival *= 1 - censorings / risk_set
        return survival

    scores = np.zeros(targets.size)
    for j, target in enumerate(targets):
        for i, (time, event) in enumerate(zip(times, events)):
            if event > 0 and time <= target:
                outcome, query = float(event == cause), time
            elif time >= target:
                outcome, query = 0.0, target
            else:
                continue
            weight = 1 / censoring_survival(query) if ipcw else 1.0
            scores[j] += weight * (outcome - probabilities[i, j]) ** 2
    return scores / times.size


@pytest.fixture
def cr_data() -> tuple[np.ndarray, ...]:
    times = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 6.0])
    events = np.array([1, 2, 0, 0, 3, 1])
    grid = np.array([0.0, 2.0, 4.0, 8.0])
    terminal = np.random.default_rng(73).dirichlet(np.ones(4), times.size)[:, :3]
    cifs = terminal[:, :, None] * np.array([0.0, 0.1, 0.5, 1.0])
    return (
        cifs,
        grid,
        times,
        events,
        np.array([1.0, 2.0, 2.0, 3.0, 5.0, 8.0]),
        np.array([2, 1, 0, 3, 0, 1]),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ipcw", [False, True])
def test_scores_match_independent_formula(
    cr_data: tuple,
    dtype: type,
    ipcw: bool,
) -> None:
    cifs, _, times, events, train_times, train_events = cr_data
    targets = np.array([3.0, 0.0, 2.0, 1.0, 3.0, 5.0], dtype=dtype)
    # Non-contiguous and read-only, with unsorted and repeated evaluation times.
    storage = np.zeros((times.size, 3, targets.size * 2), dtype=dtype)
    storage[..., ::2] = cifs[..., -1, None] * targets / 8
    predictions = storage[..., ::2]
    for values in (predictions, times, events, targets, train_times, train_events):
        values.flags.writeable = False
    before = predictions.copy()
    expected = []
    for cause in range(1, 4):
        score = _direct_brier(
            predictions[:, cause - 1],
            times,
            events,
            targets,
            train_times,
            train_events,
            cause,
            ipcw,
        )
        expected.append(score)
        np.testing.assert_allclose(
            brier_multiple_points_cr_per_event(
                predictions[:, cause - 1],
                times,
                events,
                train_times,
                train_events,
                targets,
                ipcw=ipcw,
                event_of_interest=cause,
            ),
            score,
        )
    np.testing.assert_allclose(
        brier_multiple_points_cr(
            predictions, times, events, train_times, train_events, targets, ipcw=ipcw
        ),
        np.mean(expected, axis=0),
    )
    np.testing.assert_array_equal(predictions, before)


@pytest.mark.parametrize("ipcw, expected", [(False, 0.29 / 4), (True, 0.49 / 4)])
def test_competing_events_and_same_time_censoring_contribute(
    ipcw: bool,
    expected: float,
) -> None:
    # Reverse KM removes the event at 1 before its censoring update: G(1-)=1,
    # G(2-)=1/2. The fourth subject was censored earlier and contributes zero.
    score = brier_multiple_points_cr_per_event(
        [[0.8], [0.3], [0.4], [0.6]],
        [2, 1, 2, 0],
        [1, 2, 0, 0],
        [1, 1, 3],
        [2, 0, 1],
        [2],
        ipcw=ipcw,
    )
    np.testing.assert_allclose(score, [expected])


@pytest.mark.parametrize("ipcw", [False, True])
def test_single_cause_matches_survival_scores(ipcw: bool) -> None:
    probabilities = np.array([[0.1, 0.8], [0.2, 0.5], [0.4, 0.6]])
    args = ([1, 2, 3], [1, 0, 1], [1, 2, 4], [0, 1, 1], [1, 3])
    expected = brier_multiple_points(1 - probabilities, *args, ipcw=ipcw)
    np.testing.assert_allclose(
        brier_multiple_points_cr_per_event(probabilities, *args, ipcw=ipcw), expected
    )
    np.testing.assert_allclose(
        brier_multiple_points_cr(probabilities[:, None, :], *args, ipcw=ipcw), expected
    )


def test_absent_causes_have_scores_and_remain_in_mean() -> None:
    predictions = np.array([[[0.8], [0.1]], [[0.2], [0.4]]])
    result = brier_multiple_points_cr(
        predictions, [1, 2], [1, 0], None, None, [1], ipcw=False
    )
    # Cause 1: (.04+.04)/2; absent cause 2: (.01+.16)/2.
    np.testing.assert_allclose(result, [0.0625])
    np.testing.assert_allclose(
        brier_multiple_points_cr_per_event(
            predictions[:, 1],
            [1, 2],
            [1, 0],
            None,
            None,
            [1],
            ipcw=False,
            event_of_interest=7,
        ),
        [0.085],
    )


def test_only_contributing_censoring_probabilities_are_required() -> None:
    args = ([[0.3], [0.8]], [1, 3], [2, 1], [1, 2], [1, 0])
    np.testing.assert_allclose(brier_multiple_points_cr_per_event(*args, [1]), [0.365])
    with pytest.raises(ValueError, match="Censoring survival"):
        brier_multiple_points_cr_per_event(*args, [3])
    # G(2)=0, but left-limit G(2-)=1 for both the observed event and censoring.
    np.testing.assert_allclose(
        brier_multiple_points_cr_per_event(
            [[0.8], [0.4]], [2, 2], [1, 0], [1, 2], [1, 0], [2]
        ),
        [0.1],
    )
    # No event-free contribution at 3: G(3-) is unused and may be zero.
    np.testing.assert_allclose(
        brier_multiple_points_cr_per_event([[0.3]], [1], [2], [1, 2], [1, 0], [3]),
        [0.09],
    )


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize(
    "changes, message",
    [
        ({"event_times": [-1, 2]}, "non-negative"),
        ({"event_indicators": [1, 0.5]}, "integer labels"),
        ({"event_indicators": [1, -1]}, "integer labels"),
        ({"target_times": [np.inf]}, "finite"),
        ({"target_times": [-1]}, "non-negative"),
        ({"pred_mat": [0.1, 0.2]}, "shape"),
    ],
)
def test_invalid_inputs(aggregate: bool, changes: dict, message: str) -> None:
    metric = (
        brier_multiple_points_cr if aggregate else brier_multiple_points_cr_per_event
    )
    kwargs = {
        "pred_mat": [[[0.8], [0.1]], [[0.2], [0.4]]] if aggregate else [[0.8], [0.2]],
        "event_times": [1, 2],
        "event_indicators": [1, 0],
        "train_event_times": None,
        "train_event_indicators": None,
        "target_times": [1],
        "ipcw": False,
    }
    kwargs.update(changes)
    with pytest.raises(ValueError, match=message):
        metric(**kwargs)


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("value", [np.inf, -0.1, 1.1])
def test_rejects_invalid_probabilities(aggregate: bool, value: float) -> None:
    predictions = [[[value]], [[0.2]]] if aggregate else [[value], [0.2]]
    metric = (
        brier_multiple_points_cr if aggregate else brier_multiple_points_cr_per_event
    )
    with pytest.raises(ValueError, match="probabilities"):
        metric(predictions, [1, 2], [1, 0], None, None, [1], ipcw=False)


@pytest.mark.parametrize("cause", [0, -1, True, 1.5])
def test_invalid_selected_cause(cause: int) -> None:
    with pytest.raises(ValueError, match="event_of_interest"):
        brier_multiple_points_cr_per_event(
            [[0.8]], [1], [1], None, None, [1], ipcw=False, event_of_interest=cause
        )


def test_cause_labels_must_fit_aggregate_predictions() -> None:
    with pytest.raises(ValueError, match="number of predicted causes"):
        brier_multiple_points_cr([[[0.8]]], [1], [2], None, None, [1], ipcw=False)


@pytest.mark.parametrize("aggregate", [False, True])
def test_low_level_ipcw_requires_training_even_without_test_censoring(
    aggregate: bool,
) -> None:
    metric = (
        brier_multiple_points_cr if aggregate else brier_multiple_points_cr_per_event
    )
    predictions = [[[0.8]]] if aggregate else [[0.8]]
    with pytest.raises(ValueError, match="Training"):
        metric(predictions, [1], [1], None, None, [1])
    np.testing.assert_allclose(
        metric(predictions, [1], [1], None, None, [1], ipcw=False), [0.04]
    )


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("ipcw", [False, True])
def test_evaluator_matches_direct_calculations(
    cr_data: tuple,
    interpolation: str,
    ipcw: bool,
) -> None:
    cifs, grid, times, events, train_times, train_events = cr_data
    evaluator = CompetingRisksEvaluator(
        cifs,
        grid,
        times,
        events,
        interpolation=interpolation,
        train_event_times=train_times,
        train_event_indicators=train_events,
    )
    targets = np.array([1.0, 2.0, 3.0, 5.0])
    per_cause = []
    for cause in range(1, 4):
        probabilities = np.array(
            [
                (
                    np.interp(targets, grid, curve)
                    if interpolation == "Linear"
                    else PchipInterpolator(grid, curve)(targets)
                )
                for curve in cifs[:, cause - 1]
            ]
        )
        expected = _direct_brier(
            probabilities,
            times,
            events,
            targets,
            train_times,
            train_events,
            cause,
            ipcw,
        )
        per_cause.append(expected)
        np.testing.assert_allclose(
            evaluator.brier_score_multiple_points_per_event(targets, cause, ipcw),
            expected,
        )
        assert evaluator.brier_score_per_event(cause, 3, ipcw) == pytest.approx(
            expected[2]
        )
    np.testing.assert_allclose(
        evaluator.brier_score_multiple_points(targets, ipcw), np.mean(per_cause, axis=0)
    )
    assert evaluator.brier_score(3, ipcw) == pytest.approx(
        np.mean(per_cause, axis=0)[2]
    )
    assert evaluator.brier_score(IPCW_weighted=ipcw) == evaluator.brier_score(
        evaluator.default_target_time, ipcw
    )
    assert evaluator.brier_score_per_event(
        2, IPCW_weighted=ipcw
    ) == evaluator.brier_score_per_event(2, evaluator.default_target_time, ipcw)


@pytest.mark.parametrize("ipcw", [False, True])
@pytest.mark.parametrize("method", ["trapz", "simpson"])
@pytest.mark.parametrize("grid_option", ["explicit", "generated", "inferred"])
def test_integrated_scores(
    cr_data: tuple,
    ipcw: bool,
    method: str,
    grid_option: str,
) -> None:
    cifs, grid, times, events, train_times, train_events = cr_data
    evaluator = CompetingRisksEvaluator(
        cifs,
        grid,
        times,
        events,
        train_event_times=train_times,
        train_event_indicators=train_events,
    )
    targets = {
        "explicit": np.array([1.0, 2.0, 4.0]),
        "generated": np.linspace(0, 8, 5),
        "inferred": np.array([2.0, 3.0]),
    }[grid_option]
    kwargs = (
        {"target_times": targets}
        if grid_option == "explicit"
        else {"num_points": 5} if grid_option == "generated" else {}
    )
    integrate = trapezoid if method == "trapz" else simpson
    scores = []
    for cause in range(1, 4):
        values = evaluator.brier_score_multiple_points_per_event(targets, cause, ipcw)
        expected = integrate(values, x=targets) / (targets[-1] - targets[0])
        scores.append(expected)
        assert evaluator.integrated_brier_score_per_event(
            cause, IPCW_weighted=ipcw, integration_method=method.upper(), **kwargs
        ) == pytest.approx(expected)
    assert evaluator.integrated_brier_score(
        IPCW_weighted=ipcw, integration_method=method, **kwargs
    ) == pytest.approx(np.mean(scores))


def test_ibs_sorting_and_plots(cr_data: tuple) -> None:
    cifs, grid, times, events, train_times, train_events = cr_data
    evaluator = CompetingRisksEvaluator(
        cifs,
        grid,
        times,
        events,
        train_event_times=train_times,
        train_event_indicators=train_events,
    )
    for cause in (None, 2):
        metric = (
            evaluator.integrated_brier_score
            if cause is None
            else evaluator.integrated_brier_score_per_event
        )
        kwargs = {} if cause is None else {"event_of_interest": cause}
        expected = metric(target_times=[1, 2, 4], **kwargs)
        with pytest.warns(UserWarning, match="sorting"):
            result, (fig, ax) = metric(
                target_times=[4, 1, 2], draw_figure=True, **kwargs
            )
        assert result == pytest.approx(expected)
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 4])
        plt.close(fig)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_times": [1]},
        {"target_times": [2, 2]},
        {"target_times": [1, np.inf]},
        {"target_times": [-1, 2]},
        {"num_points": 1},
        {"num_points": 2.5},
        {"target_times": [1, 2], "num_points": 3},
        {"target_times": [1, 9]},
        {"target_times": [1, 2], "integration_method": "unknown"},
    ],
)
def test_invalid_ibs_grids(cr_data: tuple, kwargs: dict) -> None:
    evaluator = CompetingRisksEvaluator(*cr_data[:4])
    for metric in (
        evaluator.integrated_brier_score,
        evaluator.integrated_brier_score_per_event,
    ):
        with pytest.raises(ValueError):
            metric(IPCW_weighted=False, **kwargs)


def test_uncensored_shortcut_and_missing_training(cr_data: tuple) -> None:
    cifs, grid, times, events, _, _ = cr_data
    censored = CompetingRisksEvaluator(cifs, grid, times, events)
    with pytest.raises(ValueError, match="Training"):
        censored.brier_score()
    uncensored = CompetingRisksEvaluator(cifs, grid, times, np.maximum(events, 1))
    for evaluator in (censored, uncensored):
        if evaluator is uncensored:
            assert evaluator.brier_score() == evaluator.brier_score(IPCW_weighted=False)
            assert evaluator.integrated_brier_score(
                num_points=5
            ) == evaluator.integrated_brier_score(num_points=5, IPCW_weighted=False)
            with pytest.raises(ValueError, match="infer"):
                evaluator.integrated_brier_score()
        np.testing.assert_equal(
            evaluator.brier_score_multiple_points([1, 2], False).shape, (2,)
        )


def test_one_censoring_fit_and_vector_queries(
    cr_data: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("SurvivalEVAL.Evaluations.PredictionError.BrierScore")
    original_predict = module.KaplanMeier.predict
    queries = []

    def record_predict(self, values: np.ndarray, *, side: str = "right") -> np.ndarray:
        assert side == "left" and values.ndim == 1
        queries.append(values.shape)
        return original_predict(self, values, side=side)

    monkeypatch.setattr(module.KaplanMeier, "predict", record_predict)
    cifs, grid, times, events, train_times, train_events = cr_data
    evaluator = CompetingRisksEvaluator(
        cifs,
        grid,
        times,
        events,
        train_event_times=train_times,
        train_event_indicators=train_events,
    )
    with patch.object(module, "KaplanMeier", wraps=module.KaplanMeier) as estimator:
        evaluator.integrated_brier_score(num_points=5)
        estimator.assert_called_once()
        np.testing.assert_array_equal(estimator.call_args.args[1], train_events > 0)
    assert len(queries) == 2


def test_multi_time_interpolation_fits_each_selected_curve_once(
    cr_data: tuple,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("SurvivalEVAL.Evaluators.CompetingRisksEvaluator")
    original = module.interpolated_curve
    fits = []

    def record(*args: object) -> object:
        fits.append(1)
        return original(*args)

    monkeypatch.setattr(module, "interpolated_curve", record)
    evaluator = CompetingRisksEvaluator(*cr_data[:4])
    evaluator.brier_score_multiple_points_per_event([1, 2, 3, 4], 2, False)
    assert len(fits) == cr_data[0].shape[0]


@pytest.mark.parametrize("n_causes", [1, 8])
def test_aggregate_uses_one_float_workspace_per_cause(n_causes: int) -> None:
    n_samples, n_times = 4000, 80
    predictions = np.full((n_samples, n_causes, n_times), 0.5 / n_causes)
    times = np.linspace(0, 10, n_samples)
    events = np.arange(n_samples) % (n_causes + 1)
    targets = np.linspace(0, 10, n_times)
    tracemalloc.start()
    try:
        brier_multiple_points_cr(
            predictions, times, events, None, None, targets, ipcw=False
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 1.6 * predictions[:, 0].nbytes


@pytest.mark.parametrize("method", ["trapz", "simpson"])
def test_survival_and_competing_ibs_share_integration(method: str) -> None:
    cifs = np.array([[[0.0, 0.2, 0.8]], [[0.0, 0.4, 0.6]]])
    times, events, grid = [1, 3], [1, 1], [0, 2, 4]
    survival = SurvivalEvaluator(1 - cifs[:, 0], grid, times, events)
    competing = CompetingRisksEvaluator(cifs, grid, times, events)
    kwargs = {
        "target_times": np.array([1.0, 2.0, 3.0]),
        "integration_method": method,
    }
    assert competing.integrated_brier_score(**kwargs) == pytest.approx(
        survival.integrated_brier_score(**kwargs)
    )


@pytest.mark.parametrize("cause", [None, True, 0, -1, 1.5, 4])
@pytest.mark.parametrize(
    "metric",
    [
        "brier_score_per_event",
        "brier_score_multiple_points_per_event",
        "integrated_brier_score_per_event",
    ],
)
def test_evaluator_rejects_invalid_selected_causes(
    cr_data: tuple,
    cause: object,
    metric: str,
) -> None:
    evaluator = CompetingRisksEvaluator(*cr_data[:4])
    kwargs = (
        {"target_time": 2}
        if metric == "brier_score_per_event"
        else {"target_times": [1, 2]}
    )
    with pytest.raises(ValueError, match="event_of_interest"):
        getattr(evaluator, metric)(
            event_of_interest=cause, IPCW_weighted=False, **kwargs
        )


@pytest.mark.parametrize(
    "train_times, train_events", [([1, 2], None), (None, [1, 2]), ([1, 2], [1, 0.5])]
)
def test_invalid_training_outcomes(train_times: object, train_events: object) -> None:
    with pytest.raises(ValueError):
        brier_multiple_points_cr_per_event(
            [[0.8], [0.3]], [1, 2], [1, 2], train_times, train_events, [1]
        )


def test_generated_grid_exceeding_cif_horizon_is_rejected(cr_data: tuple) -> None:
    cifs, grid, times, events, _, _ = cr_data
    evaluator = CompetingRisksEvaluator(
        cifs,
        grid,
        times,
        events,
        train_event_times=[1, 10],
        train_event_indicators=[0, 1],
    )
    with pytest.raises(ValueError, match="prediction horizon"):
        evaluator.integrated_brier_score(num_points=5)
    assert np.isfinite(evaluator.integrated_brier_score(target_times=[1, 2]))


def test_default_ipcw_matches_explicit_weighting(cr_data: tuple) -> None:
    evaluator = CompetingRisksEvaluator(
        *cr_data[:4], train_event_times=cr_data[4], train_event_indicators=cr_data[5]
    )
    assert evaluator.brier_score(3) == evaluator.brier_score(3, True)
    assert evaluator.brier_score(3) != pytest.approx(evaluator.brier_score(3, False))
    assert evaluator.integrated_brier_score() == evaluator.integrated_brier_score(
        IPCW_weighted=True
    )
    assert evaluator.integrated_brier_score() != pytest.approx(
        evaluator.integrated_brier_score(IPCW_weighted=False)
    )
