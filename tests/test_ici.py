"""ICI regression checks against independent R Cox and Fine-Gray calculations."""

import importlib
import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from SurvivalEVAL import (
    CompetingRisksEvaluator,
    SingleTimeEvaluator,
    SurvivalEvaluator,
    integrated_calibration_index,
    integrated_calibration_index_cr,
)
from SurvivalEVAL.evaluations.calibration.single_time.integrated_calibration_index import (
    _fit_calibration_curve,
)
from SurvivalEVAL.evaluations.calibration.single_time.util import (
    _cloglog,
    _restricted_cubic_spline,
)

REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures" / "ici_reference.json").read_text()
)
METRICS = ["ICI", "E50", "E90", "E_max"]
FUNCTIONS = [integrated_calibration_index, integrated_calibration_index_cr]
CASES = [
    pytest.param(
        dataset,
        fit,
        id=f"cr={dataset['competing']}-ties={dataset['tied']}-knots={fit['knots']}",
    )
    for dataset in REFERENCE["datasets"]
    for fit in dataset["fits"]
]


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = REFERENCE["datasets"][-1]
    return tuple(
        np.asarray(dataset[key]) for key in ("preds", "event_time", "event_indicator")
    )


def assert_summary_equal(actual: dict, expected: dict) -> None:
    """
    Compare ICI metrics and plotted curves with numerical fitting tolerance.

    Parameters
    ----------
    actual, expected : dict
        Summaries with ICI metrics and calibration curve arrays.

    Returns
    -------
    None
        Raise an assertion error if the summaries differ.
    """
    np.testing.assert_allclose(
        [actual[k] for k in METRICS],
        [expected[k] for k in METRICS],
        atol=1e-7,
        rtol=1e-7,
    )
    for key in ("grid", "cal_pred"):
        np.testing.assert_allclose(
            actual["curve"][key], expected["curve"][key], atol=1e-7, rtol=1e-7
        )


@pytest.mark.parametrize("dataset, reference", CASES)
def test_ici_matches_r_subject_predictions_curves_and_metrics(
    dataset: dict, reference: dict
) -> None:
    preds, times, events = (
        np.asarray(dataset[key]) for key in ("preds", "event_time", "event_indicator")
    )
    function = (
        integrated_calibration_index_cr
        if dataset["competing"]
        else integrated_calibration_index
    )
    summary = function(
        preds, times, events, dataset["target_time"], knots=reference["knots"]
    )
    np.testing.assert_allclose(
        [summary[k] for k in METRICS], reference["metrics"], atol=1e-7, rtol=1e-7
    )
    np.testing.assert_allclose(
        summary["curve"]["cal_pred"], reference["curve"], atol=1e-7, rtol=1e-7
    )
    transformed = _cloglog(preds)
    evaluation_x = np.concatenate((transformed, _cloglog(summary["curve"]["grid"])))
    spline, evaluation = _restricted_cubic_spline(
        transformed, evaluation_x, reference["knots"]
    )
    pooled = np.minimum(events, 2)
    calibrated = _fit_calibration_curve(
        spline, evaluation, times, pooled, dataset["target_time"]
    )
    np.testing.assert_allclose(
        calibrated[: preds.size], reference["calibrated"], atol=1e-7, rtol=1e-7
    )


def test_spline_accepts_general_covariates_and_independent_evaluation_points() -> None:
    x = np.linspace(-4, 6, 101)
    evaluation_x = np.array([-8, -2.5, 0, 3, 9])
    spline, evaluation = _restricted_cubic_spline(x, evaluation_x, knots=3)
    assert spline.shape == (x.size, 2)
    assert evaluation.shape == (evaluation_x.size, 2)
    # A restricted cubic spline represents a linear covariate exactly,
    # including evaluations beyond its boundary knots.
    coefficients = np.linalg.lstsq(spline, x - x.mean(), rcond=None)[0]
    np.testing.assert_allclose(
        evaluation @ coefficients + x.mean(), evaluation_x, atol=1e-12
    )


@pytest.mark.parametrize("knots", [3, 4, 5])
def test_one_cause_is_identical_to_survival_ici(data: tuple, knots: int) -> None:
    preds, times, events = data
    events = (events > 0).astype(int)
    expected = integrated_calibration_index(preds, times, events, 3, knots=knots)
    actual = integrated_calibration_index_cr(
        preds, times, events * 7, 3, event_of_interest=7, knots=knots
    )
    assert_summary_equal(actual, expected)


def test_competing_cause_labels_pooling_and_subject_order_do_not_change_fit(
    data: tuple,
) -> None:
    preds, times, events = data
    expected = integrated_calibration_index_cr(preds, times, events, 3)
    relabeled = np.choose(events, [0, 7, 8, 11])
    order = np.random.default_rng(93).permutation(preds.size)
    actual = integrated_calibration_index_cr(
        preds[order], times[order], relabeled[order], 3, event_of_interest=7
    )
    pooled = integrated_calibration_index_cr(preds, times, np.minimum(events, 2), 3)
    assert_summary_equal(actual, expected)
    assert_summary_equal(pooled, expected)


def test_uncensored_competing_events_stay_at_risk_with_unit_weight(data: tuple) -> None:
    preds, times, events = data
    keep = events > 0
    preds, times, events = preds[keep], times[keep], events[keep]
    expected = integrated_calibration_index_cr(preds, times, events, 3)
    extended = np.where(events == 1, times, times.max() + 1)
    actual = integrated_calibration_index(preds, extended, events == 1, 3)
    assert_summary_equal(actual, expected)


@pytest.mark.parametrize("probability", [0, 0.4, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_constant_competing_risks_use_exact_aj_probability(
    probability: float, dtype: type
) -> None:
    predictions = np.full(6, probability, dtype=dtype)
    summary = integrated_calibration_index_cr(
        predictions, [0, 1, 1, 2, 2, 3], [2, 3, 0, 3, 5, 0], 2, event_of_interest=3
    )
    np.testing.assert_allclose(
        [summary[k] for k in METRICS], abs(float(predictions[0]) - 7 / 18)
    )
    np.testing.assert_allclose(summary["curve"]["cal_pred"], 7 / 18)


@pytest.mark.parametrize(
    "events, target, cause",
    [([0, 0, 0], 3, 1), ([2, 3, 2], 3, 1), ([1, 2, 3], 3, 4), ([1, 2, 3], 0, 1)],
)
def test_no_selected_events_by_horizon_give_zero_curve(
    events: list, target: float, cause: int
) -> None:
    preds = np.array([0, 0.3, 1])
    summary = integrated_calibration_index_cr(
        preds, [1, 2, 3], events, target, event_of_interest=cause
    )
    np.testing.assert_array_equal(summary["curve"]["cal_pred"], 0)
    np.testing.assert_allclose(
        [summary[k] for k in METRICS],
        [preds.mean(), np.median(preds), np.quantile(preds, 0.9), 1],
    )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("constant", [False, True])
def test_calibrated_hazards_hold_constant_after_followup(
    data: tuple, function: Callable, constant: bool
) -> None:
    preds, times, events = data
    if function is integrated_calibration_index:
        events = events > 0
    if constant:
        preds = np.full(preds.size, 0.4)
    expected = function(preds, times, events, times.max() + 1)
    actual = function(preds, times, events, times.max() + 10)
    assert_summary_equal(actual, expected)


def test_nonconstant_curve_includes_events_at_time_zero_and_target(data: tuple) -> None:
    preds, times, events = data
    at_zero = integrated_calibration_index_cr(preds, times, events, 0)
    assert np.any(at_zero["curve"]["cal_pred"] > 0)
    before = integrated_calibration_index_cr(preds, times, events, np.nextafter(3, 0))
    at_target = integrated_calibration_index_cr(preds, times, events, 3)
    assert np.all(at_target["curve"]["cal_pred"] > before["curve"]["cal_pred"])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_boundary_probabilities_and_readonly_strided_inputs(
    data: tuple, dtype: type
) -> None:
    preds, times, events = data
    preds = preds.copy()
    preds[:2] = [0, 1]
    arrays = [
        np.repeat(array.astype(dtype), 2)[::2] for array in (preds, times, events)
    ]
    for array in arrays:
        array.flags.writeable = False
    originals = [array.copy() for array in arrays]
    summary = integrated_calibration_index_cr(*arrays, 3)
    assert all(0 <= summary[k] <= 1 for k in METRICS)
    assert np.all(np.isfinite(summary["curve"]["cal_pred"]))
    for actual, original in zip(arrays, originals):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("levels", [2, 3])
def test_unsupported_nonconstant_risks_are_rejected(
    function: Callable, levels: int
) -> None:
    preds = np.repeat(np.linspace(0, 1, levels), 20)
    # Two levels cannot support three knots; for three equally frequent
    # levels, request four knots to verify that complexity is not reduced.
    with pytest.raises(ValueError, match="distinct spline knots"):
        function(
            preds,
            np.arange(preds.size),
            np.ones(preds.size),
            3,
            knots=max(3, levels + 1),
        )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_repeated_quantile_knots_are_rejected(function: Callable) -> None:
    preds = np.r_[np.zeros(100), np.linspace(0.1, 1, 10)]
    with pytest.raises(ValueError, match="distinct spline knots"):
        function(preds, np.arange(preds.size), np.ones(preds.size), 3)


@pytest.mark.parametrize("knots", [False, 0, 2, 3.0, None])
@pytest.mark.parametrize("function", FUNCTIONS)
def test_invalid_knot_counts_are_rejected(function: Callable, knots: object) -> None:
    with pytest.raises(ValueError, match="knots must"):
        function([0.4, 0.4], [1, 2], [1, 0], 1, knots=knots)


@pytest.mark.parametrize("cause", [False, 0, -1, 1.0, None])
def test_invalid_selected_cause_is_rejected(cause: object) -> None:
    with pytest.raises(ValueError, match="event_of_interest"):
        integrated_calibration_index_cr(
            [0.4, 0.4], [1, 2], [1, 0], 1, event_of_interest=cause
        )


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize(
    "changes",
    [
        {"preds": [0.1]},
        {"preds": [[0.1, 0.2]]},
        {"preds": [-0.1, 0.2]},
        {"preds": [np.nan, 0.2]},
        {"event_time": [-1, 2]},
        {"event_time": [1]},
        {"event_indicator": [0, 1.5]},
        {"event_indicator": [-1, 1]},
        {"target_time": -1},
        {"target_time": np.nan},
        {"target_time": np.inf},
    ],
)
def test_invalid_inputs_are_rejected(function: Callable, changes: dict) -> None:
    arguments = {
        "preds": [0.4, 0.4],
        "event_time": [1, 2],
        "event_indicator": [1, 0],
        "target_time": 1,
    }
    with pytest.raises((ValueError, TypeError)):
        function(**(arguments | changes))


def test_nonidentifiable_risk_sets_are_rejected() -> None:
    # The only target event occurs after all other subjects were censored,
    # so its risk set contains no covariate information.
    with pytest.raises(ValueError, match="not identifiable"):
        integrated_calibration_index(
            np.linspace(0.1, 0.9, 20), np.arange(20), np.r_[np.zeros(19), 1], 19
        )


def test_optimizer_failure_is_reported(
    data: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(
        "SurvivalEVAL.evaluations.calibration.single_time.integrated_calibration_index"
    )
    monkeypatch.setattr(
        module,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, message="iteration limit"
        ),
    )
    with pytest.raises(ValueError, match="failed to converge: iteration limit"):
        integrated_calibration_index_cr(*data, 3)


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("cause", [1, 2, 3])
def test_evaluator_matches_standalone_at_default_and_explicit_times(
    data: tuple, interpolation: str, cause: int
) -> None:
    preds, times, events = data
    terminal = np.stack((preds * 0.5, preds * 0.3, preds * 0.1), axis=1)
    end = times.max() + 1
    curves = np.stack((np.zeros_like(terminal), terminal * 0.4, terminal), axis=2)
    evaluator = CompetingRisksEvaluator(
        curves, [0, 3, end], times, events, interpolation=interpolation
    )
    for target in (None, 2.5):
        horizon = evaluator.default_target_time if target is None else target
        predicted = evaluator.predict_cifs(np.array([horizon]), cause)[:, 0]
        expected = integrated_calibration_index_cr(
            predicted, times, events, horizon, cause
        )
        actual = evaluator.integrated_calibration_index(
            target, cause, draw_figure=False
        )
        assert_summary_equal(actual, expected)


def test_single_event_evaluators_use_corrected_shared_method(data: tuple) -> None:
    preds, times, events = data
    events = events > 0
    expected = integrated_calibration_index(preds, times, events, 3)
    single = SingleTimeEvaluator(1 - preds, times, events, target_time=3)
    curves = np.column_stack((np.ones(preds.size), 1 - preds, (1 - preds) * 0.5))
    survival = SurvivalEvaluator(curves, [0, 3, times.max() + 1], times, events)
    assert_summary_equal(
        single.integrated_calibration_index(draw_figure=False), expected
    )
    assert_summary_equal(
        survival.integrated_calibration_index(3, draw_figure=False), expected
    )


@pytest.mark.parametrize("cause", [0, 4, None, False, 1.0])
def test_evaluator_rejects_invalid_cause(cause: object) -> None:
    evaluator = CompetingRisksEvaluator(
        np.zeros((3, 3, 2)), [0, 4], [1, 2, 3], [0, 0, 0]
    )
    with pytest.raises(ValueError, match="event_of_interest"):
        evaluator.integrated_calibration_index(2, cause, draw_figure=False)


def test_evaluator_plot_defaults_and_figure_range(data: tuple) -> None:
    preds, times, events = data
    terminal = np.stack((preds * 0.5, preds * 0.3, preds * 0.1), axis=1)
    curves = np.stack((np.zeros_like(terminal), terminal), axis=2)
    evaluator = CompetingRisksEvaluator(curves, [0, times.max() + 1], times, events)
    summary, (fig, ax) = evaluator.integrated_calibration_index(
        3, figure_range=(0, 1, 0, 1)
    )
    try:
        np.testing.assert_allclose(ax.lines[0].get_xdata(), summary["curve"]["grid"])
        np.testing.assert_allclose(
            ax.lines[0].get_ydata(), summary["curve"]["cal_pred"]
        )
        assert ax.get_xlim() == (0, 1)
        assert ax.get_ylim() == (0, 1)
    finally:
        plt.close(fig)
