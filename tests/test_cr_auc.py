import importlib
from unittest.mock import patch

import numpy as np
import pytest

from SurvivalEVAL import CompetingRisksEvaluator, auc, auc_cr, auc_cr_per_event


@pytest.fixture
def cr_auc_data() -> dict[str, np.ndarray]:
    return {
        "predict_probs": np.array([0.8, 0.9, 0.4, 0.3, 0.6, 0.99]),
        "event_times": np.array([1, 2, 2.5, 4, 5, 1.5]),
        "event_indicators": np.array([1, 2, 1, 0, 2, 0]),
        "train_event_times": np.array([1, 2, 2, 4, 6]),
        "train_event_indicators": np.array([1, 1, 0, 2, 1]),
    }


def _pairwise_auc(
    probabilities: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    train_times: np.ndarray,
    train_events: np.ndarray,
    target: float,
    cause: int,
    controls: str,
    method: str,
) -> float:
    """Enumerate pairs and derive censoring weights directly from risk sets."""
    cases = [
        i for i, (t, e) in enumerate(zip(times, events)) if t <= target and e == cause
    ]
    comparators = [
        i
        for i, (t, e) in enumerate(zip(times, events))
        if t > target
        or (t == target and e == 0)
        or (controls == "NotCase" and t <= target and e not in (0, cause))
    ]
    if not cases or not comparators:
        return float("nan")

    def weight(i: int) -> float:
        if method == "Naive":
            return 1.0
        survival = 1.0
        for knot in np.unique(train_times[train_times < min(times[i], target)]):
            censorings = np.count_nonzero((train_times == knot) & (train_events == 0))
            if censorings:
                # Observed events of every cause leave before tied censorings.
                at_risk = np.count_nonzero(train_times > knot) + censorings
                survival *= 1 - censorings / at_risk
        return 1 / survival

    numerator = denominator = 0.0
    for i in cases:
        for j in comparators:
            pair_weight = weight(i) * weight(j)
            numerator += pair_weight * (
                (probabilities[i] > probabilities[j])
                + 0.5 * (probabilities[i] == probabilities[j])
            )
            denominator += pair_weight
    return numerator / denominator


@pytest.mark.parametrize(
    "control_definition, method, expected",
    [
        ("EventFree", "Naive", 3 / 4),
        ("NotCase", "Naive", 1 / 2),
        ("eVeNtFrEe", "iPcW", 7 / 10),
        ("nOtCaSe", "IPCW", 21 / 40),
    ],
)
def test_cr_auc_hand_computed_weights(
    cr_auc_data: dict[str, np.ndarray],
    control_definition: str,
    method: str,
    expected: float,
) -> None:
    # At t=3, case weights are [1, 3/2], event-free control weights [3/2, 3/2],
    # and the competing event at t=2 has weight 1, before the tied censoring jump.
    assert auc_cr_per_event(
        **cr_auc_data,
        target_time=3,
        method=method,
        control_definition=control_definition,
    ) == pytest.approx(expected)
    assert auc_cr_per_event(**cr_auc_data, target_time=3) == pytest.approx(1 / 2)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("controls", ["EventFree", "NotCase"])
@pytest.mark.parametrize("method", ["Naive", "IPCW"])
def test_cr_auc_matches_pairwise_oracle(seed: int, controls: str, method: str) -> None:
    rng = np.random.default_rng(seed)
    times = rng.integers(0, 13, 40).astype(float)
    events = rng.integers(0, 4, 40)
    times[:4], events[:4] = [0, 0, 0, 12], [1, 2, 3, 0]
    probabilities = rng.integers(0, 5, (40, 3)) / 12
    train_times = rng.integers(0, 10, 40).astype(float)
    train_events = rng.integers(0, 4, 40)
    train_times[-1], train_events[-1] = 9, 1
    for target in (2, 5, 8):
        expected = {
            cause: _pairwise_auc(
                probabilities[:, cause - 1],
                times,
                events,
                train_times,
                train_events,
                target,
                cause,
                controls,
                method,
            )
            for cause in (1, 2, 3)
        }
        options = dict(
            target_time=target,
            train_event_times=train_times,
            train_event_indicators=train_events,
            control_definition=controls,
            method=method,
        )
        mean, details = auc_cr(
            probabilities, times, events, return_details=True, **options
        )
        assert mean == pytest.approx(np.mean(list(expected.values())))
        assert details["per_event_scores"] == pytest.approx(expected)
        for cause in (1, 2, 3):
            assert auc_cr_per_event(
                probabilities[:, cause - 1],
                times,
                events,
                event_of_interest=cause,
                **options,
            ) == pytest.approx(expected[cause])


@pytest.mark.parametrize("controls", ["EventFree", "NotCase"])
@pytest.mark.parametrize("method", ["Naive", "IPCW"])
def test_cr_auc_reduces_to_single_event(
    cr_auc_data: dict[str, np.ndarray], controls: str, method: str
) -> None:
    data = dict(cr_auc_data)
    data["event_indicators"] = data["event_indicators"] > 0
    data["train_event_indicators"] = data["train_event_indicators"] > 0
    expected = auc(
        **dict(data, predict_probs=1 - data["predict_probs"]),
        target_time=3,
        method=method,
    )
    assert auc_cr_per_event(
        **data, target_time=3, method=method, control_definition=controls
    ) == pytest.approx(expected)
    assert auc_cr(
        **dict(data, predict_probs=data["predict_probs"][:, None]),
        target_time=3,
        method=method,
        control_definition=controls,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("controls", ["EventFree", "NotCase"])
def test_cr_auc_without_training_censoring_equals_naive(
    cr_auc_data: dict[str, np.ndarray], controls: str
) -> None:
    cr_auc_data["train_event_indicators"] = np.array([1, 2, 1, 3, 2])
    assert auc_cr_per_event(
        **cr_auc_data, target_time=3, method="IPCW", control_definition=controls
    ) == pytest.approx(
        auc_cr_per_event(**cr_auc_data, target_time=3, control_definition=controls)
    )


@pytest.mark.parametrize("controls", ["EventFree", "NotCase"])
@pytest.mark.parametrize("method", ["Naive", "IPCW"])
def test_cr_auc_target_boundary(controls: str, method: str) -> None:
    # Cases at 2, a competing event at 2, a censoring at 2, and unknown censoring at 1.
    result = auc_cr_per_event(
        [0.8, 0.4, 0.7, 0.5, 1],
        [1, 2, 2, 2, 1],
        [1, 1, 2, 0, 0],
        2,
        [2, 2],
        [2, 0],
        method,
        control_definition=controls,
    )
    assert result == pytest.approx(0.5)


def test_cr_auc_does_not_require_unused_horizon_survival() -> None:
    # G(3-)=0, but all cases/controls have known events before censoring at 2.
    assert (
        auc_cr_per_event(
            [0.8, 0.2, 0.99],
            [1, 1, 2],
            [1, 2, 0],
            3,
            [1, 2],
            [1, 0],
            "IPCW",
        )
        == 1
    )
    # An event-free control needs G(3-) and must trigger an error.
    with pytest.raises(ValueError, match="Censoring survival"):
        auc_cr_per_event([0.8, 0.2], [1, 4], [1, 0], 3, [1, 2], [1, 0], "IPCW")


@pytest.mark.parametrize("controls", ["EventFree", "NotCase"])
def test_cr_auc_recovers_complete_data_with_independent_censoring(
    controls: str,
) -> None:
    true_times = np.repeat([1.0, 2.0, 3.0, 5.0, 6.0], 3)
    true_causes = np.repeat([1, 2, 1, 2, 1], 3)
    probabilities = np.repeat([0.8, 0.9, 0.4, 0.3, 0.6], 3)
    censor_times = np.tile([2.0, 3.0, 10.0], 5)
    times = np.minimum(true_times, censor_times)
    causes = np.where(true_times <= censor_times, true_causes, 0)
    expected = auc_cr_per_event(
        probabilities, true_times, true_causes, 3, control_definition=controls
    )
    assert auc_cr_per_event(
        probabilities,
        times,
        causes,
        3,
        times,
        causes,
        "IPCW",
        control_definition=controls,
    ) == pytest.approx(expected)


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("value", [-0.1, 1.1, np.inf, np.nan])
def test_cr_auc_rejects_invalid_probabilities(aggregate: bool, value: float) -> None:
    function = auc_cr if aggregate else auc_cr_per_event
    probabilities = [[value], [0.2]] if aggregate else [value, 0.2]
    with pytest.raises(ValueError):
        function(probabilities, [1, 3], [1, 0], 2)


@pytest.mark.parametrize(
    "function, probabilities",
    [
        (auc_cr_per_event, [[0.8, 0.2]]),
        (auc_cr_per_event, [0.8]),
        (auc_cr, [0.8, 0.2]),
        (auc_cr, [[0.8]]),
        (auc_cr, [[[0.8]], [[0.2]]]),
    ],
)
def test_cr_auc_rejects_prediction_shape(function, probabilities: list) -> None:
    with pytest.raises(ValueError, match="predict_probs"):
        function(probabilities, [1, 3], [1, 0], 2)


def test_cr_auc_rejects_labels_beyond_predicted_causes() -> None:
    with pytest.raises(ValueError, match="number of predicted causes"):
        auc_cr([[0.8], [0.2]], [1, 3], [1, 2], 2)


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize(
    "changes",
    [
        {"train_event_times": [-1, 4]},
        {"train_event_times": [1, np.inf]},
        {"train_event_times": [1]},
        {"train_event_indicators": [1, -1]},
        {"train_event_indicators": [1, 1.5]},
    ],
)
def test_cr_auc_rejects_invalid_training(aggregate: bool, changes: dict) -> None:
    function = auc_cr if aggregate else auc_cr_per_event
    data = dict(
        predict_probs=[[0.8], [0.2]] if aggregate else [0.8, 0.2],
        event_times=[1, 3],
        event_indicators=[1, 0],
        target_time=2,
        method="IPCW",
        train_event_times=[1, 4],
        train_event_indicators=[2, 0],
    )
    data.update(changes)
    with pytest.raises(ValueError):
        function(**data)


@pytest.mark.parametrize("controls", ["EventFree", "NotCase"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cr_auc_default_target_and_readonly_inputs(
    cr_auc_data: dict[str, np.ndarray], controls: str, dtype: type
) -> None:
    for name, values in cr_auc_data.items():
        cr_auc_data[name] = values.astype(np.int64 if "indicators" in name else dtype)
        cr_auc_data[name].flags.writeable = False
    before = {name: values.copy() for name, values in cr_auc_data.items()}
    target = float(np.median(cr_auc_data["event_times"]))
    assert auc_cr_per_event(
        **cr_auc_data, method="IPCW", control_definition=controls
    ) == pytest.approx(
        auc_cr_per_event(
            **cr_auc_data,
            target_time=target,
            method="IPCW",
            control_definition=controls,
        )
    )
    for name, values in cr_auc_data.items():
        np.testing.assert_array_equal(values, before[name])


def test_cr_auc_fits_one_censoring_model(cr_auc_data: dict[str, np.ndarray]) -> None:
    module = importlib.import_module(
        "SurvivalEVAL.Evaluations.Discrimination.SingleTime.AreaUnderROCurve"
    )
    data = dict(
        cr_auc_data,
        predict_probs=np.column_stack(
            [
                cr_auc_data["predict_probs"],
                1 - cr_auc_data["predict_probs"],
            ]
        ),
    )
    with patch.object(module, "KaplanMeier", wraps=module.KaplanMeier) as fit:
        auc_cr(**data, target_time=3, method="IPCW")
    fit.assert_called_once()


@pytest.mark.parametrize("controls", ["EventFree", "NotCase"])
@pytest.mark.parametrize("method", ["Naive", "IPCW"])
def test_cr_auc_retains_undefined_causes(controls: str, method: str) -> None:
    options = dict(
        target_time=2,
        train_event_times=[1, 2, 3],
        train_event_indicators=[1, 0, 2],
        method=method,
        control_definition=controls,
    )
    score, details = auc_cr(
        [[0.8, 0.1], [0.2, 0.3]], [1, 3], [1, 0], return_details=True, **options
    )
    assert np.isnan(score)
    assert details["per_event_scores"][1] == 1
    assert np.isnan(details["per_event_scores"][2])
    with pytest.raises(ValueError, match="Cases or controls are absent"):
        auc_cr_per_event([0.1, 0.3], [1, 3], [1, 0], event_of_interest=2, **options)
    score, details = auc_cr(
        [[0.8, 0.1], [0.2, 0.3]], [1, 1], [0, 0], return_details=True, **options
    )
    assert np.isnan(score)
    assert all(np.isnan(value) for value in details["per_event_scores"].values())


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize(
    "changes, match",
    [
        ({"event_times": [1]}, "same shape"),
        ({"event_times": [-1, 3]}, "non-negative"),
        ({"event_times": [1, np.inf]}, "finite"),
        ({"event_indicators": [1, -1]}, "integer labels"),
        ({"event_indicators": [1, 0.5]}, "integer labels"),
        ({"event_indicators": [1, np.inf]}, "integer labels"),
        ({"target_time": -1}, "non-negative"),
        ({"target_time": np.nan}, "finite"),
        ({"target_time": np.inf}, "finite"),
        ({"method": "unknown"}, "method must be"),
        ({"method": "Uno"}, "method must be"),
        ({"control_definition": "unknown"}, "control_definition"),
    ],
)
def test_cr_auc_rejects_invalid_inputs(
    aggregate: bool, changes: dict, match: str
) -> None:
    function = auc_cr if aggregate else auc_cr_per_event
    data = dict(
        predict_probs=[[0.8, 0.1], [0.2, 0.3]] if aggregate else [0.8, 0.2],
        event_times=[1, 3],
        event_indicators=[1, 0],
        target_time=2,
    )
    data.update(changes)
    with pytest.raises(ValueError, match=match):
        function(**data)


@pytest.mark.parametrize("aggregate", [False, True])
@pytest.mark.parametrize("missing", ["both", "times", "indicators"])
def test_cr_auc_requires_training_even_for_uncensored_test(
    aggregate: bool, missing: str
) -> None:
    function = auc_cr if aggregate else auc_cr_per_event
    options = {"train_event_times": [1, 3], "train_event_indicators": [2, 0]}
    if missing in {"both", "times"}:
        options.pop("train_event_times")
    if missing in {"both", "indicators"}:
        options.pop("train_event_indicators")
    with pytest.raises(ValueError, match="required for IPCW"):
        function(
            [[0.8], [0.2]] if aggregate else [0.8, 0.2],
            [1, 3],
            [1, 1],
            2,
            method="IPCW",
            **options,
        )


@pytest.mark.parametrize("cause", [0, -1, 1.5, True])
def test_cr_auc_rejects_invalid_cause(cause: int) -> None:
    with pytest.raises(ValueError, match="event_of_interest"):
        auc_cr_per_event([0.8, 0.2], [1, 3], [1, 0], 2, event_of_interest=cause)


@pytest.mark.parametrize("method", ["Naive", "IPCW"])
@pytest.mark.parametrize("controls", ["EventFree", "NotCase"])
@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("explicit_target", [False, True])
def test_cr_evaluator_auc_matches_standalone(
    cr_auc_data: dict[str, np.ndarray],
    method: str,
    controls: str,
    interpolation: str,
    explicit_target: bool,
) -> None:
    probabilities = np.column_stack(
        [cr_auc_data["predict_probs"], 1 - cr_auc_data["predict_probs"]]
    )
    cifs = probabilities[:, :, None] * np.array([0, 0.4, 1])[None, None, :]
    outcomes = {
        key: value for key, value in cr_auc_data.items() if key != "predict_probs"
    }
    evaluator = CompetingRisksEvaluator(
        cifs, [0, 3, 6], interpolation=interpolation, **outcomes
    )
    target = (
        3
        if explicit_target
        else float(
            np.median(np.r_[outcomes["event_times"], outcomes["train_event_times"]])
        )
    )
    predicted = evaluator._predict_risks("CIF", None, target)
    options = dict(method=method, control_definition=controls)
    call = dict(options, target_time=3) if explicit_target else options
    expected = auc_cr(
        predicted, target_time=target, return_details=True, **outcomes, **options
    )
    np.testing.assert_equal(evaluator.auc(return_details=True, **call), expected)
    np.testing.assert_equal(evaluator.auroc(return_details=True, **call), expected)
    for cause in (1, 2):
        expected = auc_cr_per_event(
            predicted[:, cause - 1],
            target_time=target,
            event_of_interest=cause,
            **outcomes,
            **options,
        )
        module = importlib.import_module("SurvivalEVAL.CompetingRisksEvaluator")
        with patch.object(
            module, "interpolated_curve", wraps=module.interpolated_curve
        ) as interpolate:
            assert evaluator.auc_per_event(cause, **call) == pytest.approx(expected)
        assert interpolate.call_count == len(outcomes["event_times"])
        np.testing.assert_array_equal(
            [invocation.args[1] for invocation in interpolate.call_args_list],
            evaluator.pred_cifs[:, cause - 1],
        )
        assert evaluator.auroc_per_event(cause, **call) == pytest.approx(expected)


@pytest.mark.parametrize("target", [-1, np.nan, np.inf, 7])
def test_cr_evaluator_auc_rejects_invalid_horizon(target: float) -> None:
    evaluator = CompetingRisksEvaluator(
        [[[0, 0.8]], [[0, 0.2]]], [0, 6], [1, 3], [1, 0]
    )
    with pytest.raises(ValueError, match="target_time"):
        evaluator.auc(target)
    with pytest.raises(ValueError, match="target_time"):
        evaluator.auc_per_event(target_time=target)


def test_cr_evaluator_auc_default_without_training_and_missing_training() -> None:
    evaluator = CompetingRisksEvaluator(
        [[[0, 0.8]], [[0, 0.2]]], [0, 6], [1, 3], [1, 1]
    )
    assert evaluator.auc() == evaluator.auc(target_time=2) == 1
    assert evaluator.auc_per_event() == 1
    with pytest.raises(ValueError, match="required for IPCW"):
        evaluator.auc(method="IPCW")
    with pytest.raises(ValueError, match="required for IPCW"):
        evaluator.auc_per_event(method="IPCW")
    with pytest.raises(ValueError, match="event_of_interest"):
        evaluator.auc_per_event(2)
