import numpy as np
import pytest

from SurvivalEVAL.Evaluations.Discrimination import (
    auc,
    concordance,
    concordance_time_dependent,
)


def _evaluate(metric, **changes):
    kwargs = dict(event_times=[1, 2, 3], event_indicators=[1, 0, 1])
    if metric == "auc":
        kwargs.update(predict_probs=[0.1, 0.7, 0.8], target_time=2)
        function = auc
    elif metric == "concordance":
        kwargs.update(predicted_times=[1, 3, 2])
        function = concordance
    else:
        kwargs.update(risk_scores=[[0.9, 0.9], [0.1, 0.1], [0.3, 0.3]])
        function = concordance_time_dependent
    kwargs.update(changes)
    return function(**kwargs)


@pytest.mark.parametrize("metric", ["auc", "concordance", "time_dependent"])
def test_discrimination_functions_accept_list_inputs(metric):
    result = _evaluate(metric)
    assert (result if metric == "auc" else result[0]) == 1


@pytest.mark.parametrize("metric", ["auc", "concordance", "time_dependent"])
@pytest.mark.parametrize(
    "changes",
    [
        {"event_indicators": [1, 2, 1]},
        {"event_indicators": [1, -1, 1]},
        {"event_indicators": [1, 0.5, 1]},
        {"event_indicators": [1, np.nan, 1]},
        {"event_times": [-1, 2, 3]},
        {"event_times": [1, np.inf, 3]},
        {"event_times": [1, np.nan, 3]},
        {"event_times": [1, 2]},
        {"event_times": [[1, 2, 3]], "event_indicators": [[1, 0, 1]]},
    ],
)
def test_discrimination_rejects_invalid_observed_outcomes(metric, changes):
    with pytest.raises(ValueError):
        _evaluate(metric, **changes)


@pytest.mark.parametrize("metric", ["auc", "concordance", "time_dependent"])
@pytest.mark.parametrize("time", [-1, np.nan, np.inf])
def test_discrimination_rejects_invalid_evaluation_times(metric, time):
    parameter = "target_time" if metric == "auc" else "tau"
    with pytest.raises(ValueError):
        _evaluate(metric, **{parameter: time})


@pytest.mark.parametrize("metric", ["auc", "concordance", "time_dependent"])
@pytest.mark.parametrize(
    "changes",
    [
        {"train_event_indicators": [0, 2, 1]},
        {"train_event_indicators": [[0, 1, 1]]},
        {"train_event_times": [1, -1, 3]},
        {"train_event_times": [1, np.inf, 3]},
        {"train_event_times": [1, 2]},
    ],
)
def test_ipcw_validates_training_outcomes(metric, changes):
    kwargs = dict(
        train_event_times=[1, 2, 3], train_event_indicators=[0, 1, 1], method="IPCW"
    )
    kwargs.update(changes)
    with pytest.raises(ValueError):
        _evaluate(metric, **kwargs)


@pytest.mark.parametrize(
    "changes",
    [
        {"predicted_times": [[1, 3, 2]]},
        {"predicted_times": [1, 2]},
        {"predicted_times": [1, np.nan, 2]},
    ],
)
def test_concordance_validates_predictions(changes):
    with pytest.raises(ValueError):
        _evaluate("concordance", **changes)


@pytest.mark.parametrize(
    "risks", [[0.9, 0.1, 0.3], [[0.9], [0.1], [0.3]], [[0.9, 0.9], [0.1, 0.1]]]
)
def test_time_dependent_concordance_validates_risk_matrix_shape(risks):
    with pytest.raises(ValueError):
        _evaluate("time_dependent", risk_scores=risks)


@pytest.mark.parametrize(
    "probs",
    [
        [0.1, 0.7],
        [[0.1, 0.7, 0.8]],
        [-0.1, 0.7, 0.8],
        [1.1, 0.7, 0.8],
        [np.inf, 0.7, 0.8],
    ],
)
def test_auc_validates_probabilities(probs):
    with pytest.raises(ValueError):
        _evaluate("auc", predict_probs=probs)


def test_infinite_predicted_time_remains_a_valid_survival_prediction():
    result = concordance([1, np.inf, 3], [1, 2, 3], [1, 0, 1])
    np.testing.assert_allclose(result, (1, 2, 2))


def test_readonly_inputs_are_not_modified_by_shared_ipcw_setup():
    times = np.array([1, 2, 3], dtype=np.float32)
    events = np.array([1, 0, 1])
    train_times = np.array([1, 2, 4], dtype=float)
    train_events = np.array([1, 0, 1])
    before = [value.copy() for value in (times, events, train_times, train_events)]
    for value in (times, events, train_times, train_events):
        value.flags.writeable = False
    common = dict(
        event_times=times,
        event_indicators=events,
        train_event_times=train_times,
        train_event_indicators=train_events,
        method="IPCW",
        tau=3,
    )
    ordinary = concordance([1, 3, 2], **common)
    dynamic = concordance_time_dependent([[0.9, 0.9], [0.1, 0.1], [0.3, 0.3]], **common)
    np.testing.assert_allclose(ordinary, dynamic)
    for actual, expected in zip((times, events, train_times, train_events), before):
        np.testing.assert_array_equal(actual, expected)
