import numpy as np
import pytest

from SurvivalEVAL import SingleTimeEvaluator, SurvivalEvaluator
from SurvivalEVAL.Evaluations.Discrimination import auc


@pytest.fixture
def auc_data() -> dict[str, np.ndarray]:
    return {
        "predict_probs": np.array([0.8, 0.3, 0.1, 0.6, 0.2, 0.01]),
        "event_times": np.array([1.0, 2.0, 3.0, 5.0, 6.0, 1.5]),
        "event_indicators": np.array([1, 1, 1, 0, 1, 0], dtype=bool),
        "train_event_times": np.array([1.0, 2.0, 2.0, 4.0, 6.0]),
        "train_event_indicators": np.array([1, 1, 0, 1, 1], dtype=bool),
    }


def _pairwise_ipcw_auc(
    predict_probs: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray,
    train_event_indicators: np.ndarray,
    target_time: float,
) -> float:
    """Enumerate weighted pairs using censoring risk sets, without KM or sklearn."""
    cases = np.flatnonzero(event_indicators & (event_times <= target_time))
    controls = np.flatnonzero(
        (event_times > target_time) | (~event_indicators & (event_times == target_time))
    )
    numerator = denominator = 0.0
    for case in cases:
        survival = 1.0
        censoring_times = np.unique(
            train_event_times[
                ~train_event_indicators & (train_event_times < event_times[case])
            ]
        )
        for time in censoring_times:
            censorings = np.count_nonzero(
                (train_event_times == time) & ~train_event_indicators
            )
            # Original events at this time leave before tied censorings.
            risk_set = np.count_nonzero(train_event_times > time) + censorings
            survival *= 1 - censorings / risk_set
        weight = 1 / survival
        for control in controls:
            numerator += weight * (
                (predict_probs[case] < predict_probs[control])
                + 0.5 * (predict_probs[case] == predict_probs[control])
            )
            denominator += weight
    return numerator / denominator


@pytest.mark.parametrize("method", ["Naive", "nAiVe", "IPCW", "iPcW"])
def test_auc_hand_computed_mixed_training_ties(
    auc_data: dict[str, np.ndarray], method: str
) -> None:
    # G(1-)=G(2-)=1; the tied event/censoring at 2 gives G(3-)=2/3.
    # Case weights are [1, 1, 3/2], with respectively [0, 1, 2] wins.
    expected = 4 / 7 if method.lower() == "ipcw" else 0.5
    assert auc(**auc_data, target_time=3, method=method) == pytest.approx(expected)
    assert auc(**auc_data, target_time=3) == pytest.approx(0.5)


@pytest.mark.parametrize("method", ["Naive", "IPCW"])
def test_auc_target_boundary_includes_event_cases_and_censored_controls(
    method: str,
) -> None:
    assert auc(
        [0.2, 0.8, 0.5, 0.0],
        [1, 2, 2, 1.5],
        [1, 1, 0, 0],
        target_time=2,
        train_event_times=[1, 2, 2, 4, 6],
        train_event_indicators=[1, 1, 0, 1, 1],
        method=method,
    ) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "train_times, train_events, test_times, target",
    [
        ([0, 0, 2], [1, 0, 1], [0, 1, 2], 1),
        ([2, 2, 4], [1, 0, 1], [0, 3, 5], 3),
    ],
)
def test_ipcw_auc_uses_baseline_before_first_censoring_update(
    train_times: list[float],
    train_events: list[int],
    test_times: list[float],
    target: float,
) -> None:
    # The event at zero gets weight 1, the later event gets weight 2.
    assert auc(
        [0.8, 0.2, 0.5],
        test_times,
        [1, 1, 0],
        target,
        train_event_times=train_times,
        train_event_indicators=train_events,
        method="IPCW",
    ) == pytest.approx(2 / 3)


def test_ipcw_auc_accepts_target_at_terminal_censoring_jump() -> None:
    # G(2)=0 but G(2-)=1: the event at 2 and censoring at 2 are both known.
    assert auc(
        [0.2, 0.8, 0.5],
        [1, 2, 2],
        [1, 1, 0],
        2,
        train_event_times=[1, 2, 2],
        train_event_indicators=[1, 1, 0],
        method="IPCW",
    ) == pytest.approx(0.5)


def test_ipcw_auc_without_training_censoring_equals_naive(
    auc_data: dict[str, np.ndarray],
) -> None:
    auc_data["train_event_indicators"] = np.ones(5, dtype=bool)
    assert auc(**auc_data, target_time=3, method="IPCW") == pytest.approx(
        auc(**auc_data, target_time=3)
    )


def test_ipcw_auc_recovers_complete_data_auc_with_independent_discrete_censoring() -> (
    None
):
    true_times = np.repeat([1.0, 2.0, 3.0, 5.0, 6.0], 3)
    censor_times = np.tile([2.0, 3.0, 10.0], 5)
    probs = np.repeat([0.8, 0.3, 0.1, 0.6, 0.2], 3)
    observed = np.minimum(true_times, censor_times)
    indicators = true_times <= censor_times
    complete = auc(probs, true_times, np.ones(15, dtype=bool), 3)
    corrected = auc(probs, observed, indicators, 3, observed, indicators, method="IPCW")
    assert corrected == pytest.approx(complete)
    assert corrected == pytest.approx(0.5)


@pytest.mark.parametrize("seed", range(10))
def test_ipcw_auc_matches_pairwise_oracle_with_time_and_prediction_ties(
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    train_times = rng.integers(0, 10, 40).astype(float)
    train_times[-1] = 9
    train_events = rng.random(40) < 0.6
    times = rng.integers(0, 13, 40).astype(float)
    events = rng.random(40) < 0.7
    times[:2] = [0, 12]
    events[:2] = [True, False]
    probs = rng.integers(0, 5, 40) / 4
    for target in [2.0, 5.0, 8.0]:
        expected = _pairwise_ipcw_auc(
            probs, times, events, train_times, train_events, target
        )
        assert auc(
            probs, times, events, target, train_times, train_events, "IPCW"
        ) == pytest.approx(expected)


@pytest.mark.parametrize("seed", range(30))
def test_ipcw_auc_matches_scikit_survival_without_exact_time_ties(seed: int) -> None:
    metrics = pytest.importorskip("sksurv.metrics")
    util = pytest.importorskip("sksurv.util")
    rng = np.random.default_rng(seed)
    train_times = rng.uniform(0.1, 40, 200)
    train_events = rng.random(200) < 0.65
    train_events[np.argmax(train_times)] = True
    times = rng.uniform(0.2, 20, 100)
    events = rng.random(100) < 0.7
    probs = rng.integers(1, 10, 100) / 10
    train = util.Surv.from_arrays(train_events, train_times)
    test = util.Surv.from_arrays(events, times)
    for target in [3.0, 7.0, 11.0]:
        # Left and right limits agree without exact time ties. Use exact risk ties.
        expected, _ = metrics.cumulative_dynamic_auc(
            train, test, 1 - probs, [target], tied_tol=0
        )
        assert auc(
            probs, times, events, target, train_times, train_events, "IPCW"
        ) == pytest.approx(expected[0])


@pytest.mark.parametrize("missing", ["both", "times", "indicators"])
def test_ipcw_auc_requires_training_outcomes(
    auc_data: dict[str, np.ndarray], missing: str
) -> None:
    if missing in {"both", "times"}:
        auc_data.pop("train_event_times")
    if missing in {"both", "indicators"}:
        auc_data.pop("train_event_indicators")
    with pytest.raises(ValueError, match="required for IPCW AUROC"):
        auc(**auc_data, target_time=3, method="IPCW")


def test_auc_rejects_unknown_method(auc_data: dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="method must be"):
        auc(**auc_data, target_time=3, method="unknown")


@pytest.mark.parametrize("method", ["Naive", "IPCW"])
@pytest.mark.parametrize("target", [0, 6])
def test_auc_rejects_absent_cases_or_controls(method: str, target: float) -> None:
    with pytest.raises(ValueError, match="all zeros or all ones"):
        auc([0.8, 0.2], [1, 4], [1, 1], target, [1, 6], [1, 1], method)


def test_ipcw_auc_rejects_target_beyond_training_follow_up() -> None:
    with pytest.raises(ValueError, match="maximum training follow-up"):
        auc([0.8, 0.2], [1, 5], [1, 0], 3, [1, 2], [1, 0], "IPCW")


@pytest.mark.parametrize("method", ["Naive", "IPCW"])
def test_auc_default_target_remains_test_median(
    auc_data: dict[str, np.ndarray], method: str
) -> None:
    target = float(np.median(auc_data["event_times"]))
    assert auc(**auc_data, method=method) == pytest.approx(
        auc(**auc_data, target_time=target, method=method)
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ipcw_auc_does_not_modify_readonly_inputs(
    auc_data: dict[str, np.ndarray], dtype: type
) -> None:
    for name, values in auc_data.items():
        auc_data[name] = values.astype(bool if "indicators" in name else dtype)
        auc_data[name].flags.writeable = False
    before = {name: values.copy() for name, values in auc_data.items()}
    assert auc(**auc_data, target_time=3, method="IPCW") == pytest.approx(4 / 7)
    for name, values in auc_data.items():
        np.testing.assert_array_equal(values, before[name])


def _make_evaluator(
    data: dict[str, np.ndarray], kind: str, training: bool = True
) -> SurvivalEvaluator | SingleTimeEvaluator:
    outcomes = {
        name: values for name, values in data.items() if name != "predict_probs"
    }
    if not training:
        outcomes.pop("train_event_times")
        outcomes.pop("train_event_indicators")
    if kind == "single-time":
        return SingleTimeEvaluator(data["predict_probs"], target_time=3, **outcomes)
    curves = np.column_stack(
        [np.ones(6), data["predict_probs"], data["predict_probs"] / 2]
    )
    return SurvivalEvaluator(curves, np.array([0.0, 3.0, 6.0]), **outcomes)


@pytest.mark.parametrize("kind", ["survival", "single-time"])
@pytest.mark.parametrize("uncensored", [False, True])
def test_evaluators_forward_ipcw_even_with_fully_observed_test_outcomes(
    auc_data: dict[str, np.ndarray], kind: str, uncensored: bool
) -> None:
    if uncensored:
        auc_data["event_times"][-1] = 7
        auc_data["event_indicators"] = np.ones(6, dtype=bool)
    evaluator = _make_evaluator(auc_data, kind)
    kwargs = {"target_time": 3} if kind == "survival" else {}
    expected = auc(**auc_data, target_time=3, method="IPCW")
    assert evaluator.auc(method="iPcW", **kwargs) == pytest.approx(expected)
    assert evaluator.auc(**kwargs) == pytest.approx(auc(**auc_data, target_time=3))
    assert expected != pytest.approx(evaluator.auc(**kwargs))
    if kind == "survival":
        assert evaluator.auroc(method="IPCW", **kwargs) == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["survival", "single-time"])
@pytest.mark.parametrize("uncensored", [False, True])
def test_evaluators_require_ipcw_training_even_without_test_censoring(
    auc_data: dict[str, np.ndarray], kind: str, uncensored: bool
) -> None:
    if uncensored:
        auc_data["event_indicators"] = np.ones(6, dtype=bool)
    evaluator = _make_evaluator(auc_data, kind, training=False)
    with pytest.raises(TypeError, match="Train set information is missing"):
        evaluator.auc(method="IPCW")


@pytest.mark.parametrize("kind", ["survival", "single-time"])
def test_evaluator_default_target_remains_combined_median(
    auc_data: dict[str, np.ndarray], kind: str
) -> None:
    target = float(
        np.median(np.r_[auc_data["event_times"], auc_data["train_event_times"]])
    )
    if kind == "single-time":
        outcomes = {
            name: values for name, values in auc_data.items() if name != "predict_probs"
        }
        evaluator = SingleTimeEvaluator(auc_data["predict_probs"], **outcomes)
        assert evaluator.target_time == target
        probabilities = auc_data["predict_probs"]
    else:
        evaluator = _make_evaluator(auc_data, kind)
        probabilities = evaluator.predict_probability_from_curve(target)
    for method in ["Naive", "IPCW"]:
        data = dict(auc_data, predict_probs=probabilities)
        assert evaluator.auc(method=method) == pytest.approx(
            auc(**data, target_time=target, method=method)
        )
