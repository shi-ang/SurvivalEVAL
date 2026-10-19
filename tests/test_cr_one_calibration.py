import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from SurvivalEVAL import CompetingRisksEvaluator, one_calibration_cr
from SurvivalEVAL.evaluations.calibration.single_time import one_calibration


@pytest.fixture
def grouped_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.repeat([0.9, 0.5, 0.1], 4),
        np.array([1, 1, 2, 4, 0, 2, 3, 4, 1, 2, 4, 5], dtype=float),
        np.array([2, 0, 1, 1, 1, 2, 0, 2, 0, 0, 0, 0]),
    )


@pytest.mark.parametrize("strategy", ["C", "c", "H", "h"])
@pytest.mark.parametrize("cause", [1, 2])
def test_grouped_aj_incidence_with_competing_events_and_ties(
    grouped_data: tuple,
    strategy: str,
    cause: int,
) -> None:
    _, _, observed, expected = one_calibration_cr(
        *grouped_data, 2, event_of_interest=cause, num_bins=3, binning_strategy=strategy
    )
    # First group: S(1) = 3/4, then cause 1 has one event among two at risk.
    actual = [0.375, 0.25, 0] if cause == 1 else [0.25, 0.25, 0]
    predicted = [0.9, 0.5, 0.1]
    if strategy.lower() == "h":
        actual, predicted = actual[::-1], predicted[::-1]
    np.testing.assert_allclose(observed, actual)
    np.testing.assert_allclose(expected, predicted)


@pytest.mark.parametrize("target, expected", [(0, 0.25), (2, 0.25), (4, 0.25)])
def test_events_at_target_and_zero_are_included(target: float, expected: float) -> None:
    _, _, observed, _ = one_calibration_cr(
        np.full(4, 0.25), [0, 2, 3, 4], [1, 2, 0, 2], target, num_bins=1
    )
    assert observed == [expected]


@pytest.mark.parametrize("num_bins", [1, 3, 20])
def test_fully_observed_groups_match_empirical_incidence(num_bins: int) -> None:
    probabilities = np.linspace(0, 1, 8)
    times = np.array([1, 3, 2, 4, 1, 3, 2, 4])
    events = np.array([1, 2, 2, 1, 2, 1, 1, 2])
    _, _, obs, exp = one_calibration_cr(
        probabilities, times, events, 2, num_bins=num_bins
    )
    groups = [
        group for group in np.array_split(np.arange(7, -1, -1), num_bins) if group.size
    ]
    np.testing.assert_allclose(
        obs, [np.mean((times[g] <= 2) & (events[g] == 1)) for g in groups]
    )
    np.testing.assert_allclose(exp, [probabilities[g].mean() for g in groups])


@pytest.mark.parametrize("strategy", ["C", "H"])
def test_one_cause_matches_single_event_km_curve(strategy: str) -> None:
    predictions = np.linspace(0.1, 0.9, 12)
    times = np.tile([1, 2, 3, 4], 3)
    events = np.tile([1, 0, 1, 0], 3)
    _, _, obs, exp = one_calibration_cr(
        predictions, times, events, 2.5, num_bins=3, binning_strategy=strategy
    )
    _, _, single_obs, single_exp = one_calibration(
        predictions, times, events, 2.5, num_bins=3, binning_strategy=strategy
    )
    np.testing.assert_allclose(obs, single_obs)
    np.testing.assert_allclose(exp, single_exp)


@pytest.mark.parametrize(
    "events, cause", [([0, 0, 0], 1), ([2, 3, 2], 1), ([1, 2, 3], 4)]
)
def test_absent_cause_and_all_censored_groups(events: list, cause: int) -> None:
    _, _, obs, exp = one_calibration_cr(
        [0, 0.5, 1], [1, 2, 3], events, 2, cause, num_bins=8, binning_strategy="H"
    )
    np.testing.assert_array_equal(obs, [0, 0, 0])
    np.testing.assert_array_equal(exp, [0, 0.5, 1])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_readonly_noncontiguous_inputs_are_unchanged(
    grouped_data: tuple,
    dtype: type,
) -> None:
    arrays = []
    for original in grouped_data:
        data = np.repeat(original.astype(dtype), 2)[::2]
        data.flags.writeable = False
        arrays.append(data)
    _, _, obs, exp = one_calibration_cr(*arrays, 2, num_bins=3)
    np.testing.assert_allclose(obs, [0.375, 0.25, 0])
    np.testing.assert_allclose(exp, [0.9, 0.5, 0.1])
    for actual, original in zip(arrays, grouped_data):
        np.testing.assert_array_equal(actual, original.astype(dtype))


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("cause", [1, 2])
def test_evaluator_matches_standalone_and_uses_default_time(
    grouped_data: tuple,
    interpolation: str,
    cause: int,
) -> None:
    probabilities, times, events = grouped_data
    cifs = np.stack([probabilities * 0.6, probabilities * 0.3], axis=1)
    curves = np.stack([np.zeros_like(cifs), cifs, cifs + 0.02], axis=2)
    evaluator = CompetingRisksEvaluator(
        curves,
        [0, 2, 5],
        times,
        events,
        interpolation=interpolation,
        train_event_times=[1, 2, 4],
        train_event_indicators=[0, 1, 2],
    )
    target = evaluator.default_target_time
    predicted = evaluator.predict_cifs(np.array([target]), cause)[:, 0]
    expected = one_calibration_cr(predicted, times, events, target, cause, num_bins=3)
    actual = evaluator.one_calibration(event_of_interest=cause, num_bins=3)
    np.testing.assert_allclose(actual[0], expected[0], equal_nan=True)
    np.testing.assert_allclose(actual[1:], expected[2:])


@pytest.mark.parametrize("strategy", ["C", "H"])
@pytest.mark.parametrize("constant", [False, True])
def test_curve_details_plot_expected_on_x_axis(strategy: str, constant: bool) -> None:
    probabilities = np.full(12, 0.3) if constant else np.linspace(0, 0.8, 12)
    curves = np.stack([np.zeros(12), probabilities], axis=1)[:, None, :]
    evaluator = CompetingRisksEvaluator(
        curves, [0, 4], np.linspace(1, 4, 12), np.ones(12)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p_value, details = evaluator.one_calibration(
            4, num_bins=3, binning_strategy=strategy, return_details=True
        )
    try:
        _, obs, exp = evaluator.one_calibration(
            4, num_bins=3, binning_strategy=strategy
        )
        np.testing.assert_allclose(details["observed_probabilities"], obs)
        np.testing.assert_allclose(details["expected_probabilities"], exp)
        np.testing.assert_allclose(details["p_value"], p_value, equal_nan=True)
        assert "statistics" in details and "variances" in details
        order = np.argsort(exp, kind="stable")
        line = details["pp_plot"][1].lines[1]
        np.testing.assert_allclose(line.get_xdata(), np.asarray(exp)[order])
        np.testing.assert_allclose(line.get_ydata(), np.asarray(obs)[order])
        bars = details["histogram_plot"][1].patches
        np.testing.assert_allclose(
            [bar.get_height() for bar in bars[: len(obs)]], np.asarray(obs)[order]
        )
        if constant:
            assert np.isnan(details["max_local_deviation"])
    finally:
        plt.close(details["histogram_plot"][0])
        plt.close(details["pp_plot"][0])


@pytest.mark.parametrize("cause", [None, True, np.bool_(True), 0, -1, 1.5])
def test_invalid_cause(cause: object) -> None:
    with pytest.raises(ValueError, match="event_of_interest"):
        one_calibration_cr([0.2, 0.5], [1, 2], [1, 2], 2, event_of_interest=cause)


@pytest.mark.parametrize("bins", [True, 0, -1, 2.5])
def test_invalid_bin_count(bins: object) -> None:
    with pytest.raises(ValueError, match="num_bins"):
        one_calibration_cr([0.2, 0.5], [1, 2], [1, 2], 2, num_bins=bins)


@pytest.mark.parametrize(
    "preds, times, events, target",
    [
        ([0.1], [1, 2], [1, 2], 2),
        ([[0.1], [0.2]], [1, 2], [1, 2], 2),
        ([-0.1, 0.5], [1, 2], [1, 2], 2),
        ([0.1, 1.1], [1, 2], [1, 2], 2),
        ([np.inf, 0.5], [1, 2], [1, 2], 2),
        ([np.nan, 0.5], [1, 2], [1, 2], 2),
        ([0.1, 0.5], [-1, 2], [1, 2], 2),
        ([0.1, 0.5], [1, 2], [1, -1], 2),
        ([0.1, 0.5], [1, 2], [1, 1.5], 2),
        ([0.1, 0.5], [1, 2], [1, 2], -1),
        ([0.1, 0.5], [1, 2], [1, 2], np.nan),
        ([0.1, 0.5], [1, 2], [1, 2], np.inf),
    ],
)
def test_invalid_inputs(
    preds: object,
    times: object,
    events: object,
    target: float,
) -> None:
    with pytest.raises(ValueError):
        one_calibration_cr(preds, times, events, target)


def test_invalid_strategy() -> None:
    with pytest.raises(TypeError, match="binning_strategy"):
        one_calibration_cr([0.2, 0.5], [1, 2], [1, 2], 2, binning_strategy="invalid")


@pytest.mark.parametrize("cause", [None, 0, True, 2])
def test_evaluator_rejects_invalid_cause(cause: object) -> None:
    evaluator = CompetingRisksEvaluator(
        [[[0, 0.2]], [[0, 0.5]]], [0, 2], [1, 2], [1, 1]
    )
    with pytest.raises(ValueError, match="event_of_interest"):
        evaluator.one_calibration(2, event_of_interest=cause)


@pytest.mark.parametrize("target", [-1, np.nan, np.inf, 3])
def test_evaluator_rejects_invalid_horizon(target: float) -> None:
    evaluator = CompetingRisksEvaluator(
        [[[0, 0.2]], [[0, 0.5]]], [0, 2], [1, 2], [1, 1]
    )
    with pytest.raises(ValueError, match="target_time"):
        evaluator.one_calibration(target)
