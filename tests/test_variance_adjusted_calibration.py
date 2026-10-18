import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import chi2

from SurvivalEVAL import (
    CompetingRisksEvaluator,
    SingleTimeEvaluator,
    SurvivalEvaluator,
    one_calibration,
    one_calibration_cr,
)


def _weighted_cif(
    times: np.ndarray,
    events: np.ndarray,
    weights: np.ndarray,
    target: float,
    cause: int,
) -> float:
    """Independent weighted AJ product for numerical influence derivatives."""
    survival, incidence = 1.0, 0.0
    for time in np.unique(times[(events > 0) & (times <= target)]):
        risk = weights[times >= time].sum()
        failures = (times == time) & (events > 0)
        selected = (times == time) & (events == cause)
        incidence += survival * weights[selected].sum() / risk
        survival *= 1 - weights[failures].sum() / risk
    return incidence


@pytest.mark.parametrize("competing", [False, True])
@pytest.mark.parametrize("target", [0, 3, 7, 20])
@pytest.mark.parametrize("seed", [12, 29])
def test_variances_match_independent_influence_derivatives(
    competing: bool, target: float, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    times = rng.integers(0, 12, 40).astype(float)
    events = rng.integers(0, 3 if competing else 2, times.size)
    weights = np.ones(times.size)
    epsilon = 1e-4
    influence = np.empty(times.size)
    for i in range(times.size):
        weights[i] += epsilon
        upper = _weighted_cif(times, events, weights, target, 1)
        weights[i] -= 2 * epsilon
        lower = _weighted_cif(times, events, weights, target, 1)
        weights[i] = 1
        influence[i] = (upper - lower) / (2 * epsilon)
    metric = one_calibration_cr if competing else one_calibration
    _, details = metric(
        np.full(times.size, 0.3), times, events, target, num_bins=1, return_details=True
    )
    assert details["variances"][0] == pytest.approx(
        influence @ influence, rel=1e-7, abs=1e-12
    )
    assert details["observed_probabilities"][0] == pytest.approx(
        _weighted_cif(times, events, weights, target, 1)
    )


@pytest.mark.parametrize("target", [2, 10])
def test_greenwood_variance_and_step_tail(target: float) -> None:
    _, details = one_calibration(
        np.repeat([0.6, 0.5], 4),
        np.tile([1, 1, 2, 4], 2),
        np.tile([1, 0, 1, 0], 2),
        target,
        num_bins=2,
        return_details=True,
    )
    # S = (1 - 1/4)(1 - 1/2); Greenwood = S^2(1/(4*3) + 1/(2*1)).
    expected_variance = 0.375**2 * (1 / 12 + 1 / 2)
    np.testing.assert_allclose(details["variances"], expected_variance)
    np.testing.assert_allclose(details["observed_probabilities"], 0.625)
    statistic = ((0.625 - 0.6) ** 2 + (0.625 - 0.5) ** 2) / expected_variance
    assert details["statistics"] == pytest.approx(statistic)
    assert details["degrees_of_freedom"] == 2
    assert details["p_value"] == pytest.approx(chi2.sf(statistic, 2))


def test_aj_variance_matches_r_survival_with_ties() -> None:
    # R survival 3.8.12: survfit(Surv(time, factor(cause, levels=0:2)) ~ group).
    p, details = one_calibration_cr(
        np.repeat([0.6, 0.5], 4),
        [1, 1, 2, 4, 0, 2, 3, 4],
        [2, 0, 1, 1, 1, 2, 0, 2],
        2,
        num_bins=2,
        return_details=True,
    )
    np.testing.assert_allclose(details["observed_probabilities"], [0.375, 0.25])
    np.testing.assert_allclose(details["variances"], [0.08203125, 0.046875])
    statistic = (0.375 - 0.6) ** 2 / 0.08203125 + (0.25 - 0.5) ** 2 / 0.046875
    assert details["statistics"] == pytest.approx(statistic)
    assert p == pytest.approx(chi2.sf(statistic, 2))


@pytest.mark.parametrize(
    "times, events, variance",
    [([1, 1, 2, 2], [1, 2, 1, 2], 0.0625), ([1, 2, 2], [0, 1, 2], 0.125)],
)
def test_terminal_all_event_ties_keep_cause_split_variance(
    times: list, events: list, variance: float
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, details = one_calibration_cr(
            np.full(len(times), 0.5), times, events, 2, num_bins=1, return_details=True
        )
    assert details["variances"] == pytest.approx([variance])
    assert details["p_value"] == 1


@pytest.mark.parametrize("competing", [False, True])
@pytest.mark.parametrize("prediction", [0.0, 0.2, 1.0])
@pytest.mark.parametrize("event_value", [0, 1])
def test_zero_variance_is_undefined_without_dropping_groups(
    competing: bool, prediction: float, event_value: int
) -> None:
    metric = one_calibration_cr if competing else one_calibration
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        p, details = metric(
            np.full(8, prediction),
            np.arange(1, 9),
            np.full(8, event_value),
            8,
            num_bins=2,
            return_details=True,
        )
    assert np.isnan(p)
    assert details["variances"] == [0, 0]
    assert details["observed_probabilities"] == [event_value, event_value]
    assert details["bin_sizes"] == [4, 4]
    assert details["statistics"] == (0 if prediction == event_value else np.inf)
    assert details["degrees_of_freedom"] == 2


@pytest.mark.parametrize("strategy", ["C", "H"])
@pytest.mark.parametrize("adjust", [False, True])
def test_one_event_reduction_matches_all_outputs(strategy: str, adjust: bool) -> None:
    probabilities = np.repeat([0.6, 0.2], 4)
    times, events = np.tile([1, 1, 2, 4], 2), np.tile([1, 0, 1, 0], 2)
    single = one_calibration(
        probabilities,
        times,
        events,
        2,
        num_bins=2,
        binning_strategy=strategy,
        adjust_variance=adjust,
    )
    competing = one_calibration_cr(
        probabilities,
        times,
        events,
        2,
        num_bins=2,
        binning_strategy=strategy,
        adjust_variance=adjust,
    )
    np.testing.assert_allclose(single[:2], competing[:2])
    np.testing.assert_allclose(single[2:], competing[2:])


@pytest.mark.parametrize("adjust", [False, True])
def test_uncensored_uses_filtered_group_size_and_includes_target_events(
    adjust: bool,
) -> None:
    _, details = one_calibration(
        np.full(5, 0.4),
        [0.5, 1, 2, 2, 3],
        [0, 1, 1, 0, 1],
        2,
        num_bins=1,
        method="Uncensored",
        adjust_variance=adjust,
        return_details=True,
    )
    assert details["bin_sizes"] == [4]
    assert details["observed_probabilities"] == [0.5]
    assert details["variances"] == pytest.approx([0.25 / 4 if adjust else 0.24 / 4])


@pytest.mark.parametrize("competing", [False, True])
def test_unadjusted_statistic_is_original_dn_formula(competing: bool) -> None:
    metric = one_calibration_cr if competing else one_calibration
    _, details = metric(
        np.repeat([0.6, 0.2], 4),
        np.tile([1, 1, 2, 4], 2),
        np.tile([1, 0, 1, 0], 2),
        2,
        num_bins=2,
        adjust_variance=False,
        return_details=True,
    )
    expected_variances = np.array([0.6 * 0.4 / 4, 0.2 * 0.8 / 4])
    statistic = np.sum((0.625 - np.array([0.6, 0.2])) ** 2 / expected_variances)
    np.testing.assert_allclose(details["variances"], expected_variances)
    assert details["statistics"] == pytest.approx(statistic)
    assert details["p_value"] == pytest.approx(chi2.sf(statistic, 1))
    assert details["adjust_variance"] is False


@pytest.mark.parametrize("kind", ["survival", "single-time", "competing"])
@pytest.mark.parametrize("adjust", [False, True])
def test_evaluator_returns_test_metadata_and_curve_details(
    kind: str, adjust: bool
) -> None:
    probabilities = np.repeat([0.6, 0.2], 4)
    times, events = np.tile([1, 1, 2, 4], 2), np.tile([1, 0, 1, 0], 2)
    cifs = np.stack([np.zeros(8), probabilities, probabilities + 0.1], axis=1)
    grid = [0, 2, 4]
    if kind == "survival":
        evaluator = SurvivalEvaluator(1 - cifs, grid, times, events)
    elif kind == "single-time":
        evaluator = SingleTimeEvaluator(1 - probabilities, times, events, target_time=2)
    else:
        evaluator = CompetingRisksEvaluator(cifs[:, None, :], grid, times, events)
    options = {} if kind == "single-time" else {"target_time": 2}
    p, details = evaluator.one_calibration(
        num_bins=2, adjust_variance=adjust, return_details=True, **options
    )
    try:
        assert details["adjust_variance"] == adjust
        assert details["p_value"] == p
        assert details["bin_sizes"] == [4, 4]
        assert details["degrees_of_freedom"] == (2 if adjust else 1)
        simple = evaluator.one_calibration(
            num_bins=2, adjust_variance=adjust, **options
        )
        assert simple[0] == p
        np.testing.assert_allclose(
            simple[1:],
            [details["observed_probabilities"], details["expected_probabilities"]],
        )
    finally:
        plt.close(details["histogram_plot"][0])
        plt.close(details["pp_plot"][0])


def test_uncensored_evaluator_shortcut_keeps_variance_adjustment() -> None:
    probabilities = np.repeat([0.6, 0.2], 4)
    times, events = np.tile([1, 2, 3, 4], 2), np.ones(8)
    evaluator = SingleTimeEvaluator(1 - probabilities, times, events, target_time=2)
    p, obs, exp = evaluator.one_calibration(num_bins=2)
    assert obs == [0.5, 0.5]
    statistic = np.sum((0.5 - np.array([0.6, 0.2])) ** 2 / (0.25 / 4))
    assert p == pytest.approx(chi2.sf(statistic, 2))


def test_competing_label_pooling_preserves_variance() -> None:
    times = np.array([1, 1, 2, 4, 2, 3, 4, 4])
    events = np.array([1, 2, 3, 4, 0, 4, 2, 1])
    probabilities = np.full(8, 0.2)
    original = one_calibration_cr(
        probabilities,
        times,
        events,
        3,
        event_of_interest=4,
        num_bins=1,
        return_details=True,
    )
    pooled = np.where(events == 0, 0, np.where(events == 4, 1, 2))
    collapsed = one_calibration_cr(
        probabilities, times, pooled, 3, num_bins=1, return_details=True
    )
    assert original == collapsed


def test_empty_groups_do_not_inflate_degrees_of_freedom() -> None:
    _, details = one_calibration(
        np.repeat([0.2, 0.8], 4),
        np.tile([1, 1, 2, 4], 2),
        np.tile([1, 0, 1, 0], 2),
        2,
        num_bins=20,
        binning_strategy="H",
        return_details=True,
    )
    assert details["degrees_of_freedom"] == 2
    assert len(details["variances"]) == 2


@pytest.mark.parametrize("competing", [False, True])
def test_calibrated_model_simulation_with_independent_censoring(
    competing: bool,
) -> None:
    rng = np.random.default_rng(771)
    n_groups, per_group, repetitions = 6, 1000, 160
    rates = np.repeat(np.linspace(0.25, 0.75, n_groups), per_group)
    fraction = 0.7 if competing else 1.0
    predictions = fraction * (1 - np.exp(-rates * 2))
    metric = one_calibration_cr if competing else one_calibration
    adjusted_rejections, unadjusted_rejections = 0, 0
    for _ in range(repetitions):
        failure = rng.exponential(1 / rates)
        censor = rng.exponential(1 / 0.5, rates.size)
        labels = (
            rng.choice([1, 2], rates.size, p=[0.7, 0.3])
            if competing
            else np.ones(rates.size)
        )
        events = np.where(failure <= censor, labels, 0)
        times = np.minimum(failure, censor)
        adjusted, _, _, _ = metric(predictions, times, events, 2, num_bins=n_groups)
        plain, _, _, _ = metric(
            predictions, times, events, 2, num_bins=n_groups, adjust_variance=False
        )
        assert np.isfinite(adjusted)
        adjusted_rejections += adjusted < 0.05
        unadjusted_rejections += plain < 0.05
    # A wide Monte Carlo tolerance: these are approximate chi-square tests.
    assert 0.01 < adjusted_rejections / repetitions < 0.12
    assert adjusted_rejections < unadjusted_rejections
