import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.stats import kstwobign

from SurvivalEVAL import cr_d_calibration, cr_d_calibration_per_event
from SurvivalEVAL.Evaluations.DistributionCalibration import d_calibration


def _direct_curve(cifs, events, terminal, survival, cause, rho):
    """Evaluate the paper's cumulative bucket formula directly at each threshold."""
    result = []
    ratios = np.divide(cifs, terminal, out=np.zeros(cifs.shape), where=terminal > 0)
    for r in rho:
        count = np.count_nonzero((events == cause) & (ratios <= r))
        censored = events == 0
        imputed = (
            np.maximum(r * terminal[censored] - cifs[censored], 0) / survival[censored]
        )
        result.append((count + imputed.sum()) / terminal.sum())
    return np.array(result)


@pytest.mark.parametrize("num_bins", [1, 4, 10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("censoring", ["none", "all", "mixed"])
def test_bucket_accumulation_matches_paper_formula(num_bins, dtype, censoring):
    rng = np.random.default_rng(81)
    terminal = rng.uniform(0.1, 0.8, 120).astype(dtype)
    cifs = (terminal * rng.uniform(0, 1, 120)).astype(dtype)
    survival = (1 - cifs).astype(dtype)
    if censoring == "none":
        events = rng.integers(1, 3, 120)
    elif censoring == "all":
        events = np.zeros(120, dtype=int)
    else:
        events = rng.integers(0, 3, 120)
    cifs[:3] = [0, terminal[1] / 2, terminal[2]]
    survival[:3] = 1 - cifs[:3]
    for values in (cifs, terminal, survival, events):
        values.flags.writeable = False

    score, details = cr_d_calibration_per_event(
        cifs, events, terminal, survival, 1, num_bins=num_bins, return_details=True
    )
    expected = _direct_curve(cifs, events, terminal, survival, 1, details["rho"])
    tolerance = 3e-6 if dtype == np.float32 else 1e-13
    np.testing.assert_allclose(details["calibration_curve"], expected, atol=tolerance)
    assert score == pytest.approx(
        np.sqrt(trapezoid((expected - details["rho"]) ** 2, details["rho"])),
        abs=tolerance,
    )


def test_hand_computed_censoring_and_competing_events():
    score, details = cr_d_calibration_per_event(
        [0.125, 0.25, 0.125, 0.375],
        [1, 2, 0, 0],
        [0.5] * 4,
        [0.5, 0.25, 0.5, 0.25],
        1,
        num_bins=4,
        return_details=True,
    )
    # Cause 2 contributes no event count and is not imputed as a censoring.
    expected = [0, 0.5, 0.625, 0.75, 1.125]
    np.testing.assert_allclose(details["calibration_curve"], expected)
    assert score == pytest.approx(
        np.sqrt(trapezoid((np.array(expected) - details["rho"]) ** 2, details["rho"]))
    )
    # Endpoint 1.125 is retained: renormalizing it would hide marginal error.
    assert details["calibration_curve"][-1] == 1.125


def _balanced_predictions(num_bins=8, marginal=(0.5, 0.5)):
    # Dyadic thresholds keep multiplication/division exact at bucket boundaries.
    ratios = np.tile(np.linspace(1 / num_bins, 1, num_bins), 2)
    terminal = np.tile(marginal, (2 * num_bins, 1))
    cifs = terminal * ratios[:, None]
    events = np.repeat([1, 2], num_bins)
    return cifs, events, terminal, 1 - ratios


@pytest.mark.parametrize("method", ["integral", "ks", "KS"])
@pytest.mark.parametrize("reduction", ["sum", "mean", "max"])
def test_balanced_calibration_and_aggregation(method, reduction):
    args = _balanced_predictions()
    score, details = cr_d_calibration(
        *args, num_bins=8, method=method, reduction=reduction, return_details=True
    )
    assert score == pytest.approx(0, abs=1e-14)
    assert set(details["per_event_scores"]) == {1, 2}
    for item in details["per_event_details"].values():
        np.testing.assert_allclose(item["calibration_curve"], item["rho"])
    if method.lower() == "ks":
        assert details["bonferroni_approx_p_values"] == {1: 1, 2: 1}


@pytest.mark.parametrize("alpha", [1, 2, 3])
@pytest.mark.parametrize("reduction", ["sum", "mean", "max"])
def test_marginal_miscalibration_uses_paper_norm_and_all_causes(alpha, reduction):
    cifs, events, terminal, survival = _balanced_predictions(marginal=(0.25, 0.75))
    score, details = cr_d_calibration(
        cifs,
        events,
        terminal,
        survival,
        num_bins=8,
        alpha=alpha,
        reduction=reduction,
        return_details=True,
    )
    rho = np.linspace(0, 1, 9)
    norm = trapezoid(rho**alpha, rho) ** (1 / alpha)
    expected = [norm, norm / 3]
    np.testing.assert_allclose(list(details["per_event_scores"].values()), expected)
    assert score == pytest.approx(
        {"sum": np.sum, "mean": np.mean, "max": np.max}[reduction](expected)
    )
    assert details["per_event_details"][1]["calibration_curve"][-1] == 2


def test_ks_statistic_approximation_and_bonferroni_adjustment():
    cifs, events, terminal, survival = _balanced_predictions(marginal=(0.25, 0.75))
    score, details = cr_d_calibration(
        cifs, events, terminal, survival, num_bins=8, method="ks", return_details=True
    )
    np.testing.assert_allclose(list(details["per_event_scores"].values()), [1, 1 / 3])
    assert score == pytest.approx(4 / 3)
    for cause, item in details["per_event_details"].items():
        p = kstwobign.sf(np.sqrt(8) * details["per_event_scores"][cause])
        assert item["event_count"] == 8
        assert item["approx_p_value"] == pytest.approx(p)
        assert details["bonferroni_approx_p_values"][cause] == pytest.approx(
            min(1, 2 * p)
        )


def test_absent_observed_cause_is_scored_but_has_no_ks_pvalue():
    cifs, events, terminal, survival = _balanced_predictions()
    events[:] = 1
    score, details = cr_d_calibration(
        cifs, events, terminal, survival, method="ks", return_details=True
    )
    assert score > 0
    assert details["per_event_scores"][2] == 1
    assert np.isnan(details["per_event_details"][2]["approx_p_value"])
    assert np.isnan(details["bonferroni_approx_p_values"][2])


def test_one_cause_reduces_to_existing_d_calibration_curve():
    survival = np.array([0.13, 0.37, 0.62, 0.81, 1.0])
    events = np.array([1, 0, 1, 0, 0])
    _, details = cr_d_calibration_per_event(
        1 - survival, events, np.ones(5), survival, 1, num_bins=10, return_details=True
    )
    _, _, histogram = d_calibration(survival, events, num_bins=10)
    np.testing.assert_allclose(
        details["calibration_curve"][1:], np.cumsum(histogram) / 5
    )


def test_probability_boundaries_and_zero_mass_for_other_subjects():
    _, details = cr_d_calibration_per_event(
        [0, 1, 0],
        [1, 1, 2],
        [1, 1, 0],
        [1, 0, 0],
        1,
        num_bins=2,
        return_details=True,
    )
    np.testing.assert_allclose(details["calibration_curve"], [0.5, 0.5, 1])


def test_tiny_censored_survival_does_not_overflow_unused_tail_weights():
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        score, details = cr_d_calibration_per_event(
            [1], [0], [1], [1e-320], 1, method="ks", return_details=True
        )
    assert score == 1
    np.testing.assert_array_equal(details["calibration_curve"], 0)
    assert np.isnan(details["approx_p_value"])


def test_large_deviations_do_not_overflow_integral_powers():
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        score = cr_d_calibration_per_event([0.5e-200], [1], [1e-200], [0.5], 1)
    assert np.isfinite(score)
    assert score > 1e199


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_bins": 0},
        {"num_bins": 2.5},
        {"num_bins": True},
        {"alpha": 0},
        {"alpha": 0.5},
        {"alpha": np.inf},
        {"alpha": np.nan},
        {"method": "chi-square"},
        {"event_of_interest": 0},
        {"event_of_interest": True},
        {"event_of_interest": 1.5},
    ],
)
def test_invalid_options(kwargs):
    options = {"event_of_interest": 1, **kwargs}
    with pytest.raises(ValueError):
        cr_d_calibration_per_event([0.2], [1], [0.6], [0.7], **options)


@pytest.mark.parametrize(
    "cifs, events, terminal, survival",
    [
        ([0.2, 0.3], [1], [0.6, 0.6], [0.7, 0.7]),
        ([[0.2]], [[1]], [[0.6]], [[0.7]]),
        ([0.2], [0.5], [0.6], [0.7]),
        ([0.2], [-1], [0.6], [0.7]),
        ([0.2], [np.inf], [0.6], [0.7]),
        ([np.inf], [1], [0.6], [0.7]),
        ([0.2], [1], [np.nan], [0.7]),
        ([0.2], [1], [0.6], [np.inf]),
        ([-0.2], [1], [0.6], [0.7]),
        ([0.2], [1], [1.2], [0.7]),
        ([0.7], [1], [0.6], [0.2]),
        ([0.5], [1], [0.6], [0.6]),
        ([0.1], [1], [0.8], [0.2]),
        ([0.2], [0], [0.2], [0]),
        ([0], [1], [0], [1]),
        ([0], [2], [0], [1]),
    ],
)
def test_invalid_predictions_and_labels(cifs, events, terminal, survival):
    with pytest.raises(ValueError):
        cr_d_calibration_per_event(cifs, events, terminal, survival, 1)


@pytest.mark.parametrize(
    "change", ["labels", "survival", "terminal", "reduction", "rank"]
)
def test_invalid_joint_inputs(change):
    cifs, events, terminal, survival = _balanced_predictions()
    kwargs = {}
    if change == "labels":
        events[0] = 3
    elif change == "survival":
        survival[0] = 0.5
    elif change == "terminal":
        terminal *= 0.9
    elif change == "reduction":
        kwargs["reduction"] = "median"
    else:
        cifs, terminal = cifs[:, 0], terminal[:, 0]
    with pytest.raises(ValueError):
        cr_d_calibration(cifs, events, terminal, survival, **kwargs)
