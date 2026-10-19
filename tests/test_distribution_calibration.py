import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import CensoredData, chisquare, ecdf, kstest

from SurvivalEVAL.evaluations.calibration.distributional import (
    create_censor_hist,
    d_calibration,
    ksd_calibration,
)


@pytest.mark.parametrize("num_bins", [1, 2, 10, 37, 200])
@pytest.mark.parametrize("censoring", ["none", "all", "mixed"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_d_calibration_matches_individual_histograms(num_bins, censoring, dtype):
    rng = np.random.default_rng(71)
    edges = np.linspace(1, 0, num_bins + 1)
    # Include exact boundaries and their immediate neighbors, plus random
    # probabilities. This also covers p == 0 and p == 1 explicitly.
    probs = np.concatenate(
        (
            rng.uniform(size=128),
            edges,
            np.nextafter(edges[1:-1], 0),
            np.nextafter(edges[1:-1], 1),
        )
    ).astype(dtype)
    if censoring == "none":
        indicators = np.ones(probs.size, dtype=bool)
    elif censoring == "all":
        indicators = np.zeros(probs.size, dtype=bool)
    else:
        indicators = rng.random(probs.size) < 0.5

    expected_hist = np.zeros(num_bins)
    for prob, observed in zip(probs, indicators):
        if observed:
            position = max(np.digitize(prob, edges) - 1, 0)
            expected_hist[position] += 1
        else:
            expected_hist += create_censor_hist(prob, num_bins)
    expected_statistic, expected_pvalue = chisquare(expected_hist)

    probs.flags.writeable = False
    indicators.flags.writeable = False
    statistic, pvalue, hist = d_calibration(probs, indicators, num_bins)

    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(hist, expected_hist, rtol=tolerance, atol=1e-12)
    np.testing.assert_allclose(
        [statistic, pvalue],
        [expected_statistic, expected_pvalue],
        rtol=tolerance,
        atol=tolerance,
    )
    assert hist.sum() == pytest.approx(probs.size, rel=tolerance)


def test_d_calibration_handles_tiny_censored_probabilities_without_overflow():
    probs = np.array([0.0, np.nextafter(0.0, 1.0), 1e-300])
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        _, _, hist = d_calibration(probs, np.zeros(3, dtype=bool), num_bins=10)
    np.testing.assert_array_equal(hist, [0.0] * 9 + [3.0])


def test_ksd_calibration_preserves_right_censoring_on_cdf_scale() -> None:
    """Check KM jumps against hand-calculated risk sets, including tied censoring."""
    probs = np.array([0.9, 0.7, 0.7, 0.4, 0.2])
    indicators = np.array([1, 1, 0, 0, 1])
    probs.flags.writeable = False
    indicators.flags.writeable = False

    p_value, details = ksd_calibration(
        probs, indicators, return_details=True, n_resamples=199, seed=41
    )
    fig, ax = details["figure"]
    try:
        support, cdf = details["empirical_distribution"]
        np.testing.assert_allclose(support, [0.0, 0.0, 0.1, 0.3, 0.6, 0.8, 1.0])
        # Event risk sets are 5, 4, and 1. Censorings cause no CDF jump.
        np.testing.assert_allclose(cdf, [0.0, 0.0, 0.2, 0.4, 0.4, 1.0, 1.0])
        assert details["statistics"] == pytest.approx(0.4)
        assert details["p_value"] == p_value
        assert ksd_calibration(probs, indicators, n_resamples=199, seed=41) == (
            p_value,
            details["statistics"],
        )
        np.testing.assert_allclose(ax.lines[0].get_xdata(), support)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), cdf)
        assert ax.get_xlabel() == r"CDF Probability $F(t_i \mid x)$"
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "probs",
    [
        np.array([0.9, 0.7, 0.4, 0.2]),
        np.array([0.0, 0.2, 0.2, 0.7, 1.0]),
        np.array([0.3]),
    ],
)
def test_ksd_calibration_matches_uncensored_ks(probs: np.ndarray) -> None:
    """Reflection preserves the two-sided KS statistic, including ties and boundaries."""
    expected = kstest(probs, "uniform", method="exact")
    p_value, statistic = ksd_calibration(
        probs, np.ones(probs.size, dtype=bool), n_resamples=4999, seed=17
    )

    assert statistic == pytest.approx(expected.statistic)
    assert p_value == pytest.approx(expected.pvalue, abs=0.035)


def test_ksd_bootstrap_matches_independent_scipy_reference() -> None:
    """Check the null simulation using hand-derived censoring masses and SciPy KM."""
    probs = np.array([0.9, 0.7, 0.7, 0.4, 0.2])
    indicators = np.array([1, 1, 0, 0, 1])
    n_resamples, seed = 199, 52
    p_value, statistic = ksd_calibration(
        probs, indicators, n_resamples=n_resamples, seed=seed
    )

    # Reverse KM removes the event tied at 0.3 before its censoring update.
    # G drops from 1 to 2/3 at 0.3, then to 1/3 at 0.6. Remaining mass is at 1.
    censor_support = np.array([1.0 - 0.7, 1.0 - 0.4, 1.0])
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_resamples):
        events = rng.uniform(size=probs.size)
        censoring = rng.choice(censor_support, size=probs.size, p=[1 / 3] * 3)
        sample = CensoredData.right_censored(
            np.minimum(events, censoring), events > censoring
        )
        estimate = ecdf(sample).cdf
        support = np.r_[0.0, estimate.quantiles, 1.0]
        cdf = np.r_[0.0, estimate.probabilities, 1.0]
        # Evaluate both sides of every step, including the terminal jump.
        distance = max(
            np.max(np.abs(cdf - support)),
            np.max(np.abs(np.r_[0.0, cdf[:-1]] - support)),
        )
        exceedances += distance > statistic or np.isclose(
            distance, statistic, rtol=1e-13, atol=0
        )

    assert p_value == (1 + exceedances) / (n_resamples + 1)


def test_ksd_calibration_includes_terminal_jump_in_statistic() -> None:
    """Unresolved event mass stays flat until the forced jump at probability one."""
    p_value, details = ksd_calibration(
        np.array([0.9, 0.8, 0.7]),
        np.array([1, 0, 0]),
        return_details=True,
        n_resamples=19,
        seed=4,
    )
    try:
        support, cdf = details["empirical_distribution"]
        np.testing.assert_allclose(support, [0.0, 0.0, 0.1, 0.2, 0.3, 1.0])
        np.testing.assert_allclose(cdf, [0.0, 0.0, 1 / 3, 1 / 3, 1 / 3, 1.0])
        assert details["statistics"] == pytest.approx(2 / 3)
        assert 1 / 20 <= p_value <= 1
    finally:
        plt.close(details["figure"][0])


def test_ksd_calibration_immediate_censoring_contains_no_information() -> None:
    """Independent censoring at zero always reproduces the same completed curve."""
    p_value, statistic = ksd_calibration(
        np.ones(5), np.zeros(5, dtype=bool), n_resamples=19, seed=4
    )
    assert (p_value, statistic) == (1.0, 1.0)


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_ksd_calibration_bootstrap_pvalue_is_never_zero(probability: float) -> None:
    """Extreme observed events attain the minimum attainable bootstrap p-value."""
    p_value, statistic = ksd_calibration(
        np.full(5, probability), np.ones(5, dtype=bool), n_resamples=19, seed=4
    )
    assert statistic == 1.0
    assert p_value == 1 / 20


@pytest.mark.parametrize("n_resamples", [0, -1, 1.5, True])
def test_ksd_calibration_rejects_invalid_resample_counts(n_resamples: int) -> None:
    """A bootstrap requires a positive integer sample count."""
    with pytest.raises(ValueError, match="n_resamples"):
        ksd_calibration(np.array([0.5]), np.array([1]), n_resamples=n_resamples)


@pytest.mark.parametrize(
    "probs, indicators",
    [
        (np.array([0.5, 0.6]), np.array([1])),
        (np.array([[0.5, 0.6]]), np.array([1, 0])),
        (np.array([0.5]), np.array([2])),
        (np.array([-0.1]), np.array([1])),
        (np.array([1.1]), np.array([1])),
        (np.array([np.nan]), np.array([1])),
    ],
)
def test_ksd_calibration_rejects_invalid_observations(
    probs: np.ndarray,
    indicators: np.ndarray,
) -> None:
    """Reject invalid probability observations before fitting the censoring model."""
    with pytest.raises(ValueError):
        ksd_calibration(probs, indicators, n_resamples=19, seed=4)
