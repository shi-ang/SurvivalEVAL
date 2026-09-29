import numpy as np
import pytest
from scipy.stats import chisquare

from SurvivalEVAL.Evaluations.DistributionCalibration import (
    create_censor_hist,
    d_calibration,
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
