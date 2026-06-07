import numpy as np
import pytest

from SurvivalEVAL.Evaluations.OtherMetrics import cov


def test_cov_computes_event_time_coefficient_of_variation():
    cdf = np.array([
        [0.0, 0.5, 1.0],
        [0.0, 1.0, 1.0],
    ])
    time_grid = np.array([0.0, 2.0, 4.0])

    mean_cov, cov_values = cov(cdf, time_grid, return_details=True)

    np.testing.assert_allclose(cov_values, [0.5, 0.0])
    assert mean_cov == pytest.approx(0.25)


def test_cov_normalizes_probability_mass_represented_by_grid():
    cdf = np.array([[0.2, 0.5, 0.8]])
    time_grid = np.array([0.0, 2.0, 4.0])

    assert cov(cdf, time_grid) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "cdf",
    [
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.8, 0.7]]),
    ],
)
def test_cov_rejects_invalid_probability_mass(cdf):
    with pytest.raises(ValueError):
        cov(cdf, np.array([0.0, 1.0, 2.0]))
