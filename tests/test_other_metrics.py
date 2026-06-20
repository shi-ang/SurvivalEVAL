import numpy as np
import pytest

from SurvivalEVAL.Evaluations.AreaUnderROCurve import auc
from SurvivalEVAL.Evaluations.OtherMetrics import cov


def test_auc_treats_censored_at_target_time_as_event_free():
    value = auc(
        predict_probs=np.array([0.2, 0.1, 0.9]),
        event_times=np.array([1.0, 2.0, 3.0]),
        event_indicators=np.array([1, 0, 0]),
        target_time=2.0,
    )

    assert value == pytest.approx(0.5)


def test_cov_computes_event_time_coefficient_of_variation():
    cdf = np.array(
        [
            [0.0, 0.5, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
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
