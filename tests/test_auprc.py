import numpy as np
import pytest
from scipy.interpolate import interp1d

import SurvivalEVAL.Evaluations.AreaUnderPRCurve as auprc_module


def _scipy_cdf_row(cdf, times, targets, left_fill=None):
    return interp1d(
        times,
        cdf,
        kind="linear",
        fill_value=(cdf[0] if left_fill is None else left_fill, 1.0),
        bounds_error=False,
        assume_sorted=True,
    )(targets)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("left_fill", [None, 0.0, 0.15])
def test_cdf_interpolation_matches_scipy_at_boundaries(dtype, left_fill):
    times = np.array([1.0, 2.0, 5.0, 8.0], dtype=dtype)
    cdf = np.array([0.1, 0.3, 0.3, 0.8], dtype=dtype)
    targets = np.array([0.0, 1.0, 1.5, 2.0, 4.5, 5.0, 8.0, 9.0, np.inf])
    times.flags.writeable = False
    cdf.flags.writeable = False
    np.testing.assert_allclose(
        auprc_module._interp_cdf_row(cdf, times, targets, left_fill),
        _scipy_cdf_row(cdf, times, targets, left_fill),
        rtol=1e-6 if dtype == np.float32 else 1e-12,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("per_sample_grid", [False, True])
def test_auprc_scores_match_scipy_interpolation(monkeypatch, dtype, per_sample_grid):
    rng = np.random.default_rng(87)
    times = np.linspace(0.0, 8.0, 40).astype(dtype)
    if per_sample_grid:
        times = times[None, :] * rng.uniform(0.5, 1.5, size=(20, 1)).astype(dtype)
    cdf = np.sort(rng.uniform(size=(20, 40)), axis=1).astype(dtype)
    observed_times = rng.uniform(0.0, 12.0, size=20).astype(dtype)
    right = observed_times + 3
    right[::4] = np.inf
    observed_times[1] = 0.0
    right[2] = observed_times[2]

    def scores():
        return (
            auprc_module.auprc_uncensored_grid(cdf, times, observed_times, n_quad=64),
            auprc_module.auprc_right_censored_grid(
                cdf, times, observed_times, n_quad=64
            ),
            auprc_module.auprc_ic(
                cdf, times, observed_times, right, n_quad=64, return_details=True
            )[1],
        )

    actual = scores()
    monkeypatch.setattr(auprc_module, "_interp_cdf_row", _scipy_cdf_row)
    expected = scores()
    np.testing.assert_allclose(
        actual, expected, rtol=1e-6 if dtype == np.float32 else 1e-12, atol=1e-12
    )


def test_auprc_uniform_cdf_has_known_integral():
    cdf = np.array([[0.0, 1.0]])
    times = np.array([0.0, 1.0])
    observed = np.array([1.0])
    # Both integrands reduce to 1 - t on [0, 1].
    assert auprc_module.auprc_uncensored_grid(cdf, times, observed)[0] == 0.5
    assert auprc_module.auprc_right_censored_grid(cdf, times, observed)[0] == 0.5
