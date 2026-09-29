import numpy as np
import pytest

from SurvivalEVAL.Evaluations.MeanError import mean_error
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier


@pytest.mark.parametrize("error_type", ["absolute", "squared"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("log_scale", [False, True])
@pytest.mark.parametrize("truncation_time", [None, 6.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ipcw_t_matches_direct_surrogate_means(
    error_type, weighted, log_scale, truncation_time, dtype
):
    rng = np.random.default_rng(31)
    train_times = rng.integers(1, 12, size=80).astype(dtype)
    train_indicators = rng.random(80) < 0.65
    times = np.r_[rng.integers(1, 12, size=30), 12.0, 15.0].astype(dtype)
    indicators = rng.random(times.size) < 0.5
    indicators[-2:] = False
    predictions = rng.uniform(0.5, 15.0, size=times.size).astype(dtype)

    # Direct definition: average the strictly later, observed training events
    # separately for each censored test sample.
    observed_train_times = train_times[train_indicators]
    surrogates = times.astype(float)
    for i in np.flatnonzero(~indicators):
        later_times = observed_train_times[observed_train_times > times[i]]
        surrogates[i] = later_times.mean() if later_times.size else np.nan

    weights = np.ones(times.size)
    if weighted:
        km = KaplanMeier(train_times, train_indicators)
        weights[~indicators] = 1 - km.predict(times[~indicators])

    keep = ~np.isnan(surrogates)
    expected_times = surrogates[keep]
    expected_predictions = predictions[keep]
    if truncation_time is not None:
        expected_times = np.minimum(expected_times, truncation_time)
        expected_predictions = np.minimum(expected_predictions, truncation_time)
    errors = (
        np.log(expected_times) - np.log(expected_predictions)
        if log_scale
        else expected_times - expected_predictions
    )
    errors = np.abs(errors) if error_type == "absolute" else np.square(errors)
    expected = np.average(errors, weights=weights[keep])

    for values in (predictions, times, indicators, train_times, train_indicators):
        values.flags.writeable = False
    actual = mean_error(
        predictions,
        times,
        indicators,
        train_times,
        train_indicators,
        error_type=error_type,
        method="IPCW-T",
        weighted=weighted,
        log_scale=log_scale,
        truncation_time=truncation_time,
    )
    assert actual == pytest.approx(expected, rel=2e-6 if dtype == np.float32 else 1e-12)


def test_ipcw_t_excludes_tied_and_missing_later_events():
    # At time 2, only the event at time 8 belongs in the surrogate mean.
    # The censored test sample at time 8 has no later event and is excluded.
    value = mean_error(
        predicted_times=np.array([4.0, 100.0, 3.0]),
        event_times=np.array([2.0, 8.0, 2.0]),
        event_indicators=np.array([0, 0, 1]),
        train_event_times=np.array([8.0, 2.0, 10.0, 2.0, 1.0]),
        train_event_indicators=np.array([1, 1, 0, 1, 1]),
        method="IPCW-T",
        weighted=False,
    )
    assert value == pytest.approx((4.0 + 1.0) / 2)


@pytest.mark.parametrize("weighted", [False, True])
def test_ipcw_t_without_observed_training_events(weighted):
    with np.errstate(divide="ignore", invalid="ignore"):
        value = mean_error(
            predicted_times=np.array([4.0, 100.0]),
            event_times=np.array([2.0, 1.0]),
            event_indicators=np.array([1, 0]),
            train_event_times=np.array([1.0, 3.0]),
            train_event_indicators=np.array([0, 0]),
            method="IPCW-T",
            weighted=weighted,
        )
    assert value == 2.0


def test_ipcw_t_without_censored_test_samples():
    value = mean_error(
        predicted_times=np.array([4.0, 1.0]),
        event_times=np.array([2.0, 3.0]),
        event_indicators=np.array([1, 1]),
        train_event_times=np.array([1.0, 3.0]),
        train_event_indicators=np.array([1, 1]),
        method="IPCW-T",
    )
    assert value == 2.0
