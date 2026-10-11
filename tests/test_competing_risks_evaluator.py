import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from SurvivalEVAL import CompetingRisksEvaluator, cr_d_calibration


@pytest.fixture
def predictions():
    terminal = np.array([[0.75, 0.25], [0.25, 0.75], [0.5, 0.5]])
    cifs = np.stack(
        (
            np.zeros((3, 2)),
            np.array([[0.25, 0.125], [0.125, 0.25], [0.25, 0.125]]),
            terminal,
        ),
        axis=2,
    )
    return cifs, np.array([0, 2, 4]), np.array([1, 3, 4]), np.array([1, 0, 2])


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "method, reduction, alpha, return_details",
    [
        ("integral", "sum", 1, False),
        ("integral", "mean", 3, True),
        ("ks", "max", 2, True),
    ],
)
def test_curve_calibration_matches_observed_time_metric(
    predictions, interpolation, dtype, method, reduction, alpha, return_details
):
    cifs, times, observed, events = predictions
    cifs = cifs.astype(dtype)
    for values in (cifs, times, observed, events):
        values.flags.writeable = False
    evaluator = CompetingRisksEvaluator(cifs, times, observed, events, interpolation)

    at_observed = np.empty((3, 2))
    for i, time in enumerate(observed):
        for k in range(2):
            if interpolation == "Linear":
                at_observed[i, k] = np.interp(time, times, cifs[i, k])
            else:
                at_observed[i, k] = PchipInterpolator(times, cifs[i, k])(time)
    options = dict(
        num_bins=8,
        alpha=alpha,
        method=method,
        reduction=reduction,
        return_details=return_details,
    )
    expected = cr_d_calibration(
        at_observed, events, cifs[:, :, -1], 1 - at_observed.sum(axis=1), **options
    )
    actual = evaluator.cr_d_calibration(**options)
    np.testing.assert_equal(actual, expected)
    assert evaluator.pred_cifs is cifs
    np.testing.assert_array_equal(evaluator.event_indicators, [1, 0, 2])


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
def test_positive_start_grid_is_padded_without_mutating_inputs(
    predictions, interpolation
):
    cifs, times, _, events = predictions
    observed = np.array([0, 1, 3])
    positive_cifs, positive_times = cifs[:, :, 1:].copy(), times[1:].copy()
    positive_cifs.flags.writeable = False
    positive_times.flags.writeable = False
    padded = CompetingRisksEvaluator(
        positive_cifs, positive_times, observed, events, interpolation
    )
    explicit = CompetingRisksEvaluator(cifs, times, observed, events, interpolation)
    np.testing.assert_array_equal(padded.pred_cifs, cifs)
    np.testing.assert_array_equal(padded.time_coordinates, times)
    np.testing.assert_equal(
        padded.cr_d_calibration(return_details=True),
        explicit.cr_d_calibration(return_details=True),
    )
    np.testing.assert_array_equal(positive_cifs, cifs[:, :, 1:])
    np.testing.assert_array_equal(positive_times, [2, 4])


def test_finite_horizon_requires_explicit_eventual_probabilities(predictions):
    cifs, times, _, events = predictions
    terminal = cifs[:, :, -1]
    observed = np.array([1, 2, 3])
    evaluator = CompetingRisksEvaluator(cifs * 0.5, times, observed, events)
    with pytest.raises(ValueError, match="Terminal event probabilities.*sum to one"):
        evaluator.cr_d_calibration()

    at_observed = np.array([[0.0625, 0.03125], [0.0625, 0.125], [0.1875, 0.15625]])
    expected = cr_d_calibration(
        at_observed, events, terminal, 1 - at_observed.sum(axis=1)
    )
    assert evaluator.cr_d_calibration(pred_event_probs=terminal) == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="same shape"):
        evaluator.cr_d_calibration(pred_event_probs=terminal[:, 0])


@pytest.mark.parametrize(
    "change, match",
    [
        ("cif_rank", "pred_cifs must have shape"),
        ("sample_count", "one entry per predicted sample"),
        ("time_count", "CIF time axis"),
        ("time_rank", "time_coordinates must be a 1D"),
        ("one_time", "at least two points"),
        ("duplicate_time", "strictly increasing"),
        ("infinite_time", "finite, non-negative"),
        ("negative_time", "finite, non-negative"),
        ("observed_rank", "one entry per predicted sample"),
        ("negative_observed", "within the prediction horizon"),
        ("infinite_observed", "within the prediction horizon"),
        ("beyond_horizon", "within the prediction horizon"),
        ("fractional_cause", "integer cause labels"),
        ("negative_cause", "integer cause labels"),
        ("unknown_cause", "integer cause labels"),
        ("infinite_cause", "integer cause labels"),
        ("negative_cif", "probabilities in"),
        ("infinite_cif", "probabilities in"),
        ("decreasing_cif", "nondecreasing"),
        ("excess_mass", "sum to at most one"),
        ("interpolation", "interpolation must be"),
    ],
)
def test_invalid_curve_inputs(predictions, change, match):
    cifs, times, observed, events = (values.astype(float) for values in predictions)
    interpolation = "Linear"
    if change == "cif_rank":
        cifs = cifs[:, 0]
    elif change == "sample_count":
        cifs = cifs[:2]
    elif change == "time_count":
        cifs = cifs[:, :, :2]
    elif change == "time_rank":
        times = times[None, :]
    elif change == "one_time":
        cifs, times = cifs[:, :, :1], times[:1]
    elif change == "duplicate_time":
        times[1] = 0
    elif change == "infinite_time":
        times[-1] = np.inf
    elif change == "negative_time":
        times[0] = -1
    elif change == "observed_rank":
        observed, events = observed[:, None], events[:, None]
    elif change == "negative_observed":
        observed[0] = -1
    elif change == "infinite_observed":
        observed[0] = np.inf
    elif change == "beyond_horizon":
        observed[0] = 5
    elif change == "fractional_cause":
        events[0] = 0.5
    elif change == "negative_cause":
        events[0] = -1
    elif change == "unknown_cause":
        events[0] = 3
    elif change == "infinite_cause":
        events[0] = np.inf
    elif change == "negative_cif":
        cifs[0, 0, 0] = -0.1
    elif change == "infinite_cif":
        cifs[0, 0, 0] = np.inf
    elif change == "decreasing_cif":
        cifs[0, 0, 1] = 0.9
    elif change == "excess_mass":
        cifs[0, :, -1] = 0.75
    else:
        interpolation = "cubic"
    with pytest.raises(ValueError, match=match):
        CompetingRisksEvaluator(cifs, times, observed, events, interpolation)


def test_censoring_at_zero_survival_is_rejected(predictions):
    cifs, times, observed, events = predictions
    events[-1] = 0
    evaluator = CompetingRisksEvaluator(cifs, times, observed, events)
    with pytest.raises(ValueError, match="positive predicted survival"):
        evaluator.cr_d_calibration()
