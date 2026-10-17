import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

from SurvivalEVAL import (
    CompetingRisksEvaluator,
    concordance_cr,
    concordance_cr_per_event,
    cr_d_calibration,
)


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


@pytest.mark.parametrize("training", [False, True])
def test_default_target_time_reflects_current_outcomes(
    predictions, training: bool
) -> None:
    options = (
        {"train_event_times": [1, 2, 6], "train_event_indicators": [1, 0, 2]}
        if training
        else {}
    )
    evaluator = CompetingRisksEvaluator(*predictions, **options)
    assert evaluator.default_target_time == (2.5 if training else 3)

    evaluator.event_times[:] = [0.2, 0.4, 0.6]
    assert evaluator.default_target_time == pytest.approx(0.8 if training else 0.4)
    if training:
        evaluator.train_event_times[:] = [2, 3, 4]
        assert evaluator.default_target_time == pytest.approx(1.3)


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
        ("observed_rank", "1D arrays"),
        ("negative_observed", "finite and non-negative"),
        ("infinite_observed", "finite and non-negative"),
        ("beyond_horizon", "within the prediction horizon"),
        ("fractional_cause", "integer labels"),
        ("negative_cause", "integer labels"),
        ("unknown_cause", "integer cause labels"),
        ("infinite_cause", "integer labels"),
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


@pytest.mark.parametrize(
    "tau, expected",
    [
        (None, [[1.25, 0.5], [0.5, 1.25], [1.0, 0.75]]),
        (4, [[1.25, 0.5], [0.5, 1.25], [1.0, 0.75]]),
        (3, [[0.625, 0.28125], [0.28125, 0.625], [0.5625, 0.34375]]),
        (2, [[0.25, 0.125], [0.125, 0.25], [0.25, 0.125]]),
        (1, [[0.0625, 0.03125], [0.03125, 0.0625], [0.0625, 0.03125]]),
        (0, np.zeros((3, 2))),
    ],
)
def test_linear_rmtl_areas(predictions, tau, expected):
    evaluator = CompetingRisksEvaluator(*predictions)
    np.testing.assert_allclose(evaluator.predict_rmtl(tau), expected)


@pytest.mark.parametrize("tau", [0, 1, 2, 3, 4, None])
def test_pchip_rmtl_matches_numerical_quadrature(predictions, tau):
    evaluator = CompetingRisksEvaluator(*predictions, interpolation="Pchip")
    cifs, times, _, _ = predictions
    horizon = times[-1] if tau is None else tau
    expected = [
        [
            quad(PchipInterpolator(times, curve), 0, horizon, points=times[1:-1])[0]
            for curve in curves
        ]
        for curves in cifs
    ]
    np.testing.assert_allclose(evaluator.predict_rmtl(tau), expected, atol=1e-14)


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
def test_rmtl_preserves_readonly_noncontiguous_inputs(predictions, interpolation):
    inputs = [np.repeat(a.astype(np.float32), 2, axis=0)[::2] for a in predictions]
    for values in inputs:
        values.flags.writeable = False
    evaluator = CompetingRisksEvaluator(*inputs, interpolation=interpolation)
    expected = CompetingRisksEvaluator(*predictions, interpolation=interpolation)
    np.testing.assert_allclose(evaluator.predict_rmtl(3), expected.predict_rmtl(3))
    for actual, original in zip(inputs, predictions):
        np.testing.assert_array_equal(actual, original)


def test_default_concordance_uses_positive_rmtl(predictions):
    evaluator = CompetingRisksEvaluator(*predictions)
    assert evaluator.concordance() == 1
    assert evaluator.concordance_per_event(1) == (1, 2, 2)
    assert evaluator.concordance_per_event(2) == (1, 1, 1)


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("method", ["Naive", "IPCW"])
@pytest.mark.parametrize("predict_risk_method", ["RMTL", "CIF"])
@pytest.mark.parametrize("ties", ["None", "Time", "Risk", "All"])
@pytest.mark.parametrize("reduction", ["mean", "pooled"])
@pytest.mark.parametrize("control_definition", ["NotCase", "EventFree"])
def test_concordance_matches_standalone(
    predictions,
    interpolation,
    method,
    predict_risk_method,
    ties,
    reduction,
    control_definition,
):
    # Keep a censored subject through both events so both definitions are estimable.
    predictions = (predictions[0], predictions[1], np.array([1, 4, 4]), predictions[3])
    train_times = np.array([0.5, 1.5, 2.5, 5, 7])
    train_events = np.array([0, 1, 0, 2, 0])
    evaluator = CompetingRisksEvaluator(
        *predictions,
        interpolation=interpolation,
        train_event_times=train_times,
        train_event_indicators=train_events,
    )
    risks = (
        evaluator.predict_rmtl()
        if predict_risk_method == "RMTL"
        else predictions[0][..., 1]
    )
    risk_options = dict(
        predict_risk_method=predict_risk_method,
        target_time=2 if predict_risk_method == "CIF" else None,
    )
    options = dict(
        method=method,
        control_definition=control_definition,
        ties=ties,
        train_event_times=train_times,
        train_event_indicators=train_events,
    )
    expected = concordance_cr(
        risks,
        predictions[2],
        predictions[3],
        reduction=reduction,
        return_details=True,
        **options,
    )
    np.testing.assert_equal(
        evaluator.concordance(
            method=method,
            control_definition=control_definition,
            ties=ties,
            reduction=reduction,
            return_details=True,
            **risk_options,
        ),
        expected,
    )
    for cause in (1, 2):
        expected = concordance_cr_per_event(
            risks[:, cause - 1],
            predictions[2],
            predictions[3],
            event_of_interest=cause,
            **options,
        )
        np.testing.assert_allclose(
            evaluator.concordance_per_event(
                cause,
                method=method,
                control_definition=control_definition,
                ties=ties,
                **risk_options,
            ),
            expected,
        )


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("risk_method", ["RMTL", "CIF"])
@pytest.mark.parametrize("n_samples", [1, 3])
def test_selected_cause_risks_are_vectors(
    predictions, interpolation: str, risk_method: str, n_samples: int
) -> None:
    cifs, times, observed, events = predictions
    evaluator = CompetingRisksEvaluator(
        cifs[:n_samples], times, observed[:n_samples], events[:n_samples], interpolation
    )
    tau = 2.5 if risk_method == "RMTL" else None
    target_time = 2.5 if risk_method == "CIF" else None
    all_risks = evaluator.predict_risks(risk_method, tau, target_time)
    assert all_risks.shape == (n_samples, 2)
    for cause in (1, 2, np.int64(1), np.int32(2)):
        risks = evaluator.predict_risks(
            risk_method, tau, target_time, event_of_interest=cause
        )
        assert risks.shape == (n_samples,)
        np.testing.assert_allclose(risks, all_risks[:, cause - 1])


def test_tau_sets_rmtl_horizon_and_cif_uses_target_time():
    # The first subject has greater total loss, but less loss before year 2.
    cifs = np.array([[[0, 0.1, 1]], [[0, 0.3, 0.4]]])
    evaluator = CompetingRisksEvaluator(cifs, [0, 2, 4], [1, 2], [1, 1])
    assert evaluator.concordance() == 1
    assert evaluator.concordance(tau=2) == 0
    assert evaluator.concordance_per_event(tau=2) == (0, 0, 1)
    assert evaluator.concordance(predict_risk_method="CIF", target_time=2, tau=2) == 0
    assert evaluator.concordance(predict_risk_method="CIF", target_time=3, tau=2) == 1
    assert evaluator.concordance_per_event(
        predict_risk_method="CIF", target_time=3, tau=2
    ) == (1, 1, 1)

    assert np.isnan(evaluator.concordance(tau=1))  # Strict cutoff excludes time 1.


def test_missing_cause_remains_in_concordance_details(predictions):
    evaluator = CompetingRisksEvaluator(*predictions)
    score, details = evaluator.concordance(tau=3, return_details=True)
    assert np.isnan(score)
    assert details["per_event_scores"][1] == 1
    assert np.isnan(details["per_event_scores"][2])
    assert details["per_event_total_pairs"] == {1: 2, 2: 0}
    assert evaluator.concordance(tau=3, reduction="pooled") == 1


@pytest.mark.parametrize("tau", [-1, np.nan, np.inf, 5])
def test_invalid_rmtl_horizons(predictions, tau):
    evaluator = CompetingRisksEvaluator(*predictions)
    with pytest.raises(ValueError, match="tau"):
        evaluator.predict_rmtl(tau)
    with pytest.raises(ValueError, match="tau"):
        evaluator.concordance(tau=tau)
    with pytest.raises(ValueError, match="tau"):
        evaluator.concordance_per_event(tau=tau)


@pytest.mark.parametrize("cause", [None, 0, 3, 1.5, True, np.bool_(True)])
@pytest.mark.parametrize("metric", ["auc_per_event", "concordance_per_event"])
def test_invalid_per_event_cause(predictions: tuple, cause: object, metric: str) -> None:
    evaluator = CompetingRisksEvaluator(*predictions)
    with pytest.raises(ValueError, match="event_of_interest"):
        getattr(evaluator, metric)(cause)


def test_ipcw_concordance_requires_training_data(predictions):
    evaluator = CompetingRisksEvaluator(*predictions)
    with pytest.raises(ValueError, match="required for IPCW"):
        evaluator.concordance(method="IPCW")
    with pytest.raises(ValueError, match="required for IPCW"):
        evaluator.concordance_per_event(method="IPCW")


@pytest.mark.parametrize(
    "times, events, match",
    [
        ([1], None, "Supply both"),
        (None, [1], "Supply both"),
        ([[1]], [[1]], "must be 1D"),
        ([1, 2], [1], "same shape"),
        ([-1], [1], "finite and non-negative"),
        ([np.inf], [1], "finite and non-negative"),
        ([1], [-1], "integer labels"),
        ([1], [0.5], "integer labels"),
        ([1], [np.inf], "integer labels"),
    ],
)
def test_invalid_training_outcomes(predictions, times, events, match):
    with pytest.raises(ValueError, match=match):
        CompetingRisksEvaluator(
            *predictions, train_event_times=times, train_event_indicators=events
        )


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize(
    "risk_method, time",
    [
        ("RMTL", None),
        ("RMTL", 0),
        ("RMTL", 1),
        ("RMTL", 2),
        ("RMTL", 3),
        ("RMTL", 4),
        ("CIF", 0),
        ("CIF", 1),
        ("CIF", 2),
        ("CIF", 3),
        ("CIF", 4),
    ],
)
def test_per_call_risk_choices_match_standalone(
    predictions, interpolation, risk_method, time
):
    evaluator = CompetingRisksEvaluator(*predictions, interpolation=interpolation)
    if risk_method == "RMTL":
        risks = evaluator.predict_rmtl(time)
        options = dict(predict_risk_method=risk_method, tau=time)
    else:
        risks = np.array(
            [
                [
                    (
                        np.interp(time, predictions[1], curve)
                        if interpolation == "Linear"
                        else PchipInterpolator(predictions[1], curve)(time)
                    )
                    for curve in curves
                ]
                for curves in predictions[0]
            ]
        )
        options = dict(predict_risk_method=risk_method, target_time=time)
    tau = options.get("tau")
    np.testing.assert_equal(
        evaluator.concordance(return_details=True, **options),
        concordance_cr(
            risks, predictions[2], predictions[3], tau=tau, return_details=True
        ),
    )
    for cause in (1, 2):
        anchor_times = predictions[2][predictions[3] == cause]
        if tau is not None and not np.any(anchor_times < tau):
            with pytest.raises(ValueError, match="comparable pairs"):
                evaluator.concordance_per_event(cause, **options)
        else:
            expected = concordance_cr_per_event(
                risks[:, cause - 1],
                predictions[2],
                predictions[3],
                tau=tau,
                event_of_interest=cause,
            )
            np.testing.assert_allclose(
                evaluator.concordance_per_event(cause, **options), expected
            )


def test_risk_choice_and_horizon_do_not_persist_between_calls():
    cifs = np.array([[[0, 0.1, 1]], [[0, 0.3, 0.4]]])
    evaluator = CompetingRisksEvaluator(cifs, [0, 2, 4], [1, 2], [1, 1])
    assert evaluator.concordance() == 1
    assert evaluator.concordance(predict_risk_method="CIF", target_time=2) == 0
    assert evaluator.concordance(tau=2) == 0
    assert evaluator.concordance() == 1
    evaluator.pred_cifs = cifs[::-1]
    assert evaluator.concordance() == 0
    assert evaluator.concordance(predict_risk_method="CIF", target_time=2) == 1


@pytest.mark.parametrize("method_name", ["concordance", "concordance_per_event"])
@pytest.mark.parametrize(
    "options, error, match",
    [
        (dict(control_definition="unknown"), ValueError, "control_definition"),
        (dict(predict_risk_method="Median"), ValueError, "predict_risk_method"),
        (dict(predict_risk_method=None), ValueError, "predict_risk_method"),
        (dict(predict_risk_method="CIF"), ValueError, "target_time is required"),
        (dict(target_time=2), ValueError, "target_time is only used"),
        (dict(predict_risk_method="CIF", target_time=-1), ValueError, "target_time"),
        (dict(predict_risk_method="CIF", target_time=5), ValueError, "target_time"),
        (
            dict(predict_risk_method="CIF", target_time=np.nan),
            ValueError,
            "target_time",
        ),
        (
            dict(predict_risk_method="CIF", target_time=np.inf),
            ValueError,
            "target_time",
        ),
        (dict(predict_risk_method="CIF", target_time=[2]), TypeError, "numeric scalar"),
    ],
)
def test_invalid_risk_choices_and_target_times(
    predictions, method_name, options, error, match
):
    evaluator = CompetingRisksEvaluator(*predictions)
    with pytest.raises(error, match=match):
        getattr(evaluator, method_name)(**options)


def test_cif_target_time_is_independent_of_pair_cutoff(predictions):
    evaluator = CompetingRisksEvaluator(*predictions)
    options = dict(predict_risk_method="cIf", target_time=3)
    assert evaluator.concordance(**options, tau=5) == evaluator.concordance(**options)
    assert evaluator.concordance_per_event(
        1, **options, tau=5
    ) == evaluator.concordance_per_event(1, **options)
    assert np.isnan(evaluator.concordance(**options, tau=1))


def test_prediction_pair_update_pads_zero(predictions):
    evaluator = CompetingRisksEvaluator(*predictions)
    cifs = np.array([[[0.2, 0.6], [0.1, 0.4]]] * 3)
    evaluator.set_prediction_inputs(cifs, [2, 6])
    np.testing.assert_equal(evaluator.time_coordinates, [0, 2, 6])
    np.testing.assert_equal(evaluator.pred_cifs[..., 0], np.zeros((3, 2)))
    np.testing.assert_allclose(evaluator.predict_rmtl(), [[1.8, 1.1]] * 3)


def test_prediction_inputs_cannot_be_edited_in_place(predictions):
    cifs, times, observed, events = (a.copy() for a in predictions)
    evaluator = CompetingRisksEvaluator(cifs, times, observed, events)
    cifs *= 0.5
    times *= 2
    np.testing.assert_equal(evaluator.pred_cifs, predictions[0])
    np.testing.assert_equal(evaluator.time_coordinates, predictions[1])
    for values in (evaluator.pred_cifs, evaluator.time_coordinates):
        with pytest.raises(ValueError, match="read-only"):
            values.flat[0] = 0.5


@pytest.mark.parametrize(
    "setting, value",
    [
        ("pred_cifs", np.zeros((2, 2, 3))),
        ("pred_cifs", np.zeros((3, 1, 3))),
        ("pred_cifs", np.full((3, 2, 3), 0.75)),
        ("pred_cifs", np.full((3, 2, 3), np.inf)),
        ("time_coordinates", [0, 0, 4]),
        ("time_coordinates", [0, 1, 3]),
        ("time_coordinates", [0, 2, np.inf]),
        ("interpolation", "cubic"),
    ],
)
def test_invalid_setting_updates_preserve_inputs(predictions, setting, value):
    evaluator = CompetingRisksEvaluator(*predictions)
    curves, times = evaluator.pred_cifs, evaluator.time_coordinates
    with pytest.raises(ValueError):
        setattr(evaluator, setting, value)
    assert evaluator.pred_cifs is curves
    assert evaluator.time_coordinates is times
    assert evaluator.interpolation == "linear"


def test_rmtl_default_horizon_follows_updated_grid(predictions):
    evaluator = CompetingRisksEvaluator(*predictions)
    original = evaluator.predict_rmtl()
    evaluator.time_coordinates = evaluator.time_coordinates * 2
    np.testing.assert_allclose(evaluator.predict_rmtl(), original * 2)
