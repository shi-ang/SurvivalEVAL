from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from scipy.integrate import simpson, trapezoid

from SurvivalEVAL.evaluations.custom_types import Numeric
from SurvivalEVAL.evaluations.util import (
    _inverse_censoring_weights,
    check_and_convert_event_data,
    check_and_convert_probabilities,
    validate_time_points,
)
from SurvivalEVAL.nonparametric_estimator.single_event import (
    KaplanMeier,
    TurnbullEstimatorLifelines,
)


def _predict_survival_at_observed_times(
    model: WeibullAFTFitter,
    features: pd.DataFrame,
    times: np.ndarray,
) -> np.ndarray:
    """
    Predict one survival probability per subject at that subject's time.

    Lifelines evaluates every requested time for every supplied subject.
    Extract diagonals from bounded batches to avoid an n-by-n matrix when
    only n paired predictions are needed. Keep the fitted model's public
    prediction API, including its handling of zero and infinite times.

    Parameters
    ----------
    model : WeibullAFTFitter
        Fitted Weibull accelerated failure time model.
    features : pd.DataFrame
        Covariate rows in the same order as the requested subject times.
    times : ndarray
        One observed time per subject, aligned with the feature rows.

    Returns
    -------
    ndarray
        One predicted survival probability per subject at that subject's observed time.
    """
    probabilities = np.empty(times.size, dtype=float)
    batch_size = 256
    for start in range(0, times.size, batch_size):
        stop = min(start + batch_size, times.size)
        probabilities[start:stop] = (
            model.predict_survival_function(
                features.iloc[start:stop], times=times[start:stop]
            )
            .to_numpy()
            .diagonal()
        )
    return probabilities


def brier_score_ic(
    preds: np.ndarray,
    left_limits: np.ndarray,
    right_limits: np.ndarray,
    train_left_limits: np.ndarray | None = None,
    train_right_limits: np.ndarray | None = None,
    x: np.ndarray | None = None,
    x_train: np.ndarray | None = None,
    target_time: Numeric | None = None,
    method: str = "Tsouprou-marginal",
) -> float:
    """
    Calculate the Brier score at a specific time.

    The ``Tsouprou-marginal`` and ``Tsouprou-conditional`` methods adapt the
    interval-censoring approach of Tsouprou [1]_, using Turnbull and Weibull AFT
    survival estimates, respectively.

    Parameters
    ----------
    preds : ndarray, shape = (n_samples, )
        Estimated survival probabilities at the specific time for the testing samples.
    left_limits : ndarray, shape = (n_samples, )
        Actual left limit event/censor time for the testing samples.
    right_limits : ndarray, shape = (n_samples, )
        Actual right limit event/censor time for the testing samples.
    train_left_limits : ndarray, shape = (n_train_samples, )
        Actual left limit event/censor time for the training samples.
    train_right_limits : ndarray, shape = (n_train_samples, )
        Actual right limit event/censor time for the training samples.
    x : ndarray, shape = (n_samples, n_features), default=None
        Features for the testing samples. Use only when method is 'Tsouprou-conditional'.
    x_train : ndarray, shape = (n_train_samples, n_features), default=None
        Features for the training samples. Use only when method is 'Tsouprou-conditional'.
    target_time : numeric, default=None
        The specific time point for which to estimate the Brier score.
    method : str, default="Tsouprou-marginal"
        Method to use for handling censoring. One of ['uncensored', 'Tsouprou-marginal', 'Tsouprou-conditional'].
        'uncensored': Exclude samples whose event status is ambiguous at the target time.
        'Tsouprou-marginal': Use marginal survival probabilities based on Turnbull estimator.
        'Tsouprou-conditional': Use conditional survival probabilities based on Weibull AFT model.

    Returns
    -------
    brier_score : float
        Value of the brier score.

    References
    ----------
    .. [1] Tsouprou, Sofia (2015). "Measures of discrimination and predictive
       accuracy for interval censored survival data". Master's thesis,
       Leiden University. https://math.leidenuniv.nl/scripties/MasterTsouprou.pdf
    """
    if target_time is None:
        tau_vals = np.concatenate(
            [left_limits, right_limits[np.isfinite(right_limits)]]
        )
        tau = np.unique(np.sort(tau_vals))
        target_time = np.median(tau)

    method = method.lower()

    if method == "uncensored":
        # If the target time lies within the censoring interval, the event status
        # is ambiguous and the sample is excluded.
        mask = (left_limits <= target_time) & (right_limits > target_time)
        weight = 1 - mask.astype(float)
        # get the survival status at the target time
        # if the left limit is greater than the target time, then the event has not occurred, so 1
        # if the right limit is less than or equal to the target time, then the event has occurred, so 0
        survival_status = (left_limits > target_time).astype(float)
        brier_score = (np.square(preds - survival_status) * weight).sum() / weight.sum()
    elif method in {"tsouprou-marginal", "tsouprou-conditional"}:
        # method based on Sofia Tsouprou's thesis
        # Measures of discrimination and predictive accuracy for interval censored survival data
        # https://studenttheses.universiteitleiden.nl/access/item:3597164/view
        # the original method uses Weibull parametric model to estimate the survival function
        # here we give the option to use either Turnbull estimator or Weibull AFT model

        # must have training data
        if train_left_limits is None or train_right_limits is None:
            raise ValueError("Training data must be provided for Tsouprou methods.")

        if method == "tsouprou-marginal":
            marginal_estimator = TurnbullEstimatorLifelines(
                left=train_left_limits,
                right=train_right_limits,
            )
            # get the marginal survival probabilities at the target time
            left_probs = marginal_estimator.predict(left_limits)
            right_probs = marginal_estimator.predict(right_limits)
            target_probs = marginal_estimator.predict(target_time)
        elif method == "tsouprou-conditional":
            if x is None or x_train is None:
                raise ValueError(
                    "Features for both training and testing data must be provided for "
                    "Tsouprou-conditional method."
                )

            if x.ndim != x_train.ndim:
                raise ValueError(
                    "x and x_train must have the same number of dimensions."
                )

            train_data = {"left": train_left_limits, "right": train_right_limits}
            if x_train.ndim == 1:
                train_data["feature"] = x_train
            elif x_train.ndim == 2:
                for i in range(x_train.shape[1]):
                    train_data[f"feature_{i}"] = x_train[:, i]
            else:
                raise ValueError("x_train must be a 1-D or 2-D array.")
            train_df = pd.DataFrame(train_data)

            x_data = {}
            if x.ndim == 1:
                x_data["feature"] = x
            elif x.ndim == 2:
                for i in range(x.shape[1]):
                    x_data[f"feature_{i}"] = x[:, i]
            else:
                raise ValueError("x must be a 1-D or 2-D array.")
            x_df = pd.DataFrame(x_data)

            aft_model = WeibullAFTFitter()
            aft_model.fit_interval_censoring(train_df, "left", "right")
            # get the conditional survival probabilities at the left limit, right limit, and target time
            left_probs = _predict_survival_at_observed_times(
                aft_model, x_df, left_limits
            )
            right_probs = _predict_survival_at_observed_times(
                aft_model, x_df, right_limits
            )
            target_probs = aft_model.predict_survival_function(
                x_df, target_time
            ).values.flatten()
        else:
            raise ValueError(f"Method {method} is not supported.")
        # exam on non-bad indices
        # bad indices are those (1) the target time is strictly inside the
        # interval and (2) the left and right survival
        # probabilities are the same, which leads to zeros in both numerator and denominator in survival_status
        bad = (
            (left_probs == right_probs)
            & (left_limits < target_time)
            & (target_time < right_limits)
        )
        if np.sum(bad) > 0:
            left_limits = left_limits[~bad]
            right_limits = right_limits[~bad]
            preds = preds[~bad]
            left_probs = left_probs[~bad]
            right_probs = right_probs[~bad]
            if isinstance(target_time, np.ndarray):
                target_probs = target_probs[~bad]

        # supress warnings for divide by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            survival_status = (target_probs - right_probs) / (left_probs - right_probs)
        # Intervals are left-open, right-closed: (left, right].
        # At t <= left the subject is alive; at t >= right the event has occurred.
        survival_status[left_limits >= target_time] = 1
        survival_status[right_limits <= target_time] = 0

        if np.any((survival_status < 0) | (survival_status > 1)):
            raise ValueError(
                "Calculated survival status contains values outside [0, 1]."
            )

        # calculate the brier score
        brier_score = np.mean(np.square(preds - survival_status))
    else:
        raise ValueError(f"Method {method} is not supported.")
    return brier_score


def _columnwise_mean_excluding(
    values: np.ndarray,
    excluded: np.ndarray,
) -> np.ndarray:
    """
    Return column means after excluding selected cells.

    ``values`` is consumed in place so callers do not need another dense
    matrix for masking. Columns with no included values return ``nan``.

    Parameters
    ----------
    values : ndarray
        Two-dimensional numeric workspace modified in place to form column sums.
    excluded : ndarray
        Boolean mask of cells to omit when calculating column means.

    Returns
    -------
    ndarray
        One mean per column; columns without included cells contain NaN.
    """
    values[excluded] = 0.0
    included_counts = values.shape[0] - np.count_nonzero(excluded, axis=0)
    return np.divide(
        values.sum(axis=0, dtype=float),
        included_counts,
        out=np.full(values.shape[1], np.nan, dtype=float),
        where=included_counts > 0,
    )


def brier_multiple_points(
    pred_mat: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None,
    train_event_indicators: np.ndarray | None,
    target_times: np.ndarray,
    ipcw: bool = True,
) -> np.ndarray:
    """
    Calculate plain or IPCW Brier scores at multiple evaluation times.

    The Brier score measures squared probability error [1]_. With ``ipcw=True``,
    right-censoring is handled using inverse probability of censoring weights
    as proposed by Graf et al. [2]_ and studied by Gerds and Schumacher [3]_.

    Parameters
    ----------
    pred_mat : ndarray, shape = (n_samples, n_time_points)
        Predicted probability array (2-D) for each instances at each time point.
    event_times : ndarray, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    event_indicators : ndarray, shape = (n_samples, )
        Binary event indicators for the testing samples: 1 denotes an observed
        event and 0 denotes a censored observation.
    train_event_times : ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators : ndarray, shape = (n_train_samples, )
        Binary event indicators for the training samples: 1 denotes an observed
        event and 0 denotes a censored observation.
    target_times : ndarray, shape = (n_time_points,)
        The specific time points for which to estimate the Brier scores.
    ipcw : bool, default=True
        Whether to use Inverse Probability of Censoring Weighting (IPCW) in the calculation.

    Returns
    -------
    brier_scores : ndarray, shape = (n_time_points, )
        Values of multiple Brier scores in target_times order.

    References
    ----------
    .. [1] Brier, G. W. (1950). "Verification of forecasts expressed in terms
       of probability". Monthly Weather Review, 78(1), 1-3.
       https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
    .. [2] Graf, E.; Schmoor, C.; Sauerbrei, W.; Schumacher, M. (1999).
       "Assessment and comparison of prognostic classification schemes for
       survival data". Statistics in Medicine, 18(17-18), 2529-2545.
       https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5
    .. [3] Gerds, T. A.; Schumacher, M. (2006). "Consistent Estimation of the
       Expected Brier Score in General Survival Models with Right-Censored
       Event Times". Biometrical Journal, 48(6), 1029-1040.
       https://doi.org/10.1002/bimj.200610301
    """
    pred_mat, event_times, event_indicators, target_times = _check_brier_inputs(
        pred_mat, event_times, event_indicators, target_times
    )
    components = _brier_components(
        event_times,
        event_indicators,
        target_times,
        train_event_times,
        train_event_indicators,
        ipcw,
    )
    return _brier_scores(pred_mat, None, *components)


def brier_multiple_points_cr_per_event(
    pred_mat: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None,
    train_event_indicators: np.ndarray | None,
    target_times: np.ndarray,
    ipcw: bool = True,
    event_of_interest: int = 1,
) -> np.ndarray:
    """
    Calculate cause-specific Brier scores from CIF probabilities.

    The competing-risk prediction error follows Schoop et al. [2]_, extending
    the IPCW approach of Gerds and Schumacher [1]_.

    Parameters
    ----------
    pred_mat : ndarray
        Finite CIF probabilities in [0, 1], shape (n_samples, n_times).
    event_times : ndarray
        Finite, non-negative observed event or censoring times.
    event_indicators : ndarray
        Non-negative integer causes; 0 denotes censoring.
    train_event_times : ndarray | None
        Training outcomes, required for IPCW.
    train_event_indicators : ndarray | None
        Training causes. All positive labels are events
        when fitting the censoring distribution.
    target_times : ndarray
        Finite, non-negative evaluation times, in prediction order.
    ipcw : bool
        Use training reverse-KM censoring weights. Defaults to True.
    event_of_interest : int
        Positive integer cause label. Defaults to 1.

    Returns
    -------
    ndarray
        A Brier score per target time. Observed competing events have outcome 0
        and remain in the score. Earlier censorings contribute zero; the
        denominator is the full test sample size even without IPCW. A cause
        absent from the test data still has a defined score.

    Raises
    ------
    ValueError
        If inputs are invalid, IPCW training outcomes are missing,
        or a contributing censoring probability is nonfinite or nonpositive.

    References
    ----------
    .. [1] Gerds, T. A.; Schumacher, M. (2006). "Consistent Estimation of the
       Expected Brier Score in General Survival Models with Right-Censored
       Event Times". Biometrical Journal, 48(6), 1029-1040.
       https://doi.org/10.1002/bimj.200610301
    .. [2] Schoop, R.; Beyersmann, J.; Schumacher, M.; Binder, H. (2011).
       "Quantifying the predictive accuracy of time-to-event models in the
       presence of competing risks". Biometrical Journal, 53(1), 88-112.
       https://doi.org/10.1002/bimj.201000073
    """
    if (
        isinstance(event_of_interest, (bool, np.bool_))
        or not isinstance(event_of_interest, (int, np.integer))
        or event_of_interest < 1
    ):
        raise ValueError("event_of_interest must be a positive integer cause label.")
    pred_mat, event_times, event_indicators, target_times = _check_brier_inputs(
        pred_mat, event_times, event_indicators, target_times, competing_risks=True
    )
    components = _brier_components(
        event_times,
        event_indicators,
        target_times,
        train_event_times,
        train_event_indicators,
        ipcw,
        competing_risks=True,
    )
    return _brier_scores(pred_mat, event_indicators == event_of_interest, *components)


def brier_multiple_points_cr(
    pred_mat: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None,
    train_event_indicators: np.ndarray | None,
    target_times: np.ndarray,
    ipcw: bool = True,
) -> np.ndarray:
    """
    Calculate the equal-weight mean of cause-specific Brier scores.

    Each component uses the competing-risk prediction error of Schoop et al.
    [1]_. This function averages those scores equally across predicted causes.

    Parameters
    ----------
    pred_mat : ndarray
        Finite CIF probabilities in [0, 1], shape
        (n_samples, n_causes, n_times). Column k - 1 represents cause k.
    event_times : ndarray
        Finite, non-negative observed event or censoring times.
    event_indicators : ndarray
        Integer labels from 0 through n_causes; 0 is censoring.
    train_event_times : ndarray | None
        Training outcomes, required for IPCW.
    train_event_indicators : ndarray | None
        Non-negative integer training cause labels.
    target_times : ndarray
        Finite, non-negative evaluation times, in prediction order.
    ipcw : bool
        Use training reverse-KM censoring weights. Defaults to True.

    Returns
    -------
    ndarray
        A mean Brier score per target time. Every predicted cause is included,
        including causes absent from the test outcomes. This averages binary
        cause-specific scores; it does not add an event-free-state score.

    Raises
    ------
    ValueError
        If inputs or required censoring probabilities are invalid.
        See brier_multiple_points_cr_per_event for weighting conventions.

    References
    ----------
    .. [1] Schoop, R.; Beyersmann, J.; Schumacher, M.; Binder, H. (2011).
       "Quantifying the predictive accuracy of time-to-event models in the
       presence of competing risks". Biometrical Journal, 53(1), 88-112.
       https://doi.org/10.1002/bimj.201000073
    """
    pred_mat, event_times, event_indicators, target_times = _check_brier_inputs(
        pred_mat,
        event_times,
        event_indicators,
        target_times,
        competing_risks=True,
        ndim=3,
    )
    if np.any(event_indicators > pred_mat.shape[1]):
        raise ValueError("Event labels must not exceed the number of predicted causes.")
    components = _brier_components(
        event_times,
        event_indicators,
        target_times,
        train_event_times,
        train_event_indicators,
        ipcw,
        competing_risks=True,
    )
    scores = np.zeros(target_times.size, dtype=float)
    for cause, predictions in enumerate(pred_mat.transpose(1, 0, 2), start=1):
        scores += _brier_scores(predictions, event_indicators == cause, *components)
    return scores / pred_mat.shape[1]


def _check_brier_inputs(
    pred_mat: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    target_times: np.ndarray,
    *,
    competing_risks: bool = False,
    ndim: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate predictions and outcomes for survival or cause-specific Brier scores.

    Parameters
    ----------
    pred_mat : ndarray
        Predicted probabilities with sample axis first and evaluation-time axis last.
    event_times : ndarray
        Observed event or censoring times, one per sample.
    event_indicators : ndarray
        Observed status labels; 0 denotes censoring and positive labels denote observed events.
    target_times : ndarray
        One-dimensional array of non-negative evaluation times.
    competing_risks : bool
        Whether positive event labels may identify multiple competing causes.
    ndim : int
        Expected number of dimensions of the prediction array.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        Validated predictions, observed times, event labels, and target times, in that order.
    """
    pred_mat = check_and_convert_probabilities(pred_mat, "pred_mat")
    event_times, event_indicators = check_and_convert_event_data(
        event_times, event_indicators, competing_risks=competing_risks
    )
    target_times = validate_time_points(target_times, input_name="target_times")
    if not np.all(np.isfinite(target_times)):
        raise ValueError("target_times must be finite.")
    if (
        pred_mat.ndim != ndim
        or pred_mat.shape[0] != event_times.size
        or pred_mat.shape[-1] != target_times.size
    ):
        shape = (
            "(n_samples, n_time_points)"
            if ndim == 2
            else "(n_samples, n_causes, n_time_points)"
        )
        raise ValueError(f"pred_mat must have shape {shape}, got {pred_mat.shape}.")
    return pred_mat, event_times, event_indicators, target_times


def _brier_components(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    target_times: np.ndarray,
    train_event_times: np.ndarray | None,
    train_event_indicators: np.ndarray | None,
    ipcw: bool,
    *,
    competing_risks: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare shared event/event-free masks and their one-dimensional weights.

    Parameters
    ----------
    event_times : ndarray
        Observed event or censoring times, one per sample.
    event_indicators : ndarray
        Observed status labels; 0 denotes censoring and positive labels denote observed events.
    target_times : ndarray
        One-dimensional array of non-negative evaluation times.
    train_event_times : ndarray | None
        Training event or censoring times used to estimate the censoring distribution.
    train_event_indicators : ndarray | None
        Training status labels, with 0 for censoring and positive labels for observed events.
    ipcw : bool
        Whether to apply inverse probability of censoring weights estimated from the training outcomes.
    competing_risks : bool
        Whether positive event labels may identify multiple competing causes.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        Event-before-target and event-free masks, followed by event-time and target-time censoring weights.
    """
    observed = event_indicators > 0
    event_before = event_times[:, None] <= target_times[None, :]
    event_before &= observed[:, None]
    event_free = event_times[:, None] > target_times[None, :]
    censored_at_target = event_times[:, None] == target_times[None, :]
    censored_at_target &= ~observed[:, None]
    event_free |= censored_at_target
    del censored_at_target
    event_weights = np.ones(event_times.size, dtype=float)
    target_weights = np.ones(target_times.size, dtype=float)
    if ipcw:
        if train_event_times is None or train_event_indicators is None:
            raise ValueError(
                "Training event times and indicators must be provided for IPCW weighting."
            )
        train_times, train_events = check_and_convert_event_data(
            train_event_times, train_event_indicators, competing_risks=competing_risks
        )
        model = KaplanMeier(train_times, train_events > 0, reverse=True)
        contributing_events = event_before.any(axis=1)
        event_weights[contributing_events] = _inverse_censoring_weights(
            model.predict(event_times[contributing_events], side="left")
        )
        contributing_targets = event_free.any(axis=0)
        target_weights[contributing_targets] = _inverse_censoring_weights(
            model.predict(target_times[contributing_targets], side="left")
        )
    return event_before, event_free, event_weights, target_weights


def _brier_scores(
    pred_mat: np.ndarray,
    event_targets: np.ndarray | None,
    event_before: np.ndarray,
    event_free: np.ndarray,
    event_weights: np.ndarray,
    target_weights: np.ndarray,
) -> np.ndarray:
    """
    Score survival (event_targets=None) or one cause using one float workspace.

    Parameters
    ----------
    pred_mat : ndarray
        Predicted probabilities with sample axis first and evaluation-time axis last.
    event_targets : ndarray | None
        Boolean mask selecting the event cause for each sample; None evaluates survival.
    event_before : ndarray
        Boolean array of observed events at or before each target, with shape (n_samples, n_times).
    event_free : ndarray
        Boolean array of samples known to be event-free at each target, with shape (n_samples, n_times).
    event_weights : ndarray
        One inverse censoring weight per sample evaluated just before its observed event.
    target_weights : ndarray
        One inverse censoring weight per evaluation time.

    Returns
    -------
    ndarray
        Brier scores with shape (n_times,).
    """
    component_error = np.subtract(
        pred_mat, 0.0 if event_targets is None else event_targets[:, None], dtype=float
    )
    np.square(component_error, out=component_error)
    component_error *= event_before
    component_error *= event_weights[:, None]
    scores = component_error.sum(axis=0)
    del component_error

    component_error = np.subtract(
        pred_mat, 1.0 if event_targets is None else 0.0, dtype=float
    )
    np.square(component_error, out=component_error)
    component_error *= event_free
    component_error *= target_weights[None, :]
    scores += component_error.sum(axis=0)
    return scores / pred_mat.shape[0]


def _brier_time_grid(
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None,
    num_points: int | None,
    target_times: np.ndarray | None,
) -> np.ndarray:
    """
    Resolve the IBS grid, retaining the survival evaluator's censor-time default.

    Parameters
    ----------
    event_times : ndarray
        Observed event or censoring times, one per sample.
    event_indicators : ndarray
        Observed status labels; 0 denotes censoring and positive labels denote observed events.
    train_event_times : ndarray | None
        Training event or censoring times used to estimate the censoring distribution.
    num_points : int | None
        Number of equally spaced evaluation times; mutually exclusive with target_times.
    target_times : ndarray | None
        One-dimensional array of non-negative evaluation times.

    Returns
    -------
    ndarray
        Sorted evaluation times spanning a positive interval for integration.
    """
    if num_points is not None and target_times is not None:
        raise ValueError(
            "Please provide either `num_points` OR `target_times`, not both."
        )
    if target_times is not None:
        times = np.asarray(target_times, dtype=float)
        if times.ndim != 1:
            raise ValueError("`target_times` must be a 1D array of times.")
    elif num_points is not None:
        if (
            isinstance(num_points, (bool, np.bool_))
            or not isinstance(num_points, (int, np.integer))
            or num_points < 2
        ):
            raise ValueError(
                "`num_points` must be an integer >= 2 to perform integration."
            )
        max_time = event_times.max()
        if train_event_times is not None:
            max_time = max(max_time, train_event_times.max())
        times = np.linspace(0.0, max_time, num_points)
    else:
        times = np.unique(event_times[event_indicators == 0])
        if times.size < 2:
            raise ValueError(
                "Could not infer `target_times` from censored samples. "
                "Please provide `num_points` or `target_times`."
            )
    if times.size < 2:
        raise ValueError(
            "`target_times` must contain at least 2 time values for integration."
        )
    times = validate_time_points(times, input_name="target_times")
    if not np.all(np.isfinite(times)):
        raise ValueError("target_times must be finite.")
    if np.any(np.diff(times) < 0):
        warnings.warn("`target_times` is not sorted; sorting it now.", stacklevel=2)
        times = np.sort(times)
    if times[-1] <= times[0]:
        raise ValueError("target_times must span a positive range for IBS integration.")
    return times


def _integrate_brier_scores(
    scores: np.ndarray,
    target_times: np.ndarray,
    integration_method: str,
) -> float:
    """
    Average the Brier curve over a validated positive time span.

    Parameters
    ----------
    scores : ndarray
        One Brier score per evaluation time.
    target_times : ndarray
        One-dimensional array of non-negative evaluation times.
    integration_method : str
        Numerical integration rule, "trapz" or "simpson".

    Returns
    -------
    float
        Integrated score divided by the length of the evaluation interval.
    """
    integration_method = integration_method.lower()
    if integration_method == "trapz":
        integral = trapezoid(scores, x=target_times)
    elif integration_method == "simpson":
        integral = simpson(scores, x=target_times)
    else:
        raise ValueError(
            f"Integration method '{integration_method}' not supported. Use 'trapz' or 'simpson'."
        )
    return float(integral / (target_times[-1] - target_times[0]))


def brier_multiple_points_ic(
    pred_mat: np.ndarray,
    left_limits: np.ndarray,
    right_limits: np.ndarray,
    target_times: np.ndarray,
    train_left_limits: np.ndarray | None = None,
    train_right_limits: np.ndarray | None = None,
    x: np.ndarray | None = None,
    x_train: np.ndarray | None = None,
    method: str = "Tsouprou-marginal",
) -> np.ndarray:
    """
    Compute Brier scores at multiple target times for interval-censored data.

    The ``Tsouprou-marginal`` and ``Tsouprou-conditional`` methods adapt the
    interval-censoring approach of Tsouprou [1]_, using Turnbull and Weibull AFT
    survival estimates, respectively.

    Parameters
    ----------
    pred_mat : ndarray, shape = (n_samples, n_time_points)
        Predicted survival probabilities for each sample at each target time.
        pred_mat[i, j] ~= S_hat_i(target_times[j])
    left_limits : ndarray, shape = (n_samples,)
        Left interval bounds L_i.
    right_limits : ndarray, shape = (n_samples,)
        Right interval bounds R_i. Use np.inf for right-censoring (no observed event yet).
    target_times : ndarray, shape = (n_time_points,)
        Time points t_j at which to evaluate the Brier score.
    train_left_limits, train_right_limits : ndarray or None
        Training interval bounds, required for Tsouprou-based weighting.
    x, x_train : ndarray or None
        Feature arrays for conditional model ('Tsouprou-conditional').
        May be 1-D (n_samples,) or 2-D (n_samples, n_features).
    method : str
        One of ['uncensored', 'Tsouprou-marginal', 'Tsouprou-conditional'].

    Returns
    -------
    brier_scores : ndarray, shape = (n_time_points,)
        One Brier score per target time.

    References
    ----------
    .. [1] Tsouprou, Sofia (2015). "Measures of discrimination and predictive
       accuracy for interval censored survival data". Master's thesis,
       Leiden University. https://math.leidenuniv.nl/scripties/MasterTsouprou.pdf
    """

    # -------------------------
    # Basic shape checks
    # -------------------------
    if target_times.ndim != 1:
        raise TypeError("'target_times' must be one-dimensional.")

    if left_limits.ndim != 1:
        raise ValueError("left_limits must be one-dimensional.")

    n_samples = left_limits.shape[0]
    n_times = target_times.shape[0]
    method = method.lower()

    if method not in {"uncensored", "tsouprou-marginal", "tsouprou-conditional"}:
        raise ValueError(f"Method {method} is not supported.")

    if pred_mat.shape != (n_samples, n_times):
        raise ValueError(
            f"pred_mat must have shape (n_samples, n_time_points) = "
            f"({n_samples}, {n_times}), got {pred_mat.shape}"
        )
    if right_limits.shape != (n_samples,):
        raise ValueError("left_limits and right_limits must have the same length.")
    if method in {"tsouprou-marginal", "tsouprou-conditional"} and (
        train_left_limits is None or train_right_limits is None
    ):
        raise ValueError("Training data must be provided for Tsouprou methods.")
    if method == "tsouprou-conditional" and (x is None or x_train is None):
        raise ValueError("x and x_train must be provided for Tsouprou-conditional.")

    # Column and row views broadcast without materializing repeated inputs.
    left_column = left_limits[:, None]
    right_column = right_limits[:, None]
    target_times_row = target_times[None, :]

    # ============================================================
    # Case 1: 'uncensored' (naive treating intervals like exact-ish)
    # ============================================================
    if method == "uncensored":
        # For each (i,j), define survival_status_ij in {0,1}:
        #   if t_j < L_i  -> alive -> 1
        #   if t_j >= R_i -> dead  -> 0
        #   if L_i <= t_j < R_i -> ambiguous -> exclude from averaging at that t_j
        #
        # Note: if R_i == inf (right-censored), then t_j >= R_i is False for finite t_j,
        # so status will be 1 unless t_j is in [L_i, inf) which becomes ambiguous/excluded.
        # This matches "skip samples where event time is not pinned down yet".

        excluded = target_times_row >= left_column
        excluded &= target_times_row < right_column
        squared_error = (target_times_row < left_column).astype(float)
        np.subtract(pred_mat, squared_error, out=squared_error)
        np.square(squared_error, out=squared_error)
        return _columnwise_mean_excluding(squared_error, excluded)

    # Validation above leaves one of the two Tsouprou methods here.
    if method == "tsouprou-marginal":
        marginal_estimator = TurnbullEstimatorLifelines(
            left=train_left_limits,
            right=train_right_limits,
        )
        left_probs = marginal_estimator.predict(left_limits)
        right_probs = marginal_estimator.predict(right_limits)
        target_probs_mat = marginal_estimator.predict(target_times)[None, :]
    else:
        train_data = {"left": train_left_limits, "right": train_right_limits}
        if x_train.ndim == 1:
            train_data["feature"] = x_train
        elif x_train.ndim == 2:
            for k in range(x_train.shape[1]):
                train_data[f"feature_{k}"] = x_train[:, k]
        else:
            raise ValueError("x_train must be a 1-D or 2-D array.")
        train_df = pd.DataFrame(train_data)

        x_data = {}
        if x.ndim == 1:
            x_data["feature"] = x
        elif x.ndim == 2:
            for k in range(x.shape[1]):
                x_data[f"feature_{k}"] = x[:, k]
        else:
            raise ValueError("x must be a 1-D or 2-D array.")
        x_df = pd.DataFrame(x_data)

        aft_model = WeibullAFTFitter()
        aft_model.fit_interval_censoring(train_df, "left", "right")

        left_probs = _predict_survival_at_observed_times(aft_model, x_df, left_limits)
        right_probs = _predict_survival_at_observed_times(aft_model, x_df, right_limits)
        target_probs_mat = (
            aft_model.predict_survival_function(x_df, times=target_times).to_numpy().T
        )

    left_probs_column = left_probs[:, None]
    right_probs_column = right_probs[:, None]
    denominator = left_probs_column - right_probs_column

    with np.errstate(divide="ignore", invalid="ignore"):
        survival_status = (target_probs_mat - right_probs_column) / denominator

    # Intervals are left-open and right-closed: subjects are alive at the
    # left boundary and dead at the right boundary.
    survival_status[target_times_row <= left_column] = 1.0
    survival_status[target_times_row >= right_column] = 0.0

    inside_interval = target_times_row > left_column
    inside_interval &= target_times_row < right_column
    undefined = (denominator == 0.0) & inside_interval
    del inside_interval

    out_of_bounds = survival_status < 0.0
    out_of_bounds |= survival_status > 1.0
    out_of_bounds[undefined] = False
    if np.any(out_of_bounds):
        raise ValueError(
            "Calculated survival status contains values outside [0,1] for some "
            "non-excluded entries."
        )
    del out_of_bounds

    squared_error = survival_status
    np.subtract(pred_mat, squared_error, out=squared_error)
    np.square(squared_error, out=squared_error)
    return _columnwise_mean_excluding(squared_error, undefined)
