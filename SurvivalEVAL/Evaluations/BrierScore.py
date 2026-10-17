from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter
from scipy.integrate import simpson, trapezoid

from SurvivalEVAL.Evaluations.custom_types import Numeric
from SurvivalEVAL.Evaluations.util import (
    _inverse_censoring_weights,
    check_and_convert,
    check_and_convert_event_data,
    validate_time_points,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    KaplanMeier,
    TurnbullEstimatorLifelines,
)


def _predict_survival_at_observed_times(
    model: WeibullAFTFitter,
    features: pd.DataFrame,
    times: np.ndarray,
) -> np.ndarray:
    """Predict one survival probability per subject at that subject's time.

    Lifelines evaluates every requested time for every supplied subject.
    Extract diagonals from bounded batches to avoid an n-by-n matrix when
    only n paired predictions are needed. Keep the fitted model's public
    prediction API, including its handling of zero and infinite times.
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

    Parameters
    ----------
    preds: np.ndarray, shape = (n_samples, )
        Estimated survival probabilities at the specific time for the testing samples.
    left_limits: np.ndarray, shape = (n_samples, )
        Actual left limit event/censor time for the testing samples.
    right_limits: np.ndarray, shape = (n_samples, )
        Actual right limit event/censor time for the testing samples.
    train_left_limits: np.ndarray, shape = (n_train_samples, )
        Actual left limit event/censor time for the training samples.
    train_right_limits: np.ndarray, shape = (n_train_samples, )
        Actual right limit event/censor time for the training samples.
    x: np.ndarray, shape = (n_samples, n_features), default: None
        Features for the testing samples. Use only when method is 'Tsouprou-conditional'.
    x_train: np.ndarray, shape = (n_train_samples, n_features), default: None
        Features for the training samples. Use only when method is 'Tsouprou-conditional'.
    target_time: numeric, default: None
        The specific time point for which to estimate the Brier score.
    method: str, default: "Tsouprou-marginal"
        Method to use for handling censoring. One of ['uncensored', 'Tsouprou-marginal', 'Tsouprou-conditional'].
        'uncensored': Exclude samples whose event status is ambiguous at the target time.
        'Tsouprou-marginal': Use marginal survival probabilities based on Turnbull estimator.
        'Tsouprou-conditional': Use conditional survival probabilities based on Weibull AFT model.
    Returns
    -------
    brier_score: float
        Value of the brier score.
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
    """Return column means after excluding selected cells.

    ``values`` is consumed in place so callers do not need another dense
    matrix for masking. Columns with no included values return ``nan``.
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
    """Calculate plain or IPCW Brier scores at multiple evaluation times.

    Parameters
    ----------
    pred_mat: np.ndarray, shape = (n_samples, n_time_points)
        Predicted probability array (2-D) for each instances at each time point.
    event_times: np.ndarray, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    event_indicators: np.ndarray, shape = (n_samples, )
        Binary event indicators for the testing samples: 1 denotes an observed
        event and 0 denotes a censored observation.
    train_event_times: np.ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators: np.ndarray, shape = (n_train_samples, )
        Binary event indicators for the training samples: 1 denotes an observed
        event and 0 denotes a censored observation.
    target_times: np.ndarray, shape = (n_time_points,)
        The specific time points for which to estimate the Brier scores.
    ipcw: bool, default: True
        Whether to use Inverse Probability of Censoring Weighting (IPCW) in the calculation.

    Returns
    -------
    brier_scores: np.ndarray, shape = (n_time_points, )
        Values of multiple Brier scores in target_times order.
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
    """Calculate cause-specific Brier scores from CIF probabilities.

    Args:
        pred_mat: Finite CIF probabilities in [0, 1], shape (n_samples, n_times).
        event_times: Finite, non-negative observed event or censoring times.
        event_indicators: Non-negative integer causes; 0 denotes censoring.
        train_event_times: Training outcomes, required for IPCW.
        train_event_indicators: Training causes. All positive labels are events
            when fitting the censoring distribution.
        target_times: Finite, non-negative evaluation times, in prediction order.
        ipcw: Use training reverse-KM censoring weights. Defaults to True.
        event_of_interest: Positive integer cause label. Defaults to 1.

    Returns:
        A Brier score per target time. Observed competing events have outcome 0
        and remain in the score. Earlier censorings contribute zero; the
        denominator is the full test sample size even without IPCW. A cause
        absent from the test data still has a defined score.

    Raises:
        ValueError: If inputs are invalid, IPCW training outcomes are missing,
            or a contributing censoring probability is nonfinite or nonpositive.

    Notes:
        Events by the target receive weight 1/G(event_time-); observations
        after the target and censorings exactly at it receive 1/G(target_time-).
        Without IPCW these contributions receive unit weights. Marginal IPCW
        assumes independent censoring; reverse KM retains its linear tail
        beyond training follow-up, matching the package's other IPCW metrics.

    References:
        Gerds and Schumacher (2006), https://doi.org/10.1002/bimj.200610301.
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
    """Calculate the equal-weight mean of cause-specific Brier scores.

    Args:
        pred_mat: Finite CIF probabilities in [0, 1], shape
            (n_samples, n_causes, n_times). Column k - 1 represents cause k.
        event_times: Finite, non-negative observed event or censoring times.
        event_indicators: Integer labels from 0 through n_causes; 0 is censoring.
        train_event_times: Training outcomes, required for IPCW.
        train_event_indicators: Non-negative integer training cause labels.
        target_times: Finite, non-negative evaluation times, in prediction order.
        ipcw: Use training reverse-KM censoring weights. Defaults to True.

    Returns:
        A mean Brier score per target time. Every predicted cause is included,
        including causes absent from the test outcomes. This averages binary
        cause-specific scores; it does not add an event-free-state score.

    Raises:
        ValueError: If inputs or required censoring probabilities are invalid.
            See brier_multiple_points_cr_per_event for weighting conventions.

    Notes:
        Censoring weights are calculated once for all causes. After input
        conversion, working memory beyond the inputs is O(n_samples * n_times),
        independent of n_causes.
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
    """Validate predictions and outcomes for survival or cause-specific Brier scores."""
    pred_mat = check_and_convert(pred_mat)
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
    # Validate one cause at a time so temporary masks do not grow with n_causes.
    matrices = (pred_mat,) if ndim == 2 else pred_mat.transpose(1, 0, 2)
    for predictions in matrices:
        if not np.all(np.isfinite(predictions)) or np.any(
            (predictions < 0) | (predictions > 1)
        ):
            raise ValueError("pred_mat must contain finite probabilities in [0, 1].")
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
    """Prepare shared event/event-free masks and their one-dimensional weights."""
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
    """Score survival (event_targets=None) or one cause using one float workspace."""
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
    """Resolve the IBS grid, retaining the survival evaluator's censor-time default."""
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
    scores: np.ndarray, target_times: np.ndarray, integration_method: str
) -> float:
    """Average the Brier curve over a validated positive time span."""
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

    Parameters
    ----------
    pred_mat: np.ndarray, shape = (n_samples, n_time_points)
        Predicted survival probabilities for each sample at each target time.
        pred_mat[i, j] ~= S_hat_i(target_times[j])
    left_limits: np.ndarray, shape = (n_samples,)
        Left interval bounds L_i.
    right_limits: np.ndarray, shape = (n_samples,)
        Right interval bounds R_i. Use np.inf for right-censoring (no observed event yet).
    target_times: np.ndarray, shape = (n_time_points,)
        Time points t_j at which to evaluate the Brier score.
    train_left_limits, train_right_limits:
        Training interval bounds, required for Tsouprou-based weighting.
    x, x_train:
        Feature arrays for conditional model ('Tsouprou-conditional').
        May be 1-D (n_samples,) or 2-D (n_samples, n_features).
    method: str
        One of ['uncensored', 'Tsouprou-marginal', 'Tsouprou-conditional'].

    Returns
    -------
    brier_scores: np.ndarray, shape = (n_time_points,)
        One Brier score per target time.
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
