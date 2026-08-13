from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

from SurvivalEVAL.Evaluations.custom_types import Numeric
from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    KaplanMeier,
    TurnbullEstimatorLifelines,
)


def single_brier_score(
    preds: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray,
    train_event_indicators: np.ndarray,
    target_time: float | None = None,
    ipcw: bool = True,
) -> float:
    """
    Calculate the Brier score at a specific time.

    Parameters
    ----------
    preds: np.ndarray, shape = (n_samples, )
        Estimated survival probabilities at the specific time for the testing samples.
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
    target_time: float, default: None
        The specific time point for which to estimate the Brier score.
    ipcw: bool, default: True
        Whether to use Inverse Probability of Censoring Weighting (IPCW) in the calculation.

    Returns
    -------
    brier_score: float
        Value of the brier score.
    """
    if target_time is None:
        target_time = np.median(event_times)

    event_indicators = event_indicators.astype(bool, copy=False)
    event_before_or_at_target = (event_times <= target_time) & event_indicators
    event_free_at_target = (event_times > target_time) | (
        (event_times == target_time) & ~event_indicators
    )

    if ipcw:
        train_event_indicators = train_event_indicators.astype(bool, copy=False)
        inverse_train_event_indicators = ~train_event_indicators
        ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)

        ipc_pred = ipc_model.predict(event_times)
        # Catch if denominator is 0.
        ipc_pred[ipc_pred == 0] = np.inf
        # Category one calculates IPCW weight at observed time point.
        # Category one is individuals with event time lower than the time of interest and were NOT censored.
        weight_cat1 = event_before_or_at_target / ipc_pred
        # Defensively discard any undefined IPCW weights.
        weight_cat1[np.isnan(weight_cat1)] = 0
        # Category 2 is individuals whose time was greater than the time of interest (singleBrierTime)
        # contain both censored and uncensored individuals.
        ipc_target_pred = ipc_model.predict(target_time)
        if ipc_target_pred == 0:
            ipc_target_pred = np.inf
        weight_cat2 = event_free_at_target / ipc_target_pred
        # Defensively discard any undefined IPCW weights.
        weight_cat2[np.isnan(weight_cat2)] = 0
    else:
        weight_cat1 = event_before_or_at_target
        weight_cat2 = event_free_at_target

    sample_errors = (
        np.square(preds) * weight_cat1 + np.square(1 - preds) * weight_cat2
    )
    b_score = float(np.mean(sample_errors, dtype=float))
    ###########################
    # Here we are ordering event times and then using predict with level.chaos = 1 which returns
    # predictions ordered by time.
    # This is from Haider's code in R, but I feel it doesn't need to be ordered by time.
    # Refer above few lines for the justified code
    ###########################
    # order_of_times = np.argsort(event_times)
    # # Defensively discard any undefined IPCW weights.
    # weight_cat1 = ((event_times[order_of_times] <= target_time) & event_indicators[order_of_times]) /\
    #               ipc_model.predict(event_times[order_of_times])
    # weight_cat1[np.isnan(weight_cat1)] = 0
    # weight_cat2 = (event_times[order_of_times] > target_time) / ipc_model.predict(target_time)
    # weight_cat2[np.isnan(weight_cat2)] = 0
    #
    # survival_curves_ordered = survival_curves[order_of_times, :]
    # predict_probs = []
    # for i in range(survival_curves_ordered.shape[0]):
    #     predict_prob = predict_prob_from_curve(survival_curves_ordered[i, :], time_coordinates,
    #                                            event_times[order_of_times][i])
    #     predict_probs.append(predict_prob)
    # predict_probs = np.array(predict_probs)
    #
    # b_score = np.mean(np.square(predict_probs) * weight_cat1 + np.square(1 - predict_probs) * weight_cat2)
    return b_score


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
            left_probs = aft_model.predict_survival_function(
                x_df, times=left_limits
            ).values.diagonal()
            right_probs = aft_model.predict_survival_function(
                x_df, times=right_limits
            ).values.diagonal()
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
    """
    Calculate multiple Brier scores at multiple specific times.

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
        Values of multiple Brier scores.
    """
    if target_times.ndim != 1:
        error = "'time_grids' is not a one-dimensional array."
        raise TypeError(error)

    if event_times.ndim != 1:
        raise ValueError("event_times must be one-dimensional.")

    n_samples = event_times.shape[0]
    n_times = target_times.shape[0]
    if pred_mat.shape != (n_samples, n_times):
        raise ValueError(
            "pred_mat must have shape (n_samples, n_time_points) = "
            f"({n_samples}, {n_times}), got {pred_mat.shape}"
        )
    if event_indicators.shape != (n_samples,):
        raise ValueError("event_times and event_indicators must have the same length.")
    if ipcw and (train_event_times is None or train_event_indicators is None):
        raise ValueError(
            "Training event times and indicators must be provided for IPCW weighting."
        )

    target_times_row = target_times[None, :]
    event_times_column = event_times[:, None]
    event_indicators_column = event_indicators.astype(bool, copy=False)[:, None]

    if ipcw:
        censoring_indicators = ~train_event_indicators.astype(bool, copy=False)
        ipc_model = KaplanMeier(train_event_times, censoring_indicators)

        # G(T_i) is constant across target times, while G(t) is constant across
        # samples. Keep both as vectors and rely on broadcasting.
        event_censoring_survival = np.asarray(
            ipc_model.predict(event_times), dtype=float
        )
        target_censoring_survival = np.asarray(
            ipc_model.predict(target_times), dtype=float
        )
        event_censoring_survival[
            (event_censoring_survival == 0) | np.isnan(event_censoring_survival)
        ] = np.inf
        target_censoring_survival[
            (target_censoring_survival == 0) | np.isnan(target_censoring_survival)
        ] = np.inf

    # Accumulate the two mutually exclusive Brier components sequentially so
    # only one dense floating-point error matrix is live at a time.
    event_before_or_at_target = event_times_column <= target_times_row
    event_before_or_at_target &= event_indicators_column
    component_error = np.square(pred_mat, dtype=float)
    component_error *= event_before_or_at_target
    if ipcw:
        component_error /= event_censoring_survival[:, None]
    brier_sums = component_error.sum(axis=0)
    del component_error, event_before_or_at_target

    event_free_at_target = event_times_column > target_times_row
    censored_at_target = event_times_column == target_times_row
    censored_at_target &= ~event_indicators_column
    event_free_at_target |= censored_at_target
    del censored_at_target
    component_error = np.subtract(1.0, pred_mat, dtype=float)
    np.square(component_error, out=component_error)
    component_error *= event_free_at_target
    if ipcw:
        component_error /= target_censoring_survival[None, :]
    brier_sums += component_error.sum(axis=0)

    return brier_sums / n_samples


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

        left_sf = aft_model.predict_survival_function(x_df, times=left_limits)
        left_probs = left_sf.to_numpy().diagonal().copy()
        del left_sf
        right_sf = aft_model.predict_survival_function(x_df, times=right_limits)
        right_probs = right_sf.to_numpy().diagonal().copy()
        del right_sf
        target_probs_mat = aft_model.predict_survival_function(
            x_df, times=target_times
        ).to_numpy().T

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
