from typing import Optional

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
    target_time: Optional[float] = None,
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
        Binary indicators of censoring for the testing samples
    train_event_times: np.ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators: np.ndarray, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
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

    event_indicators = event_indicators.astype(bool)
    # train_event_indicators = train_event_indicators.astype(bool)

    if ipcw:
        inverse_train_event_indicators = 1 - train_event_indicators
        ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)

        ipc_pred = ipc_model.predict(event_times)
        # Catch if denominator is 0.
        ipc_pred[ipc_pred == 0] = np.inf
        # Category one calculates IPCW weight at observed time point.
        # Category one is individuals with event time lower than the time of interest and were NOT censored.
        weight_cat1 = ((event_times <= target_time) & event_indicators) / ipc_pred
        # Catch if event times goes over max training event time, i.e. predict gives NA
        weight_cat1[np.isnan(weight_cat1)] = 0
        # Category 2 is individuals whose time was greater than the time of interest (singleBrierTime)
        # contain both censored and uncensored individuals.
        weight_cat2 = (event_times > target_time) / ipc_model.predict(target_time)
        # predict returns NA if the passed-in time is greater than any of the times used to build the inverse probability
        # of censoring model.
        weight_cat2[np.isnan(weight_cat2)] = 0
    else:
        weight_cat1 = (event_times <= target_time) & event_indicators
        weight_cat2 = event_times > target_time

    b_score = (
        np.square(preds) * weight_cat1 + np.square(1 - preds) * weight_cat2
    ).mean()
    ###########################
    # Here we are ordering event times and then using predict with level.chaos = 1 which returns
    # predictions ordered by time.
    # This is from Haider's code in R, but I feel it doesn't need to be ordered by time.
    # Refer above few lines for the justified code
    ###########################
    # order_of_times = np.argsort(event_times)
    # # Catch if event times goes over max training event time, i.e. predict gives NA
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
    train_left_limits: Optional[np.ndarray] = None,
    train_right_limits: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    x_train: Optional[np.ndarray] = None,
    target_time: Optional[Numeric] = None,
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
    method: str, default: IPCW
        Method to use for handling censoring. One of ['uncensored', 'Tsouprou-marginal', 'Tsouprou-conditional'].
        'uncensored': Treat censored data as uncensored.
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

    if method == "uncensored":
        # if the target time is within the interval, then we calculate the brier score
        # otherwise, we skip the instance
        mask = (left_limits <= target_time) & (right_limits > target_time)
        weight = 1 - mask.astype(float)
        # get the survival status at the target time
        # if the left limit is greater than the target time, then the event has not occurred, so 1
        # if the right limit is less than or equal to the target time, then the event has occurred, so 0
        survival_status = (left_limits > target_time).astype(float)
        brier_score = (np.square(preds - survival_status) * weight).sum() / weight.sum()
    elif "Tsouprou" in method:
        # method based on Sofia Tsouprou's thesis
        # Measures of discrimination and predictive accuracy for interval censored survival data
        # https://studenttheses.universiteitleiden.nl/access/item:3597164/view
        # the original method uses Weibull parametric model to estimate the survival function
        # here we give the option to use either Turnbull estimator or Weibull AFT model

        # must have training data
        if train_left_limits is None or train_right_limits is None:
            raise ValueError("Training data must be provided for Tsouprou methods.")

        if method == "Tsouprou-marginal":
            marginal_estimator = TurnbullEstimatorLifelines(
                left=train_left_limits,
                right=train_right_limits,
            )
            # get the marginal survival probabilities at the target time
            left_probs = marginal_estimator.predict(left_limits)
            right_probs = marginal_estimator.predict(right_limits)
            target_probs = marginal_estimator.predict(target_time)
        elif method == "Tsouprou-conditional":
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
        # bad indices are those (1) the target time is within the interval and (2) the left and right survival
        # probabilities are the same, which leads to zeros in both numerator and denominator in survival_status
        bad = (
            (left_probs == right_probs)
            & (left_limits < target_time)
            & (target_time <= right_limits)
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
        survival_status[right_limits < target_time] = 0
        survival_status[left_limits >= target_time] = 1

        if np.any((survival_status < 0) | (survival_status > 1)):
            raise ValueError(
                "Calculated survival status contains values outside [0, 1]."
            )

        # calculate the brier score
        brier_score = np.mean(np.square(preds - survival_status))
    else:
        raise ValueError(f"Method {method} is not supported.")
    return brier_score


def brier_multiple_points(
    pred_mat: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: Optional[np.ndarray],
    train_event_indicators: Optional[np.ndarray],
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
        Binary indicators of censoring for the testing samples
    train_event_times: np.ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators: np.ndarray, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    target_times: float
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

    # bs_points_matrix = np.tile(event_times, (len(target_times), 1))
    target_times_mat = np.repeat(
        target_times.reshape(1, -1), repeats=len(event_times), axis=0
    )
    event_times_mat = np.repeat(
        event_times.reshape(-1, 1), repeats=len(target_times), axis=1
    )
    event_indicators_mat = np.repeat(
        event_indicators.reshape(-1, 1), repeats=len(target_times), axis=1
    )
    event_indicators_mat = event_indicators_mat.astype(bool)

    if ipcw:
        if train_event_times is None or train_event_indicators is None:
            raise ValueError(
                "Training event times and indicators must be provided for IPCW weighting."
            )

        inverse_train_event_indicators = 1 - train_event_indicators

        ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)

        # Category one calculates IPCW weight at observed time point.
        # Category one is individuals with event time lower than the time of interest and were NOT censored.
        ipc_pred = ipc_model.predict(event_times_mat)
        # Catch if denominator is 0.
        ipc_pred[ipc_pred == 0] = np.inf
        weight_cat1 = (
            (event_times_mat <= target_times_mat) & event_indicators_mat
        ) / ipc_pred
        # Catch if event times goes over max training event time, i.e. predict gives NA
        weight_cat1[np.isnan(weight_cat1)] = 0
        # Category 2 is individuals whose time was greater than the time of interest (singleBrierTime)
        # contain both censored and uncensored individuals.
        ipc_target_pred = ipc_model.predict(target_times_mat)
        # Catch if denominator is 0.
        ipc_target_pred[ipc_target_pred == 0] = np.inf
        weight_cat2 = (event_times_mat > target_times_mat) / ipc_target_pred
        # predict returns NA if the passed in time is greater than any of the times used to build
        # the inverse probability of censoring model.
        weight_cat2[np.isnan(weight_cat2)] = 0
    else:
        weight_cat1 = (event_times_mat <= target_times_mat) & event_indicators_mat
        weight_cat2 = event_times_mat > target_times_mat

    ipcw_square_error_mat = (
        np.square(pred_mat) * weight_cat1 + np.square(1 - pred_mat) * weight_cat2
    )
    brier_scores = np.mean(ipcw_square_error_mat, axis=0)
    return brier_scores


def brier_multiple_points_ic(
    pred_mat: np.ndarray,
    left_limits: np.ndarray,
    right_limits: np.ndarray,
    target_times: np.ndarray,
    train_left_limits: Optional[np.ndarray] = None,
    train_right_limits: Optional[np.ndarray] = None,
    x: Optional[np.ndarray] = None,
    x_train: Optional[np.ndarray] = None,
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

    n_samples = left_limits.shape[0]
    n_times = target_times.shape[0]

    if pred_mat.shape != (n_samples, n_times):
        raise ValueError(
            f"pred_mat must have shape (n_samples, n_time_points) = "
            f"({n_samples}, {n_times}), got {pred_mat.shape}"
        )

    # Broadcast helpers
    left_mat = np.repeat(
        left_limits.reshape(-1, 1), n_times, axis=1
    )  # (n_samples, n_times)
    right_mat = np.repeat(
        right_limits.reshape(-1, 1), n_times, axis=1
    )  # (n_samples, n_times)
    time_mat = np.repeat(
        target_times.reshape(1, -1), n_samples, axis=0
    )  # (n_samples, n_times)

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

        ambiguous_mask = (time_mat >= left_mat) & (time_mat < right_mat)
        usable_mask = ~ambiguous_mask  # (n_samples, n_times)

        # survival_status default to 0 then overwrite alive cases:
        survival_status_mat = np.zeros_like(pred_mat, dtype=float)
        # alive if t_j < L_i
        alive_mask = time_mat < left_mat
        survival_status_mat[alive_mask] = 1.0
        # dead if t_j >= R_i and not inf
        # (already 0 there, so nothing to set)

        # squared error
        sqerr = np.square(pred_mat - survival_status_mat)

        # masked mean per time j
        # denominator per column j = number of usable samples
        usable_counts = usable_mask.sum(axis=0).astype(float)  # (n_times,)

        # avoid divide-by-zero: if no usable samples at a time point, set Brier to np.nan
        brier_scores = np.full(n_times, np.nan, dtype=float)
        for j in range(n_times):
            if usable_counts[j] > 0:
                brier_scores[j] = (sqerr[usable_mask[:, j], j]).mean()

        return brier_scores

    # ============================================================
    # Case 2/3: Tsouprou-based, which creates fractional "status"
    # ============================================================
    if "Tsouprou" in method:
        if train_left_limits is None or train_right_limits is None:
            raise ValueError("Training data must be provided for Tsouprou methods.")

        # --------------------------------------------------------
        # 2A. Fit marginal or conditional model on training data
        # --------------------------------------------------------
        if method == "Tsouprou-marginal":
            marginal_estimator = TurnbullEstimatorLifelines(
                left=train_left_limits,
                right=train_right_limits,
            )

            # Per-sample probs at L_i and R_i
            left_probs = marginal_estimator.predict(left_limits)  # (n_samples,)
            right_probs = marginal_estimator.predict(right_limits)  # (n_samples,)

            # Per-time probs S(t_j). Same for every sample.
            target_probs_vec = marginal_estimator.predict(target_times)  # (n_times,)

            # Broadcast:
            left_probs_mat = np.repeat(left_probs.reshape(-1, 1), n_times, axis=1)
            right_probs_mat = np.repeat(right_probs.reshape(-1, 1), n_times, axis=1)
            target_probs_mat = np.repeat(
                target_probs_vec.reshape(1, -1), n_samples, axis=0
            )

        elif method == "Tsouprou-conditional":
            if x is None or x_train is None:
                raise ValueError(
                    "x and x_train must be provided for Tsouprou-conditional."
                )

            # build train_df
            train_data = {"left": train_left_limits, "right": train_right_limits}
            if x_train.ndim == 1:
                train_data["feature"] = x_train
            elif x_train.ndim == 2:
                for k in range(x_train.shape[1]):
                    train_data[f"feature_{k}"] = x_train[:, k]
            else:
                raise ValueError("x_train must be a 1-D or 2-D array.")
            train_df = pd.DataFrame(train_data)

            # build x_df for test
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

            # left_probs[i]   = S_i(L_i)
            # right_probs[i]  = S_i(R_i)
            # We'll grab these by diagonal extraction, same as single-time code:
            left_sf_df = aft_model.predict_survival_function(
                x_df, times=left_limits
            )  # rows=times, cols=samples
            right_sf_df = aft_model.predict_survival_function(x_df, times=right_limits)
            left_probs = left_sf_df.values.diagonal()  # (n_samples,)
            right_probs = right_sf_df.values.diagonal()  # (n_samples,)

            # target_probs_mat[i,j] = S_i(t_j)
            full_sf_df = aft_model.predict_survival_function(x_df, times=target_times)
            # full_sf_df: rows are t_j, cols are samples i
            target_probs_mat = full_sf_df.T.values  # shape (n_samples, n_times)

            # broadcast left/right
            left_probs_mat = np.repeat(left_probs.reshape(-1, 1), n_times, axis=1)
            right_probs_mat = np.repeat(right_probs.reshape(-1, 1), n_times, axis=1)

        else:
            raise ValueError(f"Method {method} is not supported.")

        # --------------------------------------------------------
        # 2B. Build Y_mat (fractional survival status) for every (i,j)
        # --------------------------------------------------------
        # We'll follow your single-time logic:
        # survival_status = (S(t) - S(R)) / (S(L) - S(R))
        # Then override based on position of t relative to [L,R].
        #
        # Edge cases:
        # - If denominator is 0 *and* t is strictly inside (L,R], we can't define status. We'll drop those cells.
        # - After override:
        #       if t_j > R_i  -> 0
        #       if t_j <= L_i -> 1
        # This guarantees values in [0,1].

        denom = left_probs_mat - right_probs_mat

        with np.errstate(divide="ignore", invalid="ignore"):
            survival_status_mat = (target_probs_mat - right_probs_mat) / denom

        # Apply boundary overrides
        after_right_mask = time_mat > right_mat
        before_left_mask = time_mat <= left_mat
        survival_status_mat[after_right_mask] = 0.0
        survival_status_mat[before_left_mask] = 1.0

        # Identify "bad" cells:
        # bad if denom == 0 AND t_j is within (L_i, R_i] (i.e. genuinely ambiguous)
        inside_mask = (time_mat > left_mat) & (time_mat <= right_mat)
        bad_mask = (denom == 0.0) & inside_mask

        # sanity check: any out-of-range due to numerical issues?
        oob_mask = (survival_status_mat < 0.0) | (survival_status_mat > 1.0)
        if np.any(oob_mask & ~bad_mask):
            raise ValueError(
                "Calculated survival status contains values outside [0,1] for some non-bad entries."
            )

        # We'll exclude bad_mask entries from the averaging for their column.
        usable_mask = ~bad_mask

        # --------------------------------------------------------
        # 2C. Compute squared error and average per time
        # --------------------------------------------------------
        sqerr = np.square(pred_mat - survival_status_mat)

        brier_scores = np.full(n_times, np.nan, dtype=float)
        usable_counts = usable_mask.sum(axis=0).astype(float)

        for j in range(n_times):
            if usable_counts[j] > 0:
                brier_scores[j] = (sqerr[usable_mask[:, j], j]).mean()

        return brier_scores

    raise ValueError(f"Method {method} is not supported.")
