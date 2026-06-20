from __future__ import annotations

from typing import Optional

import numpy as np
from tqdm import trange

from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier, KaplanMeierArea, km_mean
from SurvivalEVAL.Evaluations.util import predict_rmst


def mean_error(
    predicted_times: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: Optional[np.ndarray] = None,
    train_event_indicators: Optional[np.ndarray] = None,
    error_type: str = "absolute",
    method: str = "Hinge",
    weighted: bool = True,
    log_scale: bool = False,
    verbose: bool = False,
    truncation_time=None,
) -> float:
    """
    Calculate the mean absolute/squared error score for the predicted survival times.

    Parameters
    ----------
    predicted_times: np.ndarray, shape = (n_samples, )
        Predicted survival times for the testing samples
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
    error_type: string, default: "absolute"
        Type of mean error to use. Options are "absolute" and "squared".
    method: string, default: "Hinge"
        Method of handling censorship.
        Options are "Uncensored", "Hinge", "Margin", "IPCW-T", "IPCW-D", and "Pseudo_obs".
    weighted: boolean, default: True
        Whether to use weighting scheme for MAE.
        If true, each censored sample has confidence weight
        1 - S_KM(censoring time).
    log_scale: boolean, default: False
        Whether to use log scale for the loss function.
    verbose: boolean, default: False
        Whether to show the progress bar.
    truncated_time: float, default: None
        the truncated time.

    Returns
    -------
    Value for the calculated mean error.
    """
    event_indicators = event_indicators.astype(bool)
    n_test = event_times.size
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)

    error_type = error_type.lower()
    method = method.lower()

    # calculate the weighting for each sample
    if method in ["margin", "ipcw-t", "ipcw-d", "pseudo_obs"]:
        if train_event_times is None or train_event_indicators is None:
            raise ValueError(
                "If method is '{}', training set values must be included.".format(
                    method
                )
            )

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = km_model.km_linear_zero
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        # predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        censor_times = event_times[~event_indicators]
        weights = np.ones(n_test)
        if weighted:
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

    # set the error function
    if error_type == "absolute":
        error_func = np.abs
    elif error_type == "squared":
        error_func = np.square
    else:
        raise TypeError(
            "Please enter one of 'absolute' or 'squared' for calculating error."
        )

    if method == "uncensored":
        # only use uncensored data
        if log_scale:
            errors = np.log(event_times[event_indicators]) - np.log(
                predicted_times[event_indicators]
            )
        else:
            errors = event_times[event_indicators] - predicted_times[event_indicators]
        return error_func(errors).mean()
    elif method == "hinge":
        # consider only the early prediction error
        # if prediction is higher than the censored time, it is not penalized
        weights = np.ones(predicted_times.size)
        if weighted:
            if train_event_times is None or train_event_indicators is None:
                raise ValueError(
                    "If 'weighted' is True for calculating Hinge, training set values must be included."
                )
            km_model = KaplanMeierArea(train_event_times, train_event_indicators)
            censor_times = event_times[~event_indicators]
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

        if log_scale:
            errors = np.log(event_times) - np.log(predicted_times)
        else:
            errors = event_times - predicted_times
        errors[~event_indicators] = np.maximum(errors[~event_indicators], 0)
        return np.average(error_func(errors), weights=weights)
    elif method == "margin":
        # The L1-margin method proposed by https://www.jmlr.org/papers/v21/18-772.html
        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        best_guesses = km_model.best_guess(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[
            censor_times > km_linear_zero
        ]

        if truncation_time:
            best_guesses = np.clip(best_guesses, a_max=truncation_time, a_min=None)
            predicted_times = np.clip(predicted_times, a_max=truncation_time, a_min=None)
            event_times = np.clip(event_times, a_max=truncation_time, a_min=None)

        errors = np.empty(predicted_times.size)
        if log_scale:
            errors[event_indicators] = np.log(event_times[event_indicators]) - np.log(
                predicted_times[event_indicators]
            )
            errors[~event_indicators] = np.log(best_guesses) - np.log(
                predicted_times[~event_indicators]
            )
        else:
            errors[event_indicators] = (
                event_times[event_indicators] - predicted_times[event_indicators]
            )
            errors[~event_indicators] = (
                best_guesses - predicted_times[~event_indicators]
            )
        return np.average(error_func(errors), weights=weights)
    elif method == "ipcw-t":
        # This is the IPCW-T method from https://arxiv.org/pdf/2306.01196.pdf
        # Calculate the best guess time (surrogate time) based on the subsequent uncensored subjects
        best_guesses = np.empty(shape=n_test)
        for i in range(n_test):
            if event_indicators[i] == 1:
                best_guesses[i] = event_times[i]
            else:
                # Numpy will throw a warning if afterward_event_times are all false. TODO: consider change the code.
                afterward_event_idx = (
                    train_event_times[train_event_indicators == 1] > event_times[i]
                )
                best_guesses[i] = np.mean(
                    train_event_times[train_event_indicators == 1][afterward_event_idx]
                )
        # NaN values are generated because there are no events after the censor times
        nan_idx = np.argwhere(np.isnan(best_guesses))
        predicted_times = np.delete(predicted_times, nan_idx)
        best_guesses = np.delete(best_guesses, nan_idx)
        weights = np.delete(weights, nan_idx)

        if truncation_time:
            best_guesses = np.clip(best_guesses, a_max=truncation_time, a_min=None)
            predicted_times = np.clip(predicted_times, a_max=truncation_time, a_min=None)

        if log_scale:
            errors = np.log(best_guesses) - np.log(predicted_times)
        else:
            errors = best_guesses - predicted_times
        return np.average(error_func(errors), weights=weights)
    elif method == "ipcw-d":
        # This is the IPCW-D method from https://arxiv.org/pdf/2306.01196.pdf
        # Using IPCW weights to transfer the censored subjects to uncensored subjects
        inverse_train_event_indicators = 1 - train_event_indicators

        ipc_model = KaplanMeierArea(train_event_times, inverse_train_event_indicators)
        ipc_pred = ipc_model.predict(event_times)
        # Catch if denominator is 0. This happens when the time is later than the last event time in trainset.
        ipc_pred[ipc_pred == 0] = np.inf

        if truncation_time:
            event_times = np.clip(event_times, a_max=truncation_time, a_min=None)
            predicted_times = np.clip(predicted_times, a_max=truncation_time, a_min=None)

        if log_scale:
            errors = np.log(event_times) - np.log(predicted_times)
        else:
            errors = event_times - predicted_times
        return (
            error_func(errors)[event_indicators] / ipc_pred[event_indicators]
        ).mean()
    elif method == "pseudo_obs":
        # Calculate the best guess time (surrogate time) by the contribution of the censored subjects to KM curve
        n_train = train_event_times.size

        events, population_counts = (
            km_model.events.copy(),
            km_model.population_count.copy(),
        )
        times = km_model.survival_times.copy()
        probs = km_model.survival_probabilities.copy()
        # get the discrete time points where the event happens, then calculate the area under those discrete time only
        # this doesn't make any difference for step function, but it does for trapezoid rule.
        unique_idx = np.where(events != 0)[0]
        if unique_idx[-1] != len(events) - 1:
            unique_idx = np.append(unique_idx, len(events) - 1)
        times = times[unique_idx]
        population_counts = population_counts[unique_idx]
        events = events[unique_idx]
        probs = probs[unique_idx]
        sub_expect_time = km_mean(times.copy(), probs.copy())

        # use the idea of dynamic programming to calculate the multiplier of the KM estimator in advances.
        # if we add a new time point to the KM curve, the multiplier before the new time point will be
        # 1 - event_counts / (population_counts + 1), and the multiplier after the new time point will be
        # the same as before.
        multiplier = 1 - events / population_counts
        multiplier_total = 1 - events / (population_counts + 1)
        best_guesses = event_times.copy().astype(float)

        for i in trange(
            n_test, desc="Calculating surrogate times for Pseudo_obs", disable=not verbose
        ):
            if event_indicators[i] != 1:
                total_multiplier = multiplier.copy()
                insert_index = np.searchsorted(times, event_times[i], side="right")
                total_multiplier[:insert_index] = multiplier_total[:insert_index]
                survival_probabilities = np.cumprod(total_multiplier)
                if insert_index == len(times):
                    times_addition = np.append(times, event_times[i])
                    survival_probabilities_addition = np.append(
                        survival_probabilities, survival_probabilities[-1]
                    )
                    total_expect_time = km_mean(
                        times_addition, survival_probabilities_addition
                    )
                else:
                    total_expect_time = km_mean(times, survival_probabilities)
                best_guesses[i] = (
                    n_train + 1
                ) * total_expect_time - n_train * sub_expect_time

        if truncation_time:
            best_guesses = np.clip(best_guesses, a_max=truncation_time, a_min=None)
            predicted_times = np.clip(predicted_times, a_max=truncation_time, a_min=None)

        if log_scale:
            errors = np.log(best_guesses) - np.log(predicted_times)
        else:
            errors = best_guesses - predicted_times
        return np.average(error_func(errors), weights=weights)
    else:
        raise ValueError(
            "Method must be one of 'Uncensored', 'Hinge', 'Margin', 'IPCW-T', 'IPCW-D' "
            "or 'Pseudo_obs'. Got '{}' instead.".format(method)
        )


def mean_error_truncated(
        pred_rmst: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        truncation_time: float = None,
        error_type: str = "squared",
        method: str = "Pseudo_obs",
        log_scale: bool = False,
        interpolation: str = "None",
        verbose: bool = False,
) -> float:
    """
    Truncated mean error.
    It is calculated as the mean error between the predicted RMST and the true RMST up to the truncation time.
    The true RMST is calculated using Pseudo-observation method.

    Parameters
    ----------
    pred_rmst: np.ndarray, (n_samples,)
        Predicted RMST values for each sample.
    event_times: np.ndarray, (n_samples,)
        Actual event/censor time for each sample.
    event_indicators: np.ndarray, (n_samples,)
        Binary indicators of censoring for each sample.
    train_event_times: np.ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators: np.ndarray, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples.
    truncation_time: float = None
        The truncation time up to which RMST is calculated.
        If None, use the maximum event time in the training set.
    error_type: string, default: "absolute"
        Type of mean error to use. Options are "absolute" and "squared".
    method: string, default: "Pseudo_obs"
        Method of handling censorship. Only "Pseudo_obs" is supported for truncated mean error for now.
    log_scale: boolean, default: False
        Whether to use log scale for the mean error.
    interpolation: str, default: "None"
        The monotonic cubic interpolation method for RMST calculation. One of ['None', 'Linear', 'Pchip']. Default: 'None'.
        If 'None', no interpolation is applied. Use step function to calculate the area under the curve.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.
    verbose: boolean, default: False
        Whether to print detailed logs during the computation.

    Returns
    -------
    mean_error: float
        Mean error value.
    """
    event_indicators = event_indicators.astype(bool)
    n_test = event_times.size
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)
        n_train = train_event_times.size

    if truncation_time is None:
        truncation_time = max(train_event_times)
    assert 0 < truncation_time <= max(train_event_times), "truncation_time must be in (0, max training event time]"

    # set the error function
    if error_type == "absolute":
        error_func = np.abs
    elif error_type == "squared":
        error_func = np.square
    else:
        raise TypeError(
            "Please enter one of 'absolute' or 'squared' for calculating error."
        )
    
    # this is more like the estimated "true" RMST for each test sample
    true_rmst = np.empty(shape=n_test)

    if method in ["Pseudo_obs"]:
        if train_event_times is None or train_event_indicators is None:
            raise ValueError(
                "If method is '{}', training set values must be included.".format(
                    method
                )
            )

        km_model = KaplanMeier(train_event_times, train_event_indicators)

        event_counts, population_counts = (
            km_model.events.copy(),
            km_model.population_count.copy(),
        )
        times = km_model.survival_times.copy()
        probs = km_model.survival_probabilities.copy()
        
        # if the first time point is not 0, we need to add a time point at 0 with survival probability 1
        if times[0] > 0:
            times = np.insert(times, 0, 0.0)
            probs = np.insert(probs, 0, 1.0)
            population_counts = np.insert(population_counts, 0, population_counts[0])
            event_counts = np.insert(event_counts, 0, 0)

        # if truncation_time < times[-1], we need to stop at truncation_time, so that the last time point is truncation_time (and two arrays stop at the truncation time)
        if truncation_time < times[-1]:
            # index of the last time <= truncation_time
            trunc_idx = np.searchsorted(times, truncation_time, side="right")
            if np.isclose(times[trunc_idx - 1], truncation_time):
                times = times[:trunc_idx]
                probs = probs[:trunc_idx]
                population_counts = population_counts[:trunc_idx]
                event_counts = event_counts[:trunc_idx]
            else:   
                times = np.insert(times[:trunc_idx], trunc_idx, truncation_time)
                probs = np.insert(probs[:trunc_idx], trunc_idx, probs[trunc_idx - 1])
                population_counts = np.insert(
                    population_counts[:trunc_idx], 
                    trunc_idx,
                    population_counts[trunc_idx - 1]
                )
                event_counts = np.insert(event_counts[:trunc_idx], trunc_idx, 0)
        sub_rmst = predict_rmst(probs.copy(), times.copy(), interpolation=interpolation)

        # use the idea of dynamic programming to calculate the multiplier of the KM estimator in advances.
        # if we add a new time point to the KM curve, 
        # the multiplier before the new time point will be 1 - event_counts / (population_counts + 1), 
        # the multiplier at the new time point will be 1 - 1 / population_counts[insert_index - 1],
        # and the multiplier after the new time point will be the same as before.
        multiplier = 1 - event_counts / population_counts
        multiplier_total = 1 - event_counts / (population_counts + 1)

        for i in trange(
            n_test, desc="Calculating surrogate 'true' RMST for Pseudo_obs", disable=not verbose
        ):
            total_multiplier = multiplier.copy()
            ti = event_times[i]
            ei = event_indicators[i]
            insert_idx = np.searchsorted(times, ti, side="right")
            total_multiplier[:insert_idx] = multiplier_total[:insert_idx]

            # if the event time is before truncation time, we need to add the new time point to the KM curve
            has_existing_time = (insert_idx > 0 and np.isclose(times[insert_idx - 1], ti))

            if has_existing_time:
                # case 1: event time already exists in the grid
                total_times = times.copy()
                insert_idx -= 1   # index of the existing time ti, because we use side="right" in searchsorted
                
                n_events = event_counts[insert_idx] + (1 if ei else 0)
                # Risk set at this time gets +1 subject
                denom = population_counts[insert_idx] + 1
                total_multiplier[insert_idx] = 1 - n_events / denom                
            elif insert_idx < len(times):
                # Case 2: new time point strictly inside the existing time range
                # (t is strictly between times[insert_index-1] and times[insert_index])
                total_times = np.insert(times, insert_idx, ti)
                
                n_events = 1 if ei else 0
                denom = population_counts[insert_idx] + 1
                new_factor = 1 - n_events / denom
                total_multiplier = np.insert(total_multiplier, insert_idx, new_factor)
                # total_multiplier = np.insert(total_multiplier, insert_index, 1 - n_events / (population_counts[insert_index] + 1))
            else:
                # Case 3: insert_index == len(times)
                if np.isclose(ti, times[-1]):
                    # Treat as an event at the last existing time
                    total_times = times.copy()
                    idx = len(times) - 1
                    n_events = event_counts[idx] + (1 if ei else 0)
                    denom = population_counts[idx] + 1
                    total_multiplier[idx] = 1 - n_events / denom
                else:
                    # ignoring these events > truncation_time
                    total_times = times.copy()

            total_surv_prob = np.cumprod(total_multiplier)
            total_rmst = predict_rmst(total_surv_prob, total_times, interpolation=interpolation)
            true_rmst[i] = (n_train + 1) * total_rmst - n_train * sub_rmst
    else:
        raise ValueError(
            "Method must be 'Pseudo_obs' for truncated mean error. "
            "Got '{}' instead.".format(method)
        )
    
    # For experimental purpose, I want to print if the surrogate RMST is outside the possible range [0, truncation_time], print the index and the values.
    # TODO: get a sense of how often this happens in real data. And decide whether we need to handle this case by clipping the values.
    # THIS MUST BE REMOVED BEFORE THE FINAL PUBLIC RELEASE.
    # if true_rmst.min() < 0 or true_rmst.max() > truncation_time:
    #     invalid_idx = np.where((true_rmst < 0) | (true_rmst > truncation_time))[0]
    #     for trunc_idx in invalid_idx:
    #         print(f"Warning: Surrogate RMST out of bounds at index {trunc_idx}: true RMST = {true_rmst[trunc_idx]}, truncation_time = {truncation_time}")
    true_rmst = np.clip(true_rmst, a_min=0.0, a_max=truncation_time)

    if log_scale:
        errors = np.log(true_rmst) - np.log(pred_rmst)
    else:
        errors = true_rmst - pred_rmst

    return float(np.mean(error_func(errors)))

def mean_error_truncated_slow(
        pred_rmst: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        truncation_time: float = None,
        error_type: str = "squared",
        method: str = "Pseudo_obs",
        log_scale: bool = False,
        interpolation: str = "None",
        verbose: bool = False,
) -> float:
    """
    Truncated mean error.
    It is calculated as the mean error between the predicted RMST and the true RMST up to the truncation time.
    The true RMST is calculated using Pseudo-observation method.

    Parameters
    ----------
    pred_rmst: np.ndarray, (n_samples,)
        Predicted RMST values for each sample.
    event_times: np.ndarray, (n_samples,)
        Actual event/censor time for each sample.
    event_indicators: np.ndarray, (n_samples,)
        Binary indicators of censoring for each sample.
    train_event_times: np.ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators: np.ndarray, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples.
    truncation_time: float = None
        The truncation time up to which RMST is calculated.
        If None, use the maximum event time in the training set.
    error_type: string, default: "absolute"
        Type of mean error to use. Options are "absolute" and "squared".
    method: string, default: "Pseudo_obs"
        Method of handling censorship. Only "Pseudo_obs" is supported for truncated mean error for now.
    log_scale: boolean, default: False
        Whether to use log scale for the mean error.
    interpolation: str, default: "None"
        The monotonic cubic interpolation method for RMST calculation. One of ['None', 'Linear', 'Pchip']. Default: 'None'.
        If 'None', no interpolation is applied. Use step function to calculate the area under the curve.
        If 'Linear', use the interp1d method from scipy.interpolate.
        If 'Pchip', use the PchipInterpolator from scipy.interpolate.
    verbose: boolean, default: False
        Whether to print detailed logs during the computation.

    Returns
    -------
    mean_error: float
        Mean error value.
    """
    event_indicators = event_indicators.astype(bool)
    n_test = event_times.size
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)
        n_train = train_event_times.size

    if truncation_time is None:
        truncation_time = max(train_event_times)

    # set the error function
    if error_type == "absolute":
        error_func = np.abs
    elif error_type == "squared":
        error_func = np.square
    else:
        raise TypeError(
            "Please enter one of 'absolute' or 'squared' for calculating error."
        )
    
    # this is more like the estimated "true" RMST for each test sample
    true_rmst = np.empty(shape=n_test)

    if method in ["Pseudo_obs"]:
        if train_event_times is None or train_event_indicators is None:
            raise ValueError(
                "If method is '{}', training set values must be included.".format(
                    method
                )
            )

        km_model = KaplanMeier(train_event_times, train_event_indicators)

        times = km_model.survival_times.copy()
        probs = km_model.survival_probabilities.copy()
        
        # if the first time point is not 0, we need to add a time point at 0 with survival probability 1
        if times[0] > 0:
            times = np.insert(times, 0, 0.0)
            probs = np.insert(probs, 0, 1.0)

        # if truncation_time < times[-1], we need to stop at truncation_time, so that the last time point is truncation_time (and two arrays stop at the truncation time)
        if truncation_time < times[-1]:
            trunc_idx = np.searchsorted(times, truncation_time, side="right")
            if np.isclose(times[trunc_idx - 1], truncation_time):
                times = times[:trunc_idx]
                probs = probs[:trunc_idx]
            else:   
                times = np.insert(times[:trunc_idx], trunc_idx, truncation_time)
                probs = np.insert(probs[:trunc_idx], trunc_idx, probs[trunc_idx - 1])
        sub_rmst = predict_rmst(probs.copy(), times.copy(), interpolation=interpolation)

        for i in trange(
            n_test, desc="Calculating surrogate 'true' RMST for Pseudo_obs", disable=not verbose
        ):
            added_event_times = np.append(train_event_times, event_times[i])
            added_event_indicators = np.append(train_event_indicators, event_indicators[i])
            total_km_model = KaplanMeier(added_event_times, added_event_indicators)
            total_surv_prob = total_km_model.survival_probabilities.copy()
            total_times = total_km_model.survival_times.copy()

            if total_times[0] > 0:
                total_times = np.insert(total_times, 0, 0.0)
                total_surv_prob = np.insert(total_surv_prob, 0, 1.0)
            
            if truncation_time < total_times[-1]:
                trunc_idx = np.searchsorted(total_times, truncation_time, side="right")
                if np.isclose(total_times[trunc_idx - 1], truncation_time):
                    total_times = total_times[:trunc_idx]
                    total_surv_prob = total_surv_prob[:trunc_idx]
                else:
                    total_times = np.insert(total_times[:trunc_idx], trunc_idx, truncation_time)
                    total_surv_prob = np.insert(total_surv_prob[:trunc_idx], trunc_idx, total_surv_prob[trunc_idx - 1])
            total_rmst = predict_rmst(total_surv_prob, total_times, interpolation=interpolation)
            true_rmst[i] = (n_train + 1) * total_rmst - n_train * sub_rmst
    else:
        raise ValueError(
            "Method must be 'Pseudo_obs' for truncated mean error. "
            "Got '{}' instead.".format(method)
        )
    
    # For experimental purpose, I want to print if the surrogate RMST is outside the possible range [0, truncation_time], print the index and the values.
    # TODO: get a sense of how often this happens in real data. And decide whether we need to handle this case by clipping the values.
    # THIS MUST BE REMOVED BEFORE THE FINAL PUBLIC RELEASE.
    # if true_rmst.min() < 0 or true_rmst.max() > truncation_time:
    #     invalid_idx = np.where((true_rmst < 0) | (true_rmst > truncation_time))[0]
    #     for idx in invalid_idx:
    #         print(f"Warning: Surrogate RMST out of bounds at index {idx}: true RMST = {true_rmst[idx]}, truncation_time = {truncation_time}")
    true_rmst = np.clip(true_rmst, a_min=0.0, a_max=truncation_time)

    if log_scale:
        errors = np.log(true_rmst) - np.log(pred_rmst)
    else:
        errors = true_rmst - pred_rmst

    return float(np.mean(error_func(errors)))



def _prepare_interval_arrays(
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    predicted_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = np.asarray(left_bounds, float)
    R = np.asarray(right_bounds, float)
    t_hat = np.asarray(predicted_times, float)

    assert L.shape == R.shape == t_hat.shape, "shape mismatch"
    return L, R, t_hat


def _compute_inside_mask(
    left: np.ndarray,
    right: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mask for predictions inside their intervals. Also return mask for right-censored intervals.
    Parameters

    Inside test:
    - general finite-interval: (L, R]  -> (t_hat > L) & (t_hat <= R)
    - right-censored:          (L, +inf) -> t_hat > L
    - exact-interval (L==R): treat as {R}: |t_hat - R| <= atol
    ----------
    left: np.ndarray, (n_samples,)
        Left limits of the interval-censored data.
    right: np.ndarray, (n_samples,)
        Right limits of the interval-censored data (use np.inf for right-censor).
    predicted: np.ndarray, (n_samples,)
        Your point predictions (e.g., median predicted times, mean predicted times, etc).

    Returns
    -------
    inside: np.ndarray, (n_samples,)
        Boolean mask for predictions inside their intervals.
    is_right_cens: np.ndarray, (n_samples,)
        Boolean mask for right-censored intervals.
    """
    is_right_cens = np.isinf(right)
    is_exact = (~is_right_cens) & (left == right)
    inside_exact = is_exact & np.isclose(predicted, right)
    inside_finite = (
        (~is_right_cens) & (~is_exact) & (predicted > left) & (predicted <= right)
    )
    inside_right = is_right_cens & (predicted > left)
    inside = inside_exact | inside_finite | inside_right
    return inside, is_right_cens


def inclusion_rate(
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    predicted_times: np.ndarray,
) -> float:
    """
    Inclusion rate for interval-censored data: proportion of predictions inside their intervals.

    This is similar to p_out but reports the complement (1 - p_out).
    p_out is proposed in [1].

    Parameters
    ----------
    left_bounds: np.ndarray, (n_samples,)
        Left limits of the interval-censored data.
    right_bounds: np.ndarray, (n_samples,)
        Right limits of the interval-censored data (use np.inf for right-censor).
    predicted_times: np.ndarray, (n_samples,)
        Your point predictions (e.g., median predicted times, mean predicted times, etc).

    Returns
    -------
    inclusion_rate: float
        Proportion of samples with predicted times inside their intervals.
    References
    ----------
    [1] Avati et al., "Countdown regression: sharp and calibrated survival predictions", UAI 2020.
    """
    L, R, t_hat = _prepare_interval_arrays(left_bounds, right_bounds, predicted_times)
    inside, _ = _compute_inside_mask(L, R, t_hat)
    return float(np.mean(inside))


def mean_error_ic(
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    predicted_times: np.ndarray,
    error_type: str = "absolute",
    log_scale: bool = False,
) -> float:
    """
    Mean error for interval-censored data. It is a one-sided mean error that only penalizes predictions
    outside the interval-censored data.
    This is proposed in [1] as the hinge loss for interval-censored data.
    This is later reporposed by [2] as the `d_out` metric.

    Parameters
    ----------
    left_bounds: np.ndarray, (n_samples,)
        Left limits of the interval-censored data.
    right_bounds: np.ndarray, (n_samples,)
        Right limits of the interval-censored data (use np.inf for right-censor).
    predicted_times: np.ndarray, (n_samples,)
        Your point predictions (e.g., median predicted times, mean predicted times, etc).
    error_type: string, default: "absolute"
        Type of mean error to use. Options are "absolute" and "squared".
    log_scale: boolean, default: False
        Whether to use log scale for the loss function.

    Returns
    -------
    mean_error: float
        Mean error value.

    References
    ----------
    [1] Shivaswamy et al., "A support vector approach to censored targets", ICDM 2007.
    [2] Avati et al., "Countdown regression: sharp and calibrated survival predictions", UAI 2020.

    """
    L, R, t_hat = _prepare_interval_arrays(left_bounds, right_bounds, predicted_times)

    # set the error function
    error_type = error_type.lower()
    if error_type == "absolute":
        error_func = np.abs
    elif error_type == "squared":
        error_func = np.square
    else:
        raise TypeError(
            "Please enter one of 'absolute' or 'squared' for calculating error."
        )

    # TODO: We need to move the logarithm transformation later after we calculate a best-guess.
    if log_scale:
        t_hat = np.log(t_hat)
        L = np.log(L)
        R = np.log(R)

    inside, is_right_cens = _compute_inside_mask(L, R, t_hat)
    outside = ~inside
    error_to_L = error_func(t_hat - L)
    error_to_R = np.where(is_right_cens, np.inf, error_func(t_hat - R))
    d_i = np.where(outside, np.minimum(error_to_L, error_to_R), 0.0)

    return float(np.mean(d_i))


if __name__ == "__main__":
    # Test the functions
    train_t = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            74,
            75,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            117,
            118,
            119,
            120,
            120,
            120,
            121,
            121,
            124,
            125,
            126,
            127,
            128,
            129,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            155,
            156,
            157,
            158,
            159,
            161,
            182,
            183,
            186,
            190,
            191,
            192,
            192,
            192,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            202,
            203,
            204,
            202,
            203,
            204,
            212,
            213,
            214,
            215,
            216,
            217,
            222,
            223,
            224,
        ]
    )
    train_e = np.array(
        [
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    train_t = np.array([20, 97, 150, 200, 210])
    train_e = np.array([1, 1, 0, 1, 1])
    t = np.array(
        [
            5,
            10,
            19,
            31,
            43,
            59,
            63,
            75,
            97,
            113,
            134,
            150,
            163,
            176,
            182,
            195,
            200,
            210,
            220,
        ]
    )
    e = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0])
    predict_time = np.array(
        [
            18,
            19,
            5,
            12,
            75,
            100,
            120,
            85,
            36,
            95,
            170,
            41,
            200,
            210,
            260,
            86,
            100,
            120,
            140,
        ]
    )
    # "Margin", "IPCW-T", "IPCW-D", "Pseudo_obs"
    # score = mean_error(
    #     predict_time,
    #     t,
    #     e,
    #     train_t,
    #     train_e,
    #     method="Pseudo_obs",
    #     verbose=True,
    #     truncation_time=100,
    # )
    score_1 = mean_error_truncated(
    predict_time,
    t,
    e,
    train_t,
    train_e,
    truncation_time= 200,
    method="Pseudo_obs",
    verbose=True,
)
    score_2 = mean_error_truncated_slow(
    predict_time,
    t,
    e,
    train_t,
    train_e,
    truncation_time= 200,
    method="Pseudo_obs",
    verbose=True,
)
    print(np.sqrt(score_1))
    print(np.sqrt(score_2))