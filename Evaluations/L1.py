import numpy as np
import pandas as pd
from typing import Optional
import scipy.integrate as integrate
import warnings

from Evaluations.custom_types import NumericArrayLike
from Evaluations.util import (check_and_convert, KaplanMeierArea,
                              predict_mean_survival_time, predict_median_survival_time)


def l1_loss_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: Optional[NumericArrayLike] = None,
        train_event_indicators: Optional[NumericArrayLike] = None,
        method: str = "Hinge",
        log_scale: bool = False,
        predicted_time_method: str = "Median"
) -> float:
    """
    Calculate the L1 loss for the predicted survival curves from PyCox models.
    param predicted_survival_curves: pd.DataFrame, shape = (n_samples, n_times)
        Predicted survival curves for the testing samples
        DataFrame index represents the time coordinates for the given curves.
        DataFrame value represents the survival probabilities.
    param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    param train_event_times:structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    param method: string, default: "Hinge"
    param log_scale: boolean, default: False
    param predicted_time_method: string, default: "Median"
    :return:
        Value for the calculated L1 loss.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'PyCoxEvaluator' from Evaluator.py.", DeprecationWarning)
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    if (train_event_times is not None) and (train_event_indicators is not None):
        train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
    # Extracting the time buckets
    time_coordinates = predicted_survival_curves.index.values
    # computing the Survival function, and set the small negative value to zero
    survival_curves = predicted_survival_curves.values.T
    survival_curves[survival_curves < 0] = 0

    if predicted_time_method == "Median":
        predict_method = predict_median_survival_time
    elif predicted_time_method == "Mean":
        predict_method = predict_mean_survival_time
    else:
        error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
        raise TypeError(error)

    # get median/mean survival time from the predicted curve
    predicted_times = []
    for i in range(survival_curves.shape[0]):
        predicted_time = predict_method(survival_curves[i, :], time_coordinates)
        predicted_times.append(predicted_time)
    predicted_times = np.array(predicted_times)
    return l1_loss(predicted_times, event_times, event_indicators, train_event_times,
                   train_event_indicators, method, True, log_scale)


def l1_loss_sksurv(
        predicted_survival_curves: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: Optional[NumericArrayLike] = None,
        train_event_indicators: Optional[NumericArrayLike] = None,
        method: str = "Hinge",
        log_scale: bool = False,
        predicted_time_method: str = "Median"
) -> float:
    """
    Calculate the L1 loss for the predicted survival curves from scikit-survival models.
    param predicted_survival_curves: pd.DataFrame, shape = (n_samples, n_times)
        Predicted survival curves for the testing samples
        DataFrame index represents the time coordinates for the given curves.
        DataFrame value represents the survival probabilities.
    param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    param train_event_times:structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    param method: string, default: "Hinge"
    param log_scale: boolean, default: False
    param predicted_time_method: string, default: "Median"
    :return:
        Value for the calculated L1 loss.
    """
    warnings.warn("This function is deprecated and might be deleted in the future. "
                  "Please use the class 'ScikitSurvivalEvaluator' from Evaluator.py.", DeprecationWarning)
    event_times, event_indicators = check_and_convert(event_times, event_indicators)
    if (train_event_times is not None) and (train_event_indicators is not None):
        train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)

    if predicted_time_method == "Median":
        predict_method = predict_median_survival_time
    elif predicted_time_method == "Mean":
        predict_method = predict_mean_survival_time
    else:
        error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
        raise TypeError(error)

    # get median/mean survival time from the predicted curve
    predicted_times = []
    for i in range(predicted_survival_curves.shape[0]):
        predicted_time = predict_method(predicted_survival_curves[i].y, predicted_survival_curves[i].x)
        predicted_times.append(predicted_time)
    predicted_times = np.array(predicted_times)
    return l1_loss(predicted_times, event_times, event_indicators, train_event_times,
                   train_event_indicators, method, True, log_scale)


def l1_loss(
        predicted_times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        method: str = "Hinge",
        weighted: bool = True,
        log_scale: bool = False
) -> float:
    """
    Calculate the L1 loss for the predicted survival times.
    Parameters
    ----------
    predicted_times: np.ndarray, shape = (n_samples, )
        Predicted survival times for the testing samples
    event_times: np.ndarray, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    event_indicators: np.ndarray, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    train_event_times: np.ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators: np.ndarray, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    method: string, default: "Hinge"
        Type of l1 loss to use. Options are "Uncensored", "Hinge", "Margin", "IPCW-v1", "IPCW-v2", and "Pseudo_obs".
    weighted: boolean, default: True
        Whether to use weighting scheme for l1 loss.
    log_scale: boolean, default: False
        Whether to use log scale for the loss function.

    Returns
    -------
    Value for the calculated L1 loss.
    """
    event_indicators = event_indicators.astype(bool)
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)

    if method == "Uncensored":
        if log_scale:
            scores = np.log(event_times[event_indicators]) - np.log(predicted_times[event_indicators])
        else:
            scores = event_times[event_indicators] - predicted_times[event_indicators]
        return np.abs(scores).mean()
    elif method == "Hinge":
        weights = np.ones(predicted_times.size)
        if weighted:
            if train_event_times is None or train_event_indicators is None:
                error = "If 'weighted' is True for calculating Hinge, training set values must be included."
                raise ValueError(error)
            km_model = KaplanMeierArea(train_event_times, train_event_indicators)
            censor_times = event_times[~event_indicators]
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

        if log_scale:
            scores = np.log(event_times) - np.log(predicted_times)
        else:
            scores = event_times - predicted_times
        scores[~event_indicators] = np.maximum(scores[~event_indicators], 0)
        weighted_multiplier = 1 / (np.sum(event_indicators) + np.sum(weights))
        return weighted_multiplier * np.sum(np.abs(scores * weights))
    elif method == "Margin":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Margin' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        # Each best guess value has a confidence weight = 1 - KM(censoring time).
        # The earlier the patient got censored, the lower the confident weight is.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        # predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        def _km_linear_predict(time):
            slope = (1 - min(km_model.survival_probabilities)) / (0 - max(km_model.survival_times))

            # predict_prob = np.empty_like(time)
            # before_last_time_idx = time <= max(km_model.survival_times)
            # after_last_time_idx = time > max(km_model.survival_times)
            # predict_prob[before_last_time_idx] = km_model.predict(time[before_last_time_idx])
            # predict_prob[after_last_time_idx] = np.clip(1 + time[after_last_time_idx] * slope, a_min=0, a_max=None)
            if time <= max(km_model.survival_times):
                predict_prob = km_model.predict(time)
            else:
                predict_prob = max(1 + time * slope, 0)
            return predict_prob

        def _compute_best_guess(time):
            return time + integrate.quad(_km_linear_predict, time, km_linear_zero,
                                         limit=2000)[0] / km_model.predict(time)

        censor_times = event_times[~event_indicators]
        if weighted:
            weights = 1 - km_model.predict(censor_times)
        else:
            weights = np.ones(censor_times.size)
        best_guesses = km_model.best_guess_revise(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

        scores = np.empty(predicted_times.size)
        if log_scale:
            scores[event_indicators] = np.log(event_times[event_indicators]) - np.log(predicted_times[event_indicators])
            scores[~event_indicators] = weights * (np.log(best_guesses) - np.log(predicted_times[~event_indicators]))
        else:
            scores[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
            scores[~event_indicators] = weights * (best_guesses - predicted_times[~event_indicators])
        weighted_multiplier = 1 / (np.sum(event_indicators) + np.sum(weights))
        return weighted_multiplier * np.sum(np.abs(scores))
    elif method == "IPCW-v1":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'ipcw' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        # Each best guess value has a confidence weight = 1 - KM(censoring time).
        # The earlier the patient got censored, the lower the confident weight is.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        # predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        censor_times = event_times[~event_indicators]
        weights = np.ones(event_times.size)
        if weighted:
            weights[~event_indicators] = 1 - km_model.predict(censor_times)
        best_guesses = np.empty(shape=event_times.size)
        for i in range(event_times.size):
            if event_indicators[i] == 1:
                best_guesses[i] = event_times[i]
            else:
                # Numpy will throw a warning if afterward_event_times are all false. TODO: consider change the code.
                afterward_event_idx = train_event_times[train_event_indicators == 1] > event_times[i]
                best_guesses[i] = np.mean(train_event_times[train_event_indicators == 1][afterward_event_idx])
        # NaN values are generated because there are no events after the censor times
        nan_idx = np.argwhere(np.isnan(best_guesses))
        predicted_times = np.delete(predicted_times, nan_idx)
        best_guesses = np.delete(best_guesses, nan_idx)
        weights = np.delete(weights, nan_idx)
        if log_scale:
            scores = np.log(best_guesses) - np.log(predicted_times)
        else:
            scores = best_guesses - predicted_times
        weighted_multiplier = 1 / np.sum(weights)
        return weighted_multiplier * np.sum(np.abs(scores) * weights)
    elif method == "IPCW-v2":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'ipcw' is chosen, training set values must be included."
            raise ValueError(error)
        # Use KM to estimate the censor distribution
        inverse_train_event_indicators = 1 - train_event_indicators

        ipc_model = KaplanMeierArea(train_event_times, inverse_train_event_indicators)
        ipc_pred = ipc_model.predict(event_times)
        # Catch if denominator is 0. This happens when the time is later than the last event time in trainset.
        ipc_pred[ipc_pred == 0] = np.inf
        if log_scale:
            scores = np.log(event_times) - np.log(predicted_times)
        else:
            scores = event_times - predicted_times
        return (np.abs(scores)[event_indicators] / ipc_pred[event_indicators]).mean()
    elif method == "Pseudo_obs":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'pseudo_observation' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        # Each best guess value has a confidence weight = 1 - KM(censoring time).
        # The earlier the patient got censored, the lower the confident weight is.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        # predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        censor_times = event_times[~event_indicators]
        weights = np.ones(event_times.size)
        if weighted:
            weights[~event_indicators] = 1 - km_model.predict(censor_times)
        best_guesses = np.empty(shape=event_times.size)
        test_data_size = event_times.size
        sub_expect_time = km_model._compute_best_guess(0)
        train_data_size = train_event_times.size
        total_event_time = np.empty(shape=train_data_size + 1)
        total_event_indicator = np.empty(shape=train_data_size + 1)
        total_event_time[0:-1] = train_event_times
        total_event_indicator[0:-1] = train_event_indicators
        for i in range(test_data_size):
            if event_indicators[i] == 1:
                best_guesses[i] = event_times[i]
            else:
                total_event_time[-1] = event_times[i]
                total_event_indicator[-1] = event_indicators[i]
                total_km_model = KaplanMeierArea(total_event_time, total_event_indicator)
                total_expect_time = total_km_model._compute_best_guess(0)
                best_guesses[i] = (train_data_size + 1) * total_expect_time - train_data_size * sub_expect_time
        if log_scale:
            scores = np.log(best_guesses) - np.log(predicted_times)
        else:
            scores = best_guesses - predicted_times
        weighted_multiplier = 1 / np.sum(weights)
        return weighted_multiplier * np.sum(np.abs(scores * weights))
    elif method == "Pseudo_obs_pop":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Pseudo_obs_pop' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the population best guess time given the KM curve.
        # The population best guess time is identical for all people.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        sub_expect_time = km_model._compute_best_guess(0)
        best_guesses = event_times.copy()
        best_guesses[~event_indicators] = sub_expect_time
        censor_times = event_times[~event_indicators]
        weights = np.ones(event_times.size)
        if weighted:
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

        if log_scale:
            scores = np.log(best_guesses) - np.log(predicted_times)
        else:
            scores = best_guesses - predicted_times
        weighted_multiplier = 1 / np.sum(weights)
        return weighted_multiplier * np.sum(np.abs(scores * weights))
    else:
        raise ValueError("Method must be one of 'Uncensored', 'Hinge', 'Margin', 'IPCW-v1', 'IPCW-v2' "
                         "'Pseudo_obs', or 'Pseudo_obs_pop'. Got '{}' instead.".format(method))


if __name__ == "__main__":
    # Test the functions
    train_t = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        26, 27, 28, 29, 30, 31, 32, 33, 34,  60, 61, 62, 63, 64, 65, 66, 67,
                        74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                        98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                        117, 118, 119, 120, 120, 120, 121, 121, 124, 125, 126, 127, 128, 129,
                        136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                        155, 156, 157, 158, 159, 161, 182, 183, 186, 190, 191, 192, 192, 192,
                        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 202, 203,
                        204, 202, 203, 204, 212, 213, 214, 215, 216, 217, 222, 223, 224])
    train_e = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                        1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                        0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                        0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    t = np.array([5, 10, 19, 31, 43, 59, 63, 75, 97, 113, 134, 151, 163, 176, 182, 195, 200, 210, 220])
    e = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    predict_time = np.array([18, 19, 5, 12, 75, 100, 120, 85, 36, 95, 170, 41, 200, 210, 260, 86, 100, 120, 140])
    l1 = l1_loss(predict_time, t, e, train_t, train_e, method='Pseudo_obs')
    print(l1)
