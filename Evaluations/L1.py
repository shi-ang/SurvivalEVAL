import numpy as np
import pandas as pd
from typing import Optional
import scipy.integrate as integrate

from Evaluations.custom_types import NumericArrayLike
from Evaluations.util import check_and_convert, KaplanMeier, predict_mean_survival_time, predict_median_survival_time, KaplanMeierArea


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

    :param predicted_survival_curves: pd.DataFrame, shape = (n_samples, n_times)
        Predicted survival curves for the testing samples
        DataFrame index represents the time coordinates for the given curves.
        DataFrame value represents the survival probabilities.
    :param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    :param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    :param train_event_times:structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    :param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    :param method: string, default: "Hinge"
    :param log_scale: boolean, default: False
    :param predicted_time_method: string, default: "Median"
    :return:
        Value for the calculated L1 loss.
    """
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
                   train_event_indicators, method, log_scale)


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

    :param predicted_survival_curves: pd.DataFrame, shape = (n_samples, n_times)
        Predicted survival curves for the testing samples
        DataFrame index represents the time coordinates for the given curves.
        DataFrame value represents the survival probabilities.
    :param event_times: structured array, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    :param event_indicators: structured array, shape = (n_samples, )
        Binary indicators of censoring for the testing samples
    :param train_event_times:structured array, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    :param train_event_indicators: structured array, shape = (n_train_samples, )
        Binary indicators of censoring for the training samples
    :param method: string, default: "Hinge"
    :param log_scale: boolean, default: False
    :param predicted_time_method: string, default: "Median"
    :return:
        Value for the calculated L1 loss.
    """
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
                   train_event_indicators, method, log_scale)


def l1_loss(
        predicted_times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        method: str = "Hinge",
        log_scale: bool = False
) -> float:

    event_indicators = event_indicators.astype(bool)
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)

    if method == "Uncensored":
        if log_scale:
            scores = np.log(event_times[event_indicators]) - np.log(predicted_times[event_indicators])
        else:
            scores = event_times[event_indicators] - predicted_times[event_indicators]
        return np.mean(np.abs(scores))
    elif method == "Hinge":
        if log_scale:
            scores = np.log(event_times) - np.log(predicted_times)
        else:
            scores = event_times - predicted_times
        scores[~event_indicators] = np.maximum(scores[~event_indicators], 0)
        return np.mean(np.abs(scores))
    elif method == "Margin":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'margin' is chosen, training set values must be included."
            raise ValueError(error)

        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        # Each best guess value has a confidence weight = 1 - KM(censoring time).
        # The earlier the patient got censored, the lower the confident weight is.
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

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
        weights = 1 - km_model.predict(censor_times)
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
    else:
        error = """Please enter one of 'Uncensored', 'Hinge', or 'Margin' for L1 loss type."""
        raise TypeError(error)
