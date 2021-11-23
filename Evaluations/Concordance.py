import numpy as np
import pandas as pd
import sksurv.metrics as metrics
from typing import Optional

from Evaluations.custom_types import NumericArrayLike
from Evaluations.util import check_and_convert, predict_mean_survival_time, predict_median_survival_time, KaplanMeierArea


def concordance_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        ties: str = "None",
        predicted_time_method: str = "Median"
) -> (float, float, int):

    event_time, event_indicator = check_and_convert(event_time, event_indicator)
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

    return concordance(predicted_times, event_time, event_indicator, ties)


def concordance_sksurv(
        predicted_survival_curves: np.ndarray,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        ties: str = "None",
        predicted_time_method: str = "Median"
) -> (float, float, int):
    event_time, event_indicator = check_and_convert(event_time, event_indicator)

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

    return concordance(predicted_times, event_time, event_indicator, ties)


def concordance(
        predicted_times: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: Optional[np.ndarray] = None,
        train_event_indicators: Optional[np.ndarray] = None,
        method: str = "Hinge",
        ties: str = "None"
) -> (float, float, int):

    """
    :param predicted_times:
    :param event_times:
    :param event_indicators:
    :param train_event_times:
    :param train_event_indicators:
    :param method:
    :param ties:
        A string indicating the way ties should be handled. Options: "None" will throw out all ties in
        survival time and all ties from risk scores. "Time" includes ties in survival time but removes ties
        in risk scores. "Risk" includes ties in risk scores but not in survival time. "All" includes all
        ties (both in survival time and in risk scores). Note the concordance calculation is given by
        (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
    :return:
    """
    # the scikit-survival concordance function only takes risk scores to calculate.
    # So at first we should transfer the predicted time -> risk score.
    # The risk score should be higher for subjects that live shorter (i.e. lower average survival time).

    event_indicators = event_indicators.astype(bool)

    if method == "Comparable":
        risk = -1 * predicted_times
    elif method == "Margin":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Margin' is chosen, training set values must be included."
            raise ValueError(error)

        train_event_indicators = train_event_indicators.astype(bool)

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / ((1 - min(km_model.survival_probabilities))/(0 - max(km_model.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)
        risk = -1 * predicted_times

        censor_times = event_times[~event_indicators]
        best_guesses = km_model.best_guess_revise(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[censor_times > km_linear_zero]

        event_times[~event_indicators] = best_guesses
    else:
        raise TypeError("Method for calculating concordance is unrecognized.")

    cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = metrics.concordance_index_censored(
        event_indicators, event_times, estimate=risk)
    if ties == "None":
        total_pairs = concordant_pairs + discordant_pairs
    elif ties == "Time":
        total_pairs = concordant_pairs + discordant_pairs + time_ties
        concordant_pairs = concordant_pairs + 0.5 * time_ties
        cindex = concordant_pairs / total_pairs
    elif ties == "Risk":
        total_pairs = concordant_pairs + discordant_pairs + risk_ties
        concordant_pairs = concordant_pairs + 0.5 * risk_ties
        cindex = concordant_pairs / total_pairs
    elif ties == "All":
        total_pairs = concordant_pairs + discordant_pairs + risk_ties + time_ties
        concordant_pairs = concordant_pairs + 0.5 * (risk_ties + time_ties)
        cindex = concordant_pairs / total_pairs
    else:
        error = "Please enter one of 'None', 'Time', 'Risk', or 'All' for handling ties for concordance."
        raise TypeError(error)

    return cindex, concordant_pairs, total_pairs
