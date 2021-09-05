import numpy as np
import pandas as pd
import sksurv.metrics as metrics

from Evaluations.custom_types import NumericArrayLike
from Evaluations.util import check_and_convert, predict_mean_survival_time, predict_median_survival_time


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
        predicted_survival_time: np.ndarray,
        event_time: np.ndarray,
        event_indicator: np.ndarray,
        ties: str = "None"
) -> (float, float, int):

    # the scikit-survival concordance function only takes risk scores to calculate.
    # So at first we should transfer the predicted time -> risk score.
    # The risk score should be higher for subjects that live shorter (i.e. lower average survival time).
    risk = -1 * predicted_survival_time
    event_indicator = event_indicator.astype(bool)
    cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = metrics.concordance_index_censored(
        event_indicator, event_time, estimate=risk)
    if ties == "None":
        # cindex = concordant_pairs / ( concordant_pairs + discordant_pairs)
        # , which is the first output from concordance_index_censored()
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
