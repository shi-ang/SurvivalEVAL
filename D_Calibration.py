import numpy as np
import pandas as pd
from scipy.stats import chisquare
# from pysurvival.models.survival_forest import RandomSurvivalForestModel

from custom_types import NumericArrayLike
from util import check_and_convert, predict_prob_from_curve


def d_calibration_pycox(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        num_bins: int = 10
) -> (np.ndarray, float):
    """

    :param predicted_survival_curves:
    :param event_time:
    :param event_indicator:
    :param num_bins:
    :return:
    """

    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(event_time, event_indicator)
    censor_status = 1 - uncensor_status
    # Extracting the time buckets
    time_coordinates = predicted_survival_curves.index.values
    # computing the Survival function, and set the small negative value to zero
    survival_curves = predicted_survival_curves.values.T
    survival_curves[survival_curves < 0] = 0

    quantile = np.linspace(1, 0, num_bins + 1)

    predict_probs = []
    for i in range(survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(survival_curves[i, :], time_coordinates, true_event_times[i])
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)

    uncensored_probabilities = predict_probs[uncensor_status.astype(bool)]
    uncensored_position = np.digitize(uncensored_probabilities, quantile)
    uncensored_position[uncensored_position == 0] = 1     # class probability==1 to the first bin

    uncensored_binning = np.zeros([num_bins])
    for i in range(len(uncensored_position)):
        uncensored_binning[uncensored_position[i] - 1] += 1

    censored_probabilities = predict_probs[censor_status.astype(bool)]

    censor_binning = np.zeros([num_bins])
    if len(censored_probabilities) > 0:
        for i in range(len(censored_probabilities)):
            partial_binning = create_censor_binning(censored_probabilities[i], num_bins)
            censor_binning += partial_binning

    combine_binning = uncensored_binning + censor_binning
    _, pvalue = chisquare(combine_binning)
    return combine_binning, pvalue


def d_calibration_sksurv(
        predicted_survival_curves: pd.DataFrame,
        event_time: NumericArrayLike,
        event_indicator: NumericArrayLike,
        num_bins: int = 10
) -> (np.ndarray, float):
    """

    :param predicted_survival_curves:
    :param event_time:
    :param event_indicator:
    :param num_bins:
    :return:
    """

    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(event_time, event_indicator)
    censor_status = 1 - uncensor_status

    quantile = np.linspace(1, 0, num_bins + 1)

    predict_probs = []
    for i in range(predicted_survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(predicted_survival_curves[i].x, predicted_survival_curves[i].y,
                                               true_event_times[i])
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)

    uncensored_probabilities = predict_probs[uncensor_status.astype(bool)]
    uncensored_position = np.digitize(uncensored_probabilities, quantile)
    uncensored_position[uncensored_position == 0] = 1     # class probability==1 to the first bin

    uncensored_binning = np.zeros([num_bins])
    for i in range(len(uncensored_position)):
        uncensored_binning[uncensored_position[i] - 1] += 1

    censored_probabilities = predict_probs[censor_status.astype(bool)]

    censor_binning = np.zeros([num_bins])
    if len(censored_probabilities) > 0:
        for i in range(len(censored_probabilities)):
            partial_binning = create_censor_binning(censored_probabilities[i], num_bins)
            censor_binning += partial_binning

    combine_binning = uncensored_binning + censor_binning
    _, pvalue = chisquare(combine_binning)
    return combine_binning, pvalue



def d_calibration_pysurvival(model, X, T, E, num_bins=10) -> (np.ndarray, float):
    # Checking the format of the data
    true_event_times, uncensor_status = check_and_convert(T, E)
    censor_status = 1 - uncensor_status
    # computing the Survival function
    survival_curves = model.predict_survival(X, None)

    # Extracting the time buckets
    time_coordinates = model.times

    # if isinstance(model, RandomSurvivalForestModel):
    #     time_coordinates.pop()
    #     survival_curves = np.delete(survival_curves, -1, 1)
    #     print('use rsf, delete last time point')

    quantile = np.linspace(1, 0, num_bins + 1)

    predict_probs = []
    for i in range(survival_curves.shape[0]):
        predict_prob = predict_prob_from_curve(survival_curves[i, :], time_coordinates, true_event_times[i])
        predict_probs.append(predict_prob)
    predict_probs = np.array(predict_probs)

    uncensoredProbabilities = predict_probs[uncensor_status.astype(bool)]
    uncensoredPosition = np.digitize(uncensoredProbabilities, quantile)
    uncensoredPosition[uncensoredPosition == 0] = 1     # class probability==1 to the first bin

    uncensoredBinning = np.zeros([num_bins])
    for i in range(len(uncensoredPosition)):
        uncensoredBinning[uncensoredPosition[i] - 1] += 1

    censoredProbabilities = predict_probs[censor_status.astype(bool)]

    censor_binning = np.zeros([num_bins])
    if len(censoredProbabilities) > 0:
        for i in range(len(censoredProbabilities)):
            partial_binning = create_censor_binning(censoredProbabilities[i], num_bins)
            censor_binning += partial_binning

    combine_binning = uncensoredBinning + censor_binning
    _, pvalue = chisquare(combine_binning)
    return combine_binning, pvalue


def create_censor_binning(probability, num_bins) -> np.ndarray:
    """
    For censoring instance,
    b1 will be the infimum probability of the bin that contains S(c),
    for the bin of [b1, b2) which contains S(c), probability = (S(c) - b1) / S(c)
    for the rest of the bins, [b2, b3), [b3, b4), etc., probability = 1 / (B * S(c)), where B is the number of bins.
    :param probability:
        probability of the instance that will happen the event at the true event time
        based on the predicted survival curve.
    :param num_bins: number of bins
    :return: probabilities at each bin
    """
    quantile = np.linspace(1, 0, num_bins + 1)
    censor_binning = [0.0] * 10
    for i in range(num_bins):
        if probability == 1:
            censor_binning = [0.1] * 10
        elif quantile[i] > probability >= quantile[i + 1]:
            first_bin = (probability - quantile[i + 1]) / probability if probability != 0 else 1
            rest_bins = 1 / (num_bins * probability) if probability != 0 else 0
            censor_binning = [0.0] * i + [first_bin] + [rest_bins] * (num_bins - i - 1)
    # assert len(censor_binning) == 10, "censor binning should have size of 10"
    final_binning = np.array(censor_binning)
    return final_binning
