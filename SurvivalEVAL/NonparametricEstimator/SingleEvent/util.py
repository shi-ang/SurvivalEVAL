import numpy as np

from SurvivalEVAL.Evaluations.util import get_prob_at_zero


def km_mean(times: np.ndarray, survival_probabilities: np.ndarray) -> float:
    """
    Calculate the mean of the Kaplan-Meier curve.

    Parameters
    ----------
    times: np.ndarray, shape = (n_samples, )
        Survival times for KM curve of the testing samples
    survival_probabilities: np.ndarray, shape = (n_samples, )
        Survival probabilities for KM curve of the testing samples

    Returns
    -------
    The mean of the Kaplan-Meier curve.
    """
    # calculate the area under the curve for each interval
    area_probabilities = np.append(1, survival_probabilities)
    area_times = np.append(0, times)
    km_linear_zero = -1 / ((area_probabilities[-1] - 1) / area_times[-1])
    if survival_probabilities[-1] != 0:
        area_times = np.append(area_times, km_linear_zero)
        area_probabilities = np.append(area_probabilities, 0)
    area_diff = np.diff(area_times, 1)
    # we are using trap rule
    average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
    area = np.flip(np.flip(area_diff * average_probabilities).cumsum())
    area = np.append(area, 0)
    # or the step function rule (deprecated for now)
    # area_subs = area_diff * area_probabilities[0:-1]
    # area_subs[-1] = area_subs[-1] / 2
    # area = np.flip(np.flip(area_subs).cumsum())

    # calculate the mean
    surv_prob = get_prob_at_zero(times, survival_probabilities)
    return area[0] / surv_prob


def infer_survival_probabilities(
    prediction_times, survival_times, survival_probabilities
):
    indices = np.searchsorted(survival_times, prediction_times, side="right") - 1
    # Index -1 denotes the pre-observation baseline, where survival is still 1.
    before_first = indices < 0
    indices = np.clip(indices, 0, survival_times.size - 1)
    probs = survival_probabilities[indices].astype(float, copy=True)
    probs[before_first] = 1.0

    # Extrapolate linearly for times beyond the last observed time point
    # using the line connecting (t_last, S(t_last)) and (t0, S(t0))
    beyond_last = prediction_times > survival_times[-1]
    if np.any(beyond_last):
        t0, s0 = survival_times[0], survival_probabilities[0]
        t_last, s_last = survival_times[-1], survival_probabilities[-1]
        denom = t_last - t0
        slope = 0.0 if denom == 0 else (s_last - s0) / denom
        extrapolated = s_last + slope * (prediction_times[beyond_last] - t_last)
        probs[beyond_last] = np.maximum(extrapolated, 0.0)
    return probs
