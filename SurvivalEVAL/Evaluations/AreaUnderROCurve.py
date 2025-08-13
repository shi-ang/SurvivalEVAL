import numpy as np
from sklearn.metrics import roc_auc_score


def auc(
        predict_probs: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        target_time: float = None
) -> float:
    """
    Calculate the Area Under the Curve (AUC) for the survival model.

    Parameters
    ----------
    predict_probs: np.ndarray
        The predicted survival probabilities
    event_times: np.ndarray
        The event or censoring times for the test data
    event_indicators: np.ndarray
        The binary indicators of whether the event occurred (1) or was censored (0)
    target_time: float, optional
        The specific time point at which to calculate the AUC. If not specified, the median of the event times is used.

    Returns
    -------
    auc: float
        The AUC value calculated at the specified target time.
    """
    # if the target time is not specified, then we use the median of the event times
    if target_time is None:
        target_time = np.median(event_times)

    # for censored data, if the censor time is earlier than the target time,
    # (since we cannot observe the real status at the target time)
    # then we just exclude its prediction and observation from the calculation
    exclude_indicators = np.logical_and(event_times < target_time, event_indicators == 0)
    event_times = event_times[~exclude_indicators]
    predict_probs = predict_probs[~exclude_indicators]

    # get the binary status of the test data, given the target time
    binary_status = (event_times <= target_time).astype(int)

    # check if the binary status is all zeros or all ones
    if np.all(binary_status == 0) or np.all(binary_status == 1):
        raise ValueError(f"Survival status is all zeros or all ones at time: {target_time}, AUC cannot be computed.")

    # computing the AUC, given the predicted probabilities and the binary status
    risks = 1 - predict_probs
    return roc_auc_score(binary_status, risks)


if __name__ == '__main__':
    # test the AUC function
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.8, 0.9])
    times = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    indicators = np.array([1, 1, 0, 1, 1, 1, 1, 1, 1])
    target_t = 5

    print(auc(probs, times, indicators, target_t))