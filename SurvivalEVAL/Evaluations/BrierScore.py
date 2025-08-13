import numpy as np
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier


def single_brier_score(
        preds: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: np.ndarray,
        train_event_indicators: np.ndarray,
        target_time: float = None,
        ipcw: bool = True
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
        weight_cat1 = ((event_times <= target_time) & event_indicators)
        weight_cat2 = (event_times > target_time)

    b_score = (np.square(preds) * weight_cat1 + np.square(1 - preds) * weight_cat2).mean()
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


def brier_multiple_points(
        pred_mat: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        train_event_times: np.ndarray,
        train_event_indicators: np.ndarray,
        target_times: np.ndarray,
        ipcw: bool = True
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
    target_times: float, default: None
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
    target_times_mat = np.repeat(target_times.reshape(1, -1), repeats=len(event_times), axis=0)
    event_times_mat = np.repeat(event_times.reshape(-1, 1), repeats=len(target_times), axis=1)
    event_indicators_mat = np.repeat(event_indicators.reshape(-1, 1), repeats=len(target_times), axis=1)
    event_indicators_mat = event_indicators_mat.astype(bool)

    if ipcw:
        inverse_train_event_indicators = 1 - train_event_indicators

        ipc_model = KaplanMeier(train_event_times, inverse_train_event_indicators)

        # Category one calculates IPCW weight at observed time point.
        # Category one is individuals with event time lower than the time of interest and were NOT censored.
        ipc_pred = ipc_model.predict(event_times_mat)
        # Catch if denominator is 0.
        ipc_pred[ipc_pred == 0] = np.inf
        weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat) / ipc_pred
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
        weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat)
        weight_cat2 = (event_times_mat > target_times_mat)

    ipcw_square_error_mat = np.square(pred_mat) * weight_cat1 + np.square(1 - pred_mat) * weight_cat2
    brier_scores = np.mean(ipcw_square_error_mat, axis=0)
    return brier_scores
