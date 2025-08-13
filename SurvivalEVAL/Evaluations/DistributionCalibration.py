import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chisquare
from scipy.integrate import trapezoid

from SurvivalEVAL.Evaluations.util import interpolated_survival_curve
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier, NelsonAalen


def d_calibration(
        pred_probs: np.ndarray,
        event_indicators: np.ndarray,
        num_bins: int = 10
) -> (float, np.ndarray):
    """
    Calculate the D-Calibration score.
    Parameters
    ----------
    pred_probs: np.ndarray
        The predicted survival probabilities at individual's event/censor time.
    event_indicators: np.ndarray
        The event indicators.
    num_bins: int
        The number of bins to use for the D-Calibration score.

    Returns
    -------
    pvalue: float
        The p-value of the D-Calibration test.
    combine_hist: np.ndarray
        The binning histogram of the D-Calibration test.
    """
    quantile = np.linspace(1, 0, num_bins + 1)
    censor_indicators = 1 - event_indicators

    event_probs = pred_probs[event_indicators.astype(bool)]
    event_position = np.digitize(event_probs, quantile)
    event_position[event_position == 0] = 1     # class probability==1 to the first bin

    event_hist = np.zeros([num_bins])
    for i in range(len(event_position)):
        event_hist[event_position[i] - 1] += 1

    censored_probs = pred_probs[censor_indicators.astype(bool)]

    censor_hist = np.zeros([num_bins])
    if len(censored_probs) > 0:
        for i in range(len(censored_probs)):
            partial_binning = create_censor_hist(censored_probs[i], num_bins)
            censor_hist += partial_binning

    combine_hist = event_hist + censor_hist
    _, pvalue = chisquare(combine_hist)
    return pvalue, combine_hist


def create_censor_hist(
        prob: float,
        num_bins: int
) -> np.ndarray:
    """    Create the binning histogram for a right censored instance.

    The bins are defined as follows:
    [b_0, b_1), [b_1, b_2), ..., [b_{num_bins-1}, b_num_bins] (note that the last bin is closed on both side).

    For censoring instance,
    b2 will be the infimum probability of the bin that contains S(c),
    for the bin of [b2, b3) which contains S(c), probability = (S(c) - b2) / S(c)
    for the rest of the bins, [b0, b1), [b1, b2), etc., probability = 1 / (B * S(c)), where B is the number of bins.
    :param prob: float

        The predicted probability at the censored time of a censoring instance.
    :param num_bins: int
        The number of bins to use for the D-Calibration score.
    :return:
    final_binning: np.ndarray
        The "split" histogram of this censored subject.
    """
    quantile = np.linspace(1, 0, num_bins + 1)
    censor_binning = np.zeros(num_bins)
    for i in range(num_bins):
        if prob == 1:
            censor_binning += 0.1
            break
        elif quantile[i] > prob >= quantile[i + 1]:
            first_bin = (prob - quantile[i + 1]) / prob if prob != 0 else 1
            rest_bins = 1 / (num_bins * prob) if prob != 0 else 0
            censor_binning[i] += first_bin
            censor_binning[i + 1:] += rest_bins
            break
    # assert len(censor_binning) == num_bins, f"censor binning should have size of {num_bins}"
    return censor_binning


def d_cal_interval_cen(
        pred_probs_left: np.ndarray,
        pred_probs_right: np.ndarray,
        num_bins: int = 10
) -> (float, np.ndarray):
    """
    Calculate the D-Calibration score for interval censored data.
    Parameters
    ----------
    pred_probs_left: np.ndarray
        The predicted survival probabilities at individual's left event/censor time.
    pred_probs_right: np.ndarray
        The predicted survival probabilities at individual's right event/censor time.
        For right-censored instances, the right event/censor time is infinity, so the predicted probability is 0.
    num_bins: int
        The number of bins to use for the D-Calibration score.
    Returns
    -------
    pvalue: float
        The p-value of the D-Calibration test.
    binning: np.ndarray
        The binning histogram of the D-Calibration test.
    """
    assert len(pred_probs_left) == len(pred_probs_right), \
        "The length of pred_probs_left and pred_probs_right should have same length."

    assert np.all(pred_probs_left >= pred_probs_right), \
        "The left survival probabilities should be greater than or equal to the right survival probabilities."

    assert (np.all(pred_probs_left >= 0) and np.all(pred_probs_right >= 0) and
            np.all(pred_probs_left <= 1) and np.all(pred_probs_right <= 1)), \
        "The predicted probabilities should be in the range [0, 1]."

    n = len(pred_probs_left)

    binning = np.zeros([num_bins])
    for i in range(n):
        partial_binning = create_interval_c_hist(pred_probs_left[i], pred_probs_right[i], num_bins)
        binning += partial_binning

    _, pvalue = chisquare(binning)
    return pvalue, binning


def create_interval_c_hist(
        prob_left: float,
        prob_right: float,
        num_bins: int
) -> np.ndarray:
    """
    Create the binning histogram for an interval censored instance.
    For interval censored instance, we have two predicted probabilities: left and right.
    The left probability is the predicted survival probability at the left censored time,
    and the right probability is the predicted survival probability at the right censored time.

    The bins are defined as follows:
    [b_0, b_1), [b_1, b_2), ..., [b_{num_bins-1}, b_num_bins] (note that the last bin is closed on both side).

    (1) If the left and right probabilities are within the same bin, then the histogram will be 1 for that bin,
    and 0 for the rest of the bins.
    (2) If the left and right probabilities are in different bins, then the histogram will be split:
        - The bin (e.g., [b3, b4))that contains the left probability will have a probability of
        (prob_left - b3) / (prob_left - prob_right)
        - The bin (e.g., [b1, b2)) that contains the right probability will have a probability of
        (b2 - prob_right) / (prob_left - prob_right)
        - The intermediate bins (e.g., [b2, b3)) will have a probability of
        1 / (num_bins * (prob_left - prob_right))

    :param prob_left: float
        The predicted probability at the left censored time of an interval censoring instance.
    :param prob_right: float
        The predicted probability at the right censored time of an interval censoring instance.
    :param num_bins: int
        The number of bins to use for the D-Calibration score.
    :return:
    hist: np.ndarray
        The "split" histogram of this interval censored subject.
    """
    # make sure the left and right probabilities are in the range [0, 1]
    if prob_right == 0:
        # if the right probability is 0, then it is a right-censored instance,
        return create_censor_hist(prob_left, num_bins)
    else:
        if prob_left < prob_right:
            # enforce the natural order for survival probs
            prob_left, prob_right = prob_right, prob_left

        edges = np.linspace(1.0, 0.0, num_bins + 1)  # descending: 1, 1-1/K, ..., 0

        left_idx = np.digitize(prob_left, edges) - 1 if prob_left < 1.0 else 0
        right_idx = np.digitize(prob_right, edges) - 1
        hist = np.zeros(num_bins, dtype=float)
        if left_idx == right_idx:
            hist[left_idx] = 1.0
            return hist
        else:
            # if the left and right probabilities are in different bins
            first_hist = (prob_left - edges[left_idx + 1]) / (prob_left - prob_right)
            hist[left_idx] += first_hist

            last_hist = (edges[right_idx] - prob_right) / (prob_left - prob_right)
            hist[right_idx] += last_hist

            # fill the intermediate bins with equal probability
            if right_idx > left_idx + 1:
                intermediate_hist = (1.0 - first_hist - last_hist) / (right_idx - left_idx - 1)
                hist[left_idx + 1:right_idx] = intermediate_hist

            return hist


_residual_names = {
    "CoxSnell": "Cox-Snell Residuals",
    "Modified CoxSnell-v1": "Cox-Snell Residuals",
    "Modified CoxSnell-v2": "Cox-Snell Residuals",
    "Martingale": "Martingale Residuals",
    "Deviance": "Deviance Residuals"
}


def residuals(
        pred_probs: np.ndarray,
        event_indicators: np.ndarray,
        method: str = "CoxSnell",
        draw_figure: bool = False
) -> np.ndarray:
    """
    Calculate the residuals based on the predicted probabilities and event times.
    For details of the residuals, please refer to the Chapter 4 of the book:
    "Modelling survival data in medical research" By David Collett

    Parameters
    ----------
    pred_probs: np.ndarray
        The predicted survival probabilities at individual's event/censor time.
    event_indicators: np.ndarray
        The event indicators.
    method: str
        The method to calculate residuals. Options are "CoxSnell", "Modified CoxSnell-v1", "Modified CoxSnell-v2",
        "Martingale", "Deviance".
    draw_figure: bool
        Whether to visualize the Cox-Snell residuals and draw the cumulative hazard plot.
    Returns
    -------
    np.ndarray
        The calculated residuals.
    """
    cox_residuals = - np.log(pred_probs)

    if method == "CoxSnell":
        residuals = cox_residuals
    elif method == "Modified CoxSnell-v1" or method == "Modified CoxSnell-v2":
        # Compare with standard CoxSnell residuals, this method adds an 'excess residual' for censored instances.
        # The excess residual should also follow a unit exponential distribution, based on the lack of memory property.
        # There are two choices of excess residuals:
        # (1) use the mean of the unit exponential distribution, which is 1,
        # or (2) use the median of the unit exponential distribution, which is ln(2).
        excess_residual = 1 if method == "Modified CoxSnell-v1" else np.log(2)
        residuals = cox_residuals + excess_residual * (1 - event_indicators)
    elif method == "Martingale":
        residuals = event_indicators - cox_residuals
    elif method == "Deviance":
        def safe_log(x):
            return np.log(x + 1e-8)

        martingale_res = event_indicators - cox_residuals
        # get the sign of the martingale residuals
        sign = np.sign(martingale_res)
        residuals = sign * np.sqrt(-2 * (martingale_res + event_indicators * safe_log(cox_residuals)))
    else:
        raise ValueError("Unknown method {}".format(method))

    if draw_figure:
        cum_haz_empirical = NelsonAalen(cox_residuals, event_indicators)
        max_res = np.max(cox_residuals)
        fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True, dpi=400)
        ax[0].plot(cum_haz_empirical.survival_times, cum_haz_empirical.cumulative_hazard, label='Empirical')
        ax[0].plot([0, max_res], [0, max_res], label='Ideal', linestyle='--', color='red')
        ax[0].legend()
        ax[0].set_xlabel('Cox-Snell Residuals')
        ax[0].set_ylabel('Cumlative hazard of residuals')

        # use solid scatter points for uncensored instances and hollow scatter points for censored instances
        idx = np.arange(len(residuals))
        event_indicators = event_indicators.astype(bool)
        ax[1].scatter(idx[event_indicators], residuals[event_indicators], alpha=0.5, color='red', label='Event')
        ax[1].scatter(idx[~event_indicators], residuals[~event_indicators], facecolors='none', edgecolors='red', alpha=0.5, label='Censored')
        ax[1].set_xlabel('Index')
        ax[1].set_ylabel(_residual_names[method])
        ax[1].legend()
        plt.show()
    return residuals


def km_calibration(
        average_survival_curve: np.ndarray,
        time_coordinates: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        interpolation_method: str = 'Linear',
        draw_figure: bool = False
) -> float:
    """
    Calculate the KM calibration score between the average prediction curve and KM curve.
    The first version of KM calibration [1] is by visual inspection of the KM curve and the average curve.
    The second version of KM calibration [2] is by calculating the KL divergence between the KM curve and the average.
    This function actually calculates the (normalized) integrated mean squared error
    between the KM curve and the average prediction curve.

    This version has three benefits over [1] and [2]:
    1. It is a single number, so it is easier to compare.
    2. This calculation is symmetric (note that KL-divergence is not).
    3. The score is between 0 and 1, where 0 means perfect calibration and 1 means worst calibration.
        And the random prediction curve will have a score of 0.25.

    [1] Chapfuwa et al., Calibration and Uncertainty in Neural Time-to-Event Modeling， TNNLS， 2020
    [2] Yanagisawa, Proper Scoring Rules for Survival Analysis, ICML, 2023

    Parameters
    ----------
    average_survival_curve: np.ndarray
        The average survival curves.
    time_coordinates: np.ndarray
        The time coordinates of the average survival curves.
    event_times: np.ndarray
        The event time of the test data.
    event_indicators: np.ndarray
        The event indicator of the test data.\
    draw_figure: bool
        Whether to visualize the comparison of the KM curve and average curve.

    Returns
    -------
    mse: float
        The (normalized) integrated mean squared error between the KM curve and the average prediction curve.
    """
    unique_event_times = np.unique(event_times[event_indicators == 1])

    km_model = KaplanMeier(event_times, event_indicators)
    km_curve = km_model.predict(unique_event_times)

    # add 0 to the beginning of the time coordinates, and 1 to the beginning of the average_survival_curves
    unique_event_times = np.concatenate([[0], unique_event_times])
    km_curve = np.concatenate([[1], km_curve])
    if time_coordinates[0] != 0:
        time_coordinates = np.concatenate([[0], time_coordinates])
        average_survival_curve = np.concatenate([[1], average_survival_curve])

    # interpolate the average curve, so that it will have the same time coordinates as km_curve
    spline = interpolated_survival_curve(time_coordinates, average_survival_curve, interpolation_method)
    average_survival_curve = spline(unique_event_times)
    average_survival_curve = np.clip(average_survival_curve, 0, 1)

    # integrated over the joint time coordinates
    mse = trapezoid((average_survival_curve - km_curve) ** 2, unique_event_times)
    # normalize by the maximum time coordinate
    mse /= np.max(unique_event_times)

    if draw_figure:
        plt.plot(unique_event_times, average_survival_curve, label='Average Prediction Curve')
        plt.plot(unique_event_times, km_curve, label='KM Curve')
        plt.fill_between(unique_event_times, average_survival_curve, km_curve, alpha=0.2)
        score_text = r'KM-Calibration$= {:.3f}$'.format(mse)
        plt.plot([], [], ' ', label=score_text)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.show()

    return mse


if __name__ == '__main__':
    ### test the KM calibration

    # # first we define the time coordinates
    # times = np.linspace(0, 100, 11)
    # # then we define the survival probabilities at each time coordinate
    # survival_probabilities = np.exp(-times / 100)
    #
    #
    # # randomly generate the event time and event indicator for 20 samples
    # num_samples = 20
    # true_t = np.random.randint(0, 100, num_samples)
    # true_e = np.random.randint(0, 2, num_samples)
    #
    # # make some random predictions between 0 and 1
    # pred_at_observed_times = np.random.rand(num_samples)
    #
    # # # calculate the KM calibration score
    # # km_calibration_score = km_calibration(survival_probabilities, times, true_t, true_e, draw_figure=True)
    # # print(km_calibration_score)
    # # calculate the Cox-Snell residuals
    # coxsnell_residuals = residuals(pred_at_observed_times, true_e, method="Martingale", draw_figure=True)

    # test interval censored D-calibration
    pred_prob_left = np.array([0.84, 0.65, 0.42, 0.75, 1, 1])
    pred_prob_right = np.array([0.72, 0.63, 0.03, 0, 0.24, 0])
    pvalue, binning = d_cal_interval_cen(pred_prob_left, pred_prob_right, num_bins=10)