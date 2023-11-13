import numpy as np
import matplotlib.pyplot as plt

from Evaluations.util import KaplanMeier


def km_calibration(
        average_survival_curves: np.ndarray,
        time_coordinates: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
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
    average_survival_curves: np.ndarray
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
    km_model = KaplanMeier(event_times, event_indicators)
    km_curve = km_model.survival_probabilities
    km_time = km_model.survival_times

    # add 0 to the beginning of the time coordinates, and 1 to the beginning of the average_survival_curves
    km_time = np.concatenate([[0], km_time])
    km_curve = np.concatenate([[1], km_curve])
    if time_coordinates[0] != 0:
        time_coordinates = np.concatenate([[0], time_coordinates])
        average_survival_curves = np.concatenate([[1], average_survival_curves])

    # interpolate the average curve and km_curve, so that they have the same time coordinates
    joint_time_coordinates = np.unique(np.concatenate([time_coordinates, km_time]))
    # interpolate the average curve and km_curve, using linear interpolation
    average_survival_curves = np.interp(joint_time_coordinates, time_coordinates, average_survival_curves)
    km_curve = np.interp(joint_time_coordinates, km_time, km_curve)

    # integrated over the joint time coordinates
    mse = np.trapz((average_survival_curves - km_curve) ** 2, joint_time_coordinates)
    # normalize by the maximum time coordinate
    mse /= np.max(joint_time_coordinates)

    if draw_figure:
        plt.plot(joint_time_coordinates, average_survival_curves, label='Average Prediction Curve')
        plt.plot(joint_time_coordinates, km_curve, label='KM Curve')
        plt.fill_between(joint_time_coordinates, average_survival_curves, km_curve, alpha=0.2)
        score_text = r'KM-Calibration$= {:.3f}$'.format(mse)
        plt.plot([], [], ' ', label=score_text)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.legend()
        plt.show()

    return mse


if __name__ == '__main__':
    # test the KM calibration

    # first we define the time coordinates
    times = np.linspace(0, 100, 11)
    # then we define the survival probabilities at each time coordinate
    survival_probabilities = np.exp(-times / 100)

    # randomly generate the event time and event indicator for 20 samples
    num_samples = 20
    true_t = np.random.randint(0, 100, num_samples)
    true_e = np.random.randint(0, 2, num_samples)

    # calculate the KM calibration score
    km_calibration_score = km_calibration(survival_probabilities, times, true_t, true_e, draw_figure=True)
    print(km_calibration_score)
