"""Calibration of the average predicted survival curve against Kaplan-Meier."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import trapezoid

from SurvivalEVAL.Evaluations.util import interpolated_curve
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier


def km_calibration(
    average_survival_curve: np.ndarray,
    time_coordinates: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    interpolation_method: str = "Linear",
    draw_figure: bool = False,
) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
    """
    Calculate the KM calibration score between the average prediction curve and KM curve.

    The first version of KM calibration [1]_ is by visual inspection of the KM curve and the average curve.
    The second version of KM calibration [2]_ is by calculating the KL divergence between the KM curve and the average.
    This function actually calculates the (normalized) integrated mean squared error
    between the KM curve and the average prediction curve.

    This version has three benefits over [1]_ and [2]_:
    1. It is a single number, so it is easier to compare.
    2. This calculation is symmetric (note that KL-divergence is not).
    3. The score is between 0 and 1, where 0 means perfect calibration and 1 means worst calibration.
        And the random prediction curve will have a score of 0.25.

    Parameters
    ----------
    average_survival_curve : ndarray
        The average survival curves.
    time_coordinates : ndarray
        The time coordinates of the average survival curves.
    event_times : ndarray
        The event time of the test data.
    event_indicators : ndarray
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    interpolation_method : str, default "Linear"
        Interpolation method for evaluating the average survival curve.
        Options are "Linear" and "Pchip".
    draw_figure : bool
        Whether to visualize the comparison of the KM curve and average curve.

    Returns
    -------
    mse : float
        The (normalized) integrated mean squared error between the KM curve and the average prediction curve.
    fig : tuple(plt.Figure, plt.Axes)
        The matplotlib figure and axes objects for the calibration curve plot. Returned only if draw_figure
        is True.

    References
    ----------
    .. [1] Chapfuwa et al. (2020). "Calibration and Uncertainty in Neural
       Time-to-Event Modeling". IEEE Transactions on Neural Networks and Learning Systems.
       https://pmc.ncbi.nlm.nih.gov/articles/PMC8439415/
    .. [2] Yanagisawa (2023). "Proper Scoring Rules for Survival Analysis".
       Proceedings of Machine Learning Research, 202, 39165-39182.
       https://proceedings.mlr.press/v202/yanagisawa23a.html
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
    spline = interpolated_curve(
        time_coordinates, average_survival_curve, interpolation_method
    )
    average_survival_curve = spline(unique_event_times)
    average_survival_curve = np.clip(average_survival_curve, 0, 1)

    # integrated over the joint time coordinates
    mse = trapezoid((average_survival_curve - km_curve) ** 2, unique_event_times)
    # normalize by the maximum time coordinate
    mse /= np.max(unique_event_times)

    if draw_figure:
        fig, ax = plt.subplots(dpi=400)
        ax.plot(
            unique_event_times, average_survival_curve, label="Average Prediction Curve"
        )
        ax.plot(unique_event_times, km_curve, label="KM Curve")
        ax.fill_between(unique_event_times, average_survival_curve, km_curve, alpha=0.2)
        score_text = rf"KM-Calibration$= {mse:.3f}$"
        ax.plot([], [], " ", label=score_text)
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        plt.show()
        return mse, (fig, ax)

    return mse
