"""Survival residuals and their calibration diagnostics."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from SurvivalEVAL.NonparametricEstimator.SingleEvent import NelsonAalen

_residual_names = {
    "coxsnell": "Cox-Snell Residuals",
    "modified coxsnell-v1": "Cox-Snell Residuals",
    "modified coxsnell-v2": "Cox-Snell Residuals",
    "martingale": "Martingale Residuals",
    "deviance": "Deviance Residuals",
}


def residuals(
    pred_probs: np.ndarray,
    event_indicators: np.ndarray,
    method: str = "CoxSnell",
    draw_figure: bool = False,
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
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
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
    cox_residuals = -np.log(pred_probs)
    event_indicators = event_indicators.astype(bool, copy=False)
    method = method.lower()

    if method == "coxsnell":
        residuals = cox_residuals
    elif method in ["modified coxsnell-v1", "modified coxsnell-v2"]:
        # Compare with standard CoxSnell residuals, this method adds an 'excess residual' for censored instances.
        # The excess residual should also follow a unit exponential distribution, based on the lack of memory property.
        # There are two choices of excess residuals:
        # (1) use the mean of the unit exponential distribution, which is 1,
        # or (2) use the median of the unit exponential distribution, which is ln(2).
        excess_residual = 1 if method == "modified coxsnell-v1" else np.log(2)
        residuals = cox_residuals + excess_residual * (~event_indicators)
    elif method == "martingale":
        residuals = event_indicators - cox_residuals
    elif method == "deviance":

        def safe_log(x):
            return np.log(x + 1e-8)

        martingale_res = event_indicators - cox_residuals
        # get the sign of the martingale residuals
        sign = np.sign(martingale_res)
        residuals = sign * np.sqrt(
            -2 * (martingale_res + event_indicators * safe_log(cox_residuals))
        )
    else:
        raise ValueError(f"Unknown method {method}")

    if draw_figure:
        cum_haz_empirical = NelsonAalen(cox_residuals, event_indicators)
        max_res = np.max(cox_residuals)
        _fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True, dpi=400)
        ax[0].plot(
            cum_haz_empirical.survival_times,
            cum_haz_empirical.cumulative_hazard,
            label="Empirical",
        )
        ax[0].plot(
            [0, max_res], [0, max_res], label="Ideal", linestyle="--", color="red"
        )
        ax[0].legend()
        ax[0].set_xlabel("Cox-Snell Residuals")
        ax[0].set_ylabel("Cumlative hazard of residuals")

        # use solid scatter points for uncensored instances and hollow scatter points for censored instances
        idx = np.arange(len(residuals))
        ax[1].scatter(
            idx[event_indicators],
            residuals[event_indicators],
            alpha=0.5,
            color="red",
            label="Event",
        )
        ax[1].scatter(
            idx[~event_indicators],
            residuals[~event_indicators],
            facecolors="none",
            edgecolors="red",
            alpha=0.5,
            label="Censored",
        )
        ax[1].set_xlabel("Index")
        ax[1].set_ylabel(_residual_names[method])
        ax[1].legend()
        plt.show()
    return residuals
