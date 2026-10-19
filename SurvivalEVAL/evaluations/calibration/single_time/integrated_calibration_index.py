"""Integrated calibration index and calibration curves at a specified time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from lifelines import CoxPHFitter
from patsy import dmatrix

from SurvivalEVAL.evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.evaluations.util import (
    check_and_convert_event_data,
    check_and_convert_probabilities,
)
from SurvivalEVAL.nonparametric_estimator.single_event import KaplanMeier


def _cloglog(probabilities: np.ndarray) -> np.ndarray:
    """
    Apply the complementary log-log transform with finite boundaries.

    Parameters
    ----------
    probabilities : ndarray
        Event probabilities in [0, 1].

    Returns
    -------
    ndarray
        Float64 values of log(-log(1 - p)), with the input shape preserved.
        Probabilities are clipped to [1e-10, 1 - 1e-10] after conversion to
        float64 so both boundaries remain finite for float32 inputs too.
        The input array is not modified.
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-10, 1 - 1e-10)
    return np.log(-np.log1p(-probabilities))


def integrated_calibration_index(
    preds: NumericArrayLike,
    event_time: NumericArrayLike,
    event_indicator: NumericArrayLike,
    target_time: Numeric,
    knots: int = 3,
    draw_figure: bool = False,
    figure_range: tuple | None = None,
) -> dict | tuple[dict, tuple[plt.Figure, plt.Axes]]:
    """
    Compute the Integrated Calibration Index (ICI) for a given set of predictions and true event times.

    The method is presented in [1]_. The implementation is based on the R code available in Appendix A of [1]_.

    Constant predictions use the Kaplan-Meier event probability at the target
    time. Other predictions use a spline and Cox model, clipping probabilities
    only for the log transform while retaining the original values for errors.

    Parameters
    ----------
    preds : NumericArrayLike
        The predicted probabilities of experiencing the event at the time of interest.
    event_time : NumericArrayLike
        The true event times.
    event_indicator : NumericArrayLike
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    target_time : Numeric
        The time of interest for calibration.
    knots : int
        The number of knots to use for the spline basis. Default is 3.
        Austin et al. (2020) [1]_ compared 3-5 knots and found that 3 knots is best.
    draw_figure : bool
        Whether to plot the graphical calibration curve and return the plot. Default is False.
    figure_range : tuple
        The range of the x-axis and y-axis for the plot.
        It should be a tuple of the form (x_min, x_max, y_min, y_max).
        If None, it will be set to the range of predicted event probabilities.
        Default is None.

    Returns
    -------
    summary : dict
            A dictionary containing the integrated calibration index (ICI), E50, E90, E_max, and the information about
        the calibration curve.
    fig : tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes objects for the calibration curve plot. Returned only if draw_figure is True.

    References
    ----------
    .. [1] Austin et al. (2020). "Graphical calibration curves and the integrated
       calibration index (ICI) for survival models". Statistics in Medicine, 39(21),
       2714-2742. https://doi.org/10.1002/sim.8570
    """
    preds = check_and_convert_probabilities(preds, "preds")
    event_time, event_indicator = check_and_convert_event_data(
        event_time, event_indicator
    )
    if preds.shape != event_time.shape:
        raise ValueError(
            "preds, event_time, and event_indicator must have the same shape."
        )
    grid = np.linspace(np.quantile(preds, 0.01), np.quantile(preds, 0.99), 100)
    pred_clls = _cloglog(preds)
    distinct_predictions = np.unique(pred_clls).size
    if distinct_predictions == 1:
        observed = 1 - KaplanMeier(event_time, event_indicator).predict(target_time)
        calibrated = np.full(preds.size + grid.size, observed)
    else:
        spline_df = min(knots, distinct_predictions - 1)
        spline = dmatrix(
            f"bs(x, df={spline_df}, degree={min(3, spline_df)}, include_intercept=False) - 1",
            {"x": pred_clls},
            return_type="dataframe",
        )
        data = spline.copy()
        data["time"] = event_time
        data["event"] = event_indicator
        fitter = CoxPHFitter().fit(data, duration_col="time", event_col="event")
        evaluation_spline = dmatrix(
            spline.design_info,
            {"x": _cloglog(np.concatenate([preds, grid]))},
            return_type="dataframe",
        )
        calibrated = (
            1
            - fitter.predict_survival_function(
                evaluation_spline, times=[target_time]
            ).values.flatten()
        )

    abs_err = np.abs(preds - calibrated[: preds.size])
    cal_pred = calibrated[preds.size :]
    summary = {
        "ICI": np.mean(abs_err, dtype=float),
        "E50": np.median(abs_err),
        "E90": np.quantile(abs_err, 0.9),
        "E_max": np.max(abs_err),
        "curve": {"grid": grid, "cal_pred": cal_pred},
    }

    if draw_figure:
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(grid, cal_pred, label="Calibration Curve", color="blue")
        ax.plot(grid, grid, label="Perfect Calibration", linestyle="--", color="grey")
        ax.set_xlabel("Predicted Event Probability")
        ax.set_ylabel("Observed Event Probability")
        ax.set_title("Graphical Calibration Curve")
        ax.legend()
        if figure_range is not None:
            ax.set_xlim(figure_range[0], figure_range[1])
            ax.set_ylim(figure_range[2], figure_range[3])
        return summary, (fig, ax)

    return summary
