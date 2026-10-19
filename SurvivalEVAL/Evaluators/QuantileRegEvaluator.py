"""Evaluation of predicted event-time quantiles."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from SurvivalEVAL.Evaluations.Calibration import km_calibration
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike
from SurvivalEVAL.Evaluations.util import check_and_convert, quantile_to_survival

from .SurvivalEvaluator import SurvivalEvaluator


class QuantileRegEvaluator(SurvivalEvaluator):
    def __init__(
        self,
        pred_regs: NumericArrayLike,
        quantile_levels: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ) -> None:
        """
        Initialize the quantile regression evaluator.

        Parameters
        ----------
        pred_regs : NumericArrayLike
            Accept shapes: (n_quantiles,) or (n_samples, n_quantiles).
            Predicted event-time quantiles for the testing samples. Each column
            corresponds to the matching entry in `quantile_levels`.
            At least one of `pred_regs` or `quantile_levels` must be a 2D array.
        quantile_levels : NumericArrayLike
            Accept shapes: (n_quantiles,) or (n_samples, n_quantiles).
            Cumulative event-probability levels corresponding to `pred_regs`.
            At least one of `pred_regs` or `quantile_levels` must be a 2D array.
        event_times : NumericArrayLike, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        event_indicators : NumericArrayLike, shape = (n_samples, )
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        train_event_times : NumericArrayLike, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        train_event_indicators : NumericArrayLike, shape = (n_train_samples, )
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        predict_time_method : str, default="Median"
            Method for calculating predicted survival time. Available options are "Median", "Mean" or "RMST".
        interpolation : str, default="Linear"
            Method for interpolation. Available options are ['Linear', 'Pchip'].
        """
        quantile_levels = check_and_convert(quantile_levels)
        survival_level = 1 - quantile_levels
        if predict_time_method == "RMST":
            raise NotImplementedError(
                "RMST prediction time method is not implemented for quantile curves."
            )
        super().__init__(
            survival_level,
            pred_regs,
            event_times,
            event_indicators,
            train_event_times,
            train_event_indicators,
            predict_time_method,
            interpolation,
        )

    def km_calibration(
        self,
        draw_figure: bool = False,
    ) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
        """
        Calculate the KM calibration score from the predicted survival curve.

        Related marginal-calibration diagnostics compare survival curves
        visually [1]_ or through KL divergence [2]_. This method uses
        integrated squared error.

        Parameters
        ----------
        draw_figure : bool, default=False
            Whether to draw the figure of the KM calibration.

        Returns
        -------
        float or tuple[float, tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]
            Normalized integrated squared error between the mean predicted survival curve and the
            Kaplan-Meier estimate. When draw_figure is True, return (score, (figure, axes)).

        References
        ----------
        .. [1] Chapfuwa et al. (2020). "Calibration and Uncertainty in Neural
           Time-to-Event Modeling". IEEE Transactions on Neural Networks and Learning Systems.
           https://pmc.ncbi.nlm.nih.gov/articles/PMC8439415/
        .. [2] Yanagisawa (2023). "Proper Scoring Rules for Survival Analysis".
           Proceedings of Machine Learning Research, 202, 39165-39182.
           https://proceedings.mlr.press/v202/yanagisawa23a.html
        """
        unique_times = np.unique(self.event_times[self.event_indicators == 1])
        survival_curves = quantile_to_survival(
            1 - self._pred_survs,
            self.time_coordinates,
            unique_times,
            interpolate=self.interpolation,
        )
        avg_surv = np.mean(survival_curves, axis=0, dtype=float)

        return km_calibration(
            average_survival_curve=avg_surv,
            time_coordinates=unique_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            interpolation_method=self.interpolation,
            draw_figure=draw_figure,
        )
