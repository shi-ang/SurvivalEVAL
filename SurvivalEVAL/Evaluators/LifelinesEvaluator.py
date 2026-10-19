"""Adapter for lifelines survival predictions."""

from __future__ import annotations

from abc import ABC

import pandas as pd

from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike

from .PycoxEvaluator import PycoxEvaluator


class LifelinesEvaluator(PycoxEvaluator, ABC):
    def __init__(
        self,
        surv: pd.DataFrame,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ) -> None:
        """
        Evaluator for survival models in Lifelines packages.

        Parameters
        ----------
        surv : pd.DataFrame, shape = (n_time_points, n_samples)
            Predicted survival curves for the testing samples
        event_times : NumericArrayLike, shape = (n_samples,)
            Event times for the testing samples.
        event_indicators : NumericArrayLike, shape = (n_samples,)
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        train_event_times : NumericArrayLike, shape = (n_samples,), optional
            Event times for the training samples.
        train_event_indicators : NumericArrayLike, shape = (n_samples,), optional
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        predict_time_method : str, default="Median"
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean" and "RMST".
        interpolation : str, default="Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        super().__init__(
            surv,
            event_times,
            event_indicators,
            train_event_times,
            train_event_indicators,
            predict_time_method,
            interpolation,
        )
