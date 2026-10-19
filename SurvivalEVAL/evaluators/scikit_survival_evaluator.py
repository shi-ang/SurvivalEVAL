"""Adapter for scikit-survival predictions."""

from __future__ import annotations

from abc import ABC

import numpy as np

from SurvivalEVAL.evaluations.custom_types import NumericArrayLike

from .survival_evaluator import SurvivalEvaluator


class ScikitSurvivalEvaluator(SurvivalEvaluator, ABC):
    def __init__(
        self,
        surv: np.ndarray,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ) -> None:
        """
        Evaluator for survival models in scikit-survival packages.

        Parameters
        ----------
        surv : ndarray, shape = (n_samples,)
            Predicted survival curves for the testing samples from scikit-survival model.
            Each element is a scikit-survival customized object.
            '.x' attribute is the time coordinates for the given curve. '.y' attribute is the survival probabilities.
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
            The method used to calculate the predicted event time. Options: "Median" (default), "Mean", and "RMST".
        interpolation : str, default="Linear"
            The interpolation method used to calculate the predicted event time.
            Options: "Linear" (default), "Pchip".
        """
        time_coordinates = surv[0].x
        predict_curves = []
        for i in range(len(surv)):
            predict_curve = surv[i].y
            if False in (time_coordinates == surv[i].x):
                raise KeyError(
                    f"{i}-th survival curve does not have same time coordinates"
                )
            predict_curves.append(predict_curve)
        predicted_curves = np.array(predict_curves)
        # Ensure flat all-one curves have a finite extrapolated event time.
        end_probabilities = predicted_curves[:, -1]
        idx_need_fix = end_probabilities == 1
        if np.any(idx_need_fix):
            reference_end_probabilities = end_probabilities[~idx_need_fix]
            if reference_end_probabilities.size == 0:
                replacement_probability = 0.99
            else:
                max_prob_at_end = np.max(reference_end_probabilities)
                replacement_probability = max(0.1 * max_prob_at_end + 0.9, 0.99)
            predicted_curves[idx_need_fix, -1] = replacement_probability
        super().__init__(
            predicted_curves,
            time_coordinates,
            event_times,
            event_indicators,
            train_event_times,
            train_event_indicators,
            predict_time_method,
            interpolation,
        )
