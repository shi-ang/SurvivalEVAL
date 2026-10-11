"""Evaluation of cumulative incidence predictions for competing risks."""

from __future__ import annotations

import numpy as np

from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike
from SurvivalEVAL.Evaluations.DistributionCalibration import cr_d_calibration
from SurvivalEVAL.Evaluations.util import check_and_convert, interpolated_curve


class CompetingRisksEvaluator:
    """Evaluate competing-risks CIF curves against observed times and causes."""

    def __init__(
        self,
        pred_cifs: NumericArrayLike,
        time_coordinates: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        interpolation: str = "Linear",
    ) -> None:
        """Initialize the evaluator with CIF curves on a shared time grid.

        Parameters
        ----------
        pred_cifs : NumericArrayLike, shape (n_samples, n_causes, n_time_points)
            Nondecreasing cumulative incidence curves in [0, 1]. Cause k is
            stored at index k - 1 on the second axis. CIFs across causes must
            sum to at most one at every time point.
        time_coordinates : NumericArrayLike, shape (n_time_points,)
            Finite, non-negative, strictly increasing times shared by all
            curves. At least two points are required. If the grid starts
            after zero, a time-zero point with zero CIFs is prepended.
        event_times : NumericArrayLike, shape (n_samples,)
            Finite, non-negative event/censor times within the prediction
            horizon. Predictions are interpolated at these times.
        event_indicators : NumericArrayLike, shape (n_samples,)
            Cause labels: 0 for censoring and 1 through n_causes for events.
        interpolation : {"Linear", "Pchip"}, default="Linear"
            Interpolation method, matching the survival evaluator.
        """
        pred_cifs = check_and_convert(pred_cifs)
        time_coordinates = check_and_convert(time_coordinates)
        event_times, event_indicators = check_and_convert(event_times, event_indicators)
        if pred_cifs.ndim != 3:
            raise ValueError(
                "pred_cifs must have shape (n_samples, n_causes, n_time_points)."
            )
        if time_coordinates.ndim != 1 or time_coordinates.size < 2:
            raise ValueError(
                "time_coordinates must be a 1D array with at least two points."
            )
        if pred_cifs.shape[2] != time_coordinates.size:
            raise ValueError("The CIF time axis must match time_coordinates.")
        if event_times.ndim != 1 or pred_cifs.shape[0] != event_times.size:
            raise ValueError("Event arrays must have one entry per predicted sample.")
        if (
            not np.all(np.isfinite(time_coordinates))
            or np.any(time_coordinates < 0)
            or np.any(np.diff(time_coordinates) <= 0)
        ):
            raise ValueError(
                "time_coordinates must be finite, non-negative and strictly increasing."
            )
        if (
            not np.all(np.isfinite(event_times))
            or np.any(event_times < 0)
            or np.any(event_times > time_coordinates[-1])
        ):
            raise ValueError(
                "Event times must be finite, non-negative and within the prediction horizon."
            )
        if (
            not np.all(np.isfinite(event_indicators))
            or np.any(event_indicators != np.floor(event_indicators))
            or np.any((event_indicators < 0) | (event_indicators > pred_cifs.shape[1]))
        ):
            raise ValueError(
                "event_indicators must contain integer cause labels from 0 to n_causes."
            )
        if not np.all(np.isfinite(pred_cifs)) or np.any(
            (pred_cifs < 0) | (pred_cifs > 1)
        ):
            raise ValueError("CIFs must contain finite probabilities in [0, 1].")
        if np.any(np.diff(pred_cifs, axis=2) < 0):
            raise ValueError("CIF curves must be nondecreasing.")
        if np.any(pred_cifs.sum(axis=1) > 1 + 1e-6):
            raise ValueError("CIFs across causes must sum to at most one.")
        interpolation = interpolation.lower()
        if interpolation not in ("linear", "pchip"):
            raise ValueError("interpolation must be 'Linear' or 'Pchip'.")

        if time_coordinates[0] > 0:
            time_coordinates = np.concatenate(([0], time_coordinates))
            pred_cifs = np.concatenate(
                (np.zeros((*pred_cifs.shape[:2], 1), dtype=pred_cifs.dtype), pred_cifs),
                axis=2,
            )
        self.pred_cifs = pred_cifs
        self.time_coordinates = time_coordinates
        self.event_times = event_times
        self.event_indicators = event_indicators
        self.interpolation = interpolation

    def cr_d_calibration(
        self,
        num_bins: int = 100,
        alpha: float = 2,
        method: str = "integral",
        reduction: str = "sum",
        return_details: bool = False,
        *,
        pred_event_probs: NumericArrayLike | None = None,
    ) -> float | tuple[float, dict]:
        """Calculate CR-D-calibration from interpolated CIF predictions.

        Parameters
        ----------
        num_bins : int, default=100
            Number of equal-width intervals in the calibration risk grid.
        alpha : float, default=2
            Finite exponent at least 1 for the integral norm. `alpha=1`: area under 
            the absolute-error curve; errors contribute linearly. 
            `alpha=2` (default): root integrated squared error; 
            Ignored for KS.
        method : {"integral", "ks"}, default="integral"
            Integral norm or maximum grid deviation from the identity curve.
        reduction : {"sum", "mean", "max"}, default="sum"
            Aggregation of per-cause scores. Sum follows the paper's definition.
        return_details : bool, default=False
            Include per-cause scores and calibration curves. KS details also
            include heuristic p-values and Bonferroni adjustments.
        pred_event_probs : NumericArrayLike, shape (n_samples, n_causes), optional
            Eventual cause probabilities F_k(infinity), summing to one per
            subject. Defaults to the final CIF values, which require a long
            enough prediction horizon that remaining survival is negligible.
            Supply these probabilities explicitly for a shorter horizon.

        Returns
        -------
        score : float
            Aggregated calibration error. Lower is better.
        details : dict, optional
            The same diagnostic information as the standalone metric.
        """
        cifs = np.empty(self.pred_cifs.shape[:2], dtype=float)
        for i, time in enumerate(self.event_times):
            for k, curve in enumerate(self.pred_cifs[i]):
                cifs[i, k] = interpolated_curve(
                    self.time_coordinates, curve, self.interpolation
                )(time)
        # Bound interpolation roundoff by the valid CIF endpoints.
        cifs = np.clip(cifs, 0, self.pred_cifs[:, :, -1])
        survs = np.clip(1 - cifs.sum(axis=1), 0, 1)
        if pred_event_probs is None:
            pred_event_probs = self.pred_cifs[:, :, -1]
        return cr_d_calibration(
            pred_cifs=cifs,
            event_indicators=self.event_indicators,
            pred_event_probs=pred_event_probs,
            pred_survs=survs,
            num_bins=num_bins,
            alpha=alpha,
            method=method,
            reduction=reduction,
            return_details=return_details,
        )
