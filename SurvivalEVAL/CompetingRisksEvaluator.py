"""Evaluation of cumulative incidence predictions for competing risks."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator

from SurvivalEVAL.Evaluations.Calibration import cr_d_calibration
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike
from SurvivalEVAL.Evaluations.Discrimination import (
    concordance_cr,
    concordance_cr_per_event,
)
from SurvivalEVAL.Evaluations.util import (
    check_and_convert,
    interpolated_curve,
    validate_time_point,
    zero_padding,
)


class CompetingRisksEvaluator:
    """Evaluate competing-risks CIF curves against observed times and causes."""

    def __init__(
        self,
        pred_cifs: NumericArrayLike,
        time_coordinates: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        interpolation: str = "Linear",
        *,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
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
        train_event_times, train_event_indicators : NumericArrayLike, optional
            Matching 1D training outcomes, required for IPCW concordance.
            Times must be finite and non-negative; labels are non-negative
            integers with 0 for censoring. Training times may extend beyond
            the prediction horizon. Supply both arrays together.
        """
        event_times, event_indicators = check_and_convert(event_times, event_indicators)
        if event_times.ndim != 1:
            raise ValueError("Event arrays must have one entry per predicted sample.")
        if not np.all(np.isfinite(event_times)) or np.any(event_times < 0):
            raise ValueError(
                "Event times must be finite, non-negative and within the prediction horizon."
            )
        if (
            not np.all(np.isfinite(event_indicators))
            or np.any(event_indicators != np.floor(event_indicators))
            or np.any(event_indicators < 0)
        ):
            raise ValueError(
                "event_indicators must contain integer cause labels from 0 to n_causes."
            )
        if (train_event_times is None) != (train_event_indicators is None):
            raise ValueError("Supply both training event times and indicators.")
        if train_event_times is not None:
            train_event_times, train_event_indicators = check_and_convert(
                train_event_times, train_event_indicators
            )
            if train_event_times.ndim != 1:
                raise ValueError("Training event arrays must be 1D.")
            if not np.all(np.isfinite(train_event_times)) or np.any(
                train_event_times < 0
            ):
                raise ValueError(
                    "Training event times must be finite and non-negative."
                )
            if (
                not np.all(np.isfinite(train_event_indicators))
                or np.any(train_event_indicators < 0)
                or np.any(train_event_indicators != np.floor(train_event_indicators))
            ):
                raise ValueError(
                    "Training indicators must be non-negative integer labels."
                )

        self.event_times = event_times
        self.event_indicators = event_indicators
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators
        self.set_prediction_inputs(pred_cifs, time_coordinates)
        self.interpolation = interpolation

    def set_prediction_inputs(
        self,
        pred_cifs: NumericArrayLike,
        time_coordinates: NumericArrayLike,
    ) -> None:
        """Validate and replace CIF curves and their shared grid together.

        Parameters
        ----------
        pred_cifs : NumericArrayLike, shape (n_samples, n_causes, n_time_points)
            Nondecreasing CIFs in [0, 1], summing to at most one across causes.
        time_coordinates : NumericArrayLike, shape (n_time_points,)
            Finite, non-negative, strictly increasing times covering the test
            outcomes. A zero point is prepended
            if necessary, with zero CIFs.

        Notes
        -----
        Stored prediction arrays are read-only. Use this method or the property
        setters to update predictions. Invalid updates preserve the current
        inputs.
        """
        pred_cifs = check_and_convert(pred_cifs)
        time_coordinates = check_and_convert(time_coordinates)
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
        if pred_cifs.shape[0] != self.event_times.size:
            raise ValueError("Event arrays must have one entry per predicted sample.")
        if (
            not np.all(np.isfinite(time_coordinates))
            or np.any(time_coordinates < 0)
            or np.any(np.diff(time_coordinates) <= 0)
        ):
            raise ValueError(
                "time_coordinates must be finite, non-negative and strictly increasing."
            )
        if np.any(self.event_times > time_coordinates[-1]):
            raise ValueError("Event times must be within the prediction horizon.")
        if np.any(self.event_indicators > pred_cifs.shape[1]):
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
        pred_cifs, time_coordinates = zero_padding(
            pred_cifs, time_coordinates, initial_value=0.0
        )
        # Own writable inputs to preserve the validated curve/grid invariants.
        if pred_cifs.flags.writeable:
            pred_cifs = pred_cifs.copy()
            pred_cifs.flags.writeable = False
        if time_coordinates.flags.writeable:
            time_coordinates = time_coordinates.copy()
            time_coordinates.flags.writeable = False
        self._pred_cifs = pred_cifs
        self._time_coordinates = time_coordinates

    @property
    def pred_cifs(self) -> np.ndarray:
        """Read-only CIF curves; assignment validates against the current grid."""
        return self._pred_cifs

    @pred_cifs.setter
    def pred_cifs(self, value: NumericArrayLike) -> None:
        self.set_prediction_inputs(value, self.time_coordinates)

    @property
    def time_coordinates(self) -> np.ndarray:
        """Read-only shared time grid; assignment validates against current CIFs."""
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, value: NumericArrayLike) -> None:
        self.set_prediction_inputs(self.pred_cifs, value)

    @property
    def interpolation(self) -> str:
        """Interpolation method used for CIF probabilities and RMTL."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: str) -> None:
        if not isinstance(value, str) or value.lower() not in {"linear", "pchip"}:
            raise ValueError("interpolation must be 'Linear' or 'Pchip'.")
        self._interpolation = value.lower()

    def predict_rmtl(self, tau: float | None = None) -> np.ndarray:
        """Return cause-specific restricted mean time lost, integral_0^tau F_k(t).

        Parameters
        ----------
        tau : float, optional
            Finite, non-negative integration horizon within the prediction
            grid. Defaults to the final prediction time.

        Returns
        -------
        np.ndarray, shape (n_samples, n_causes)
            Areas under the CIF curves, in time units. Larger values indicate
            greater risk, so concordance uses these values without negation.
            Integration follows the evaluator's Linear or Pchip interpolation.
        """
        horizon = (
            float(self.time_coordinates[-1])
            if tau is None
            else validate_time_point(tau, "tau")
        )
        if not np.isfinite(horizon) or horizon > self.time_coordinates[-1]:
            raise ValueError("tau must be finite and within the prediction horizon.")

        if self.interpolation == "pchip":
            return np.array(
                [
                    PchipInterpolator(self.time_coordinates, curves, axis=-1).integrate(
                        0, horizon
                    )
                    for curves in self.pred_cifs
                ]
            )

        times = self.time_coordinates.astype(float, copy=False)
        stop = np.searchsorted(times, horizon, side="right")
        curves = self.pred_cifs[..., :stop].astype(float, copy=False)
        areas = trapezoid(curves, times[:stop], axis=-1)
        if stop < times.size and horizon > times[stop - 1]:
            width = horizon - times[stop - 1]
            left = curves[..., -1]
            right = self.pred_cifs[..., stop]
            endpoint = left + (right - left) * width / (times[stop] - times[stop - 1])
            areas += (left + endpoint) * width / 2
        return areas

    def _predict_risks(
        self,
        predict_risk_method: str,
        tau: float | None,
        target_time: float | None,
    ) -> np.ndarray:
        """Compute per-cause risk scores from RMTL or fixed-time CIF probabilities.

        RMTL integrates each CIF to tau (or the final prediction time).
        CIF evaluates each curve at an explicit target_time. Prediction horizons
        must be finite and within the grid. Both choices return an array of
        shape (n_samples, n_causes), with higher scores indicating greater risk.
        """
        risk_method = (
            predict_risk_method.lower()
            if isinstance(predict_risk_method, str)
            else None
        )
        if risk_method not in {"rmtl", "cif"}:
            raise ValueError("predict_risk_method must be 'RMTL' or 'CIF'.")
        if risk_method == "rmtl":
            if target_time is not None:
                raise ValueError(
                    "target_time is only used with predict_risk_method='CIF'."
                )
            return self.predict_rmtl(tau)
        if target_time is None:
            raise ValueError("target_time is required for predict_risk_method='CIF'.")
        target_time = validate_time_point(target_time, "target_time")
        if not np.isfinite(target_time) or target_time > self.time_coordinates[-1]:
            raise ValueError(
                "target_time must be finite and within the prediction horizon."
            )
        risks = np.empty(self.pred_cifs.shape[:2], dtype=float)
        for i, curves in enumerate(self.pred_cifs):
            for k, curve in enumerate(curves):
                risks[i, k] = interpolated_curve(
                    self.time_coordinates, curve, self.interpolation
                )(target_time)
        return np.clip(risks, 0, self.pred_cifs[..., -1])

    def concordance(
        self,
        ties: str = "Risk",
        method: str = "Naive",
        tau: float | None = None,
        reduction: str = "mean",
        return_details: bool = False,
        *,
        predict_risk_method: str = "RMTL",
        target_time: float | None = None,
    ) -> float | tuple[float, dict]:
        """Calculate competing-risks concordance across all predicted causes.

        Parameters
        ----------
        ties : {"None", "Time", "Risk", "All"}, default="Risk"
            Tie policy from ``concordance_cr_per_event``. Risk ties receive
            half credit by default; selected-cause event-time ties are excluded.
        method : {"Naive", "IPCW"}, default="Naive"
            Unit pair weights or inverse censoring weights estimated from
            the training outcomes supplied at initialization.
        tau : float, optional
            Only selected-cause anchors with event time strictly below tau
            contribute. Also sets the RMTL integration horizon, which must lie
            within the prediction grid. None uses the final prediction time
            for RMTL and applies no pair truncation. For CIF, tau only
            truncates pairs.
        reduction : {"mean", "pooled"}, default="mean"
            Equal-weight mean of per-cause scores or pooling of pair counts.
            The mean is NaN if any cause has undefined concordance.
        return_details : bool, default=False
            Also return per-cause scores, concordant counts and total counts,
            keyed by cause label, as in ``concordance_cr``.
        predict_risk_method : {"RMTL", "CIF"}, default="RMTL"
            Positive area under each CIF up to tau, or CIF probability at
            target_time. Computed from the evaluator's CIF curves on each call.
        target_time : float, optional
            Shared finite, non-negative prediction time within the grid.
            Required for CIF; must be omitted for RMTL.

        Returns
        -------
        c_index : float
            Aggregated competing-risks concordance.
        details : dict, optional
            Per-cause scores and counts; counts are weighted for IPCW.
        """
        risks = self._predict_risks(predict_risk_method, tau, target_time)
        return concordance_cr(
            predicted_risks=risks,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties,
            tau=tau,
            reduction=reduction,
            return_details=return_details,
        )

    def concordance_per_event(
        self,
        event_of_interest: int = 1,
        ties: str = "Risk",
        method: str = "Naive",
        tau: float | None = None,
        *,
        predict_risk_method: str = "RMTL",
        target_time: float | None = None,
    ) -> tuple[float, float, float]:
        """Calculate concordance for one cause using positive RMTL by default.

        Parameters
        ----------
        event_of_interest : int, default=1
            Integer cause label from 1 through n_causes.
        ties : {"None", "Time", "Risk", "All"}, default="Risk"
            Tie policy; see ``concordance``.
        method : {"Naive", "IPCW"}, default="Naive"
            Unit pair weights or training-based inverse censoring weights.
        tau : float, optional
            Strict anchor cutoff and RMTL integration horizon. None uses the
            final prediction time for RMTL and applies no pair truncation.
            For CIF, tau only truncates pairs.
        predict_risk_method : {"RMTL", "CIF"}, default="RMTL"
            Positive RMTL up to tau or CIF probability at target_time, computed
            from the evaluator's CIF curves on each call.
        target_time : float, optional
            Required for CIF and omitted for RMTL. Must be finite,
            non-negative and within the prediction grid.

        Returns
        -------
        c_index, num_concordant_pairs, num_total_pairs : tuple of float
            Per-cause concordance and pair counts, including half credit for
            retained ties. Counts are weighted for IPCW.
        """
        if (
            isinstance(event_of_interest, (bool, np.bool_))
            or not isinstance(event_of_interest, (int, np.integer))
            or not 1 <= event_of_interest <= self.pred_cifs.shape[1]
        ):
            raise ValueError("event_of_interest must be an integer from 1 to n_causes.")
        risks = self._predict_risks(predict_risk_method, tau, target_time)[
            :, event_of_interest - 1
        ]
        return concordance_cr_per_event(
            predicted_risks=risks,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties,
            tau=tau,
            event_of_interest=event_of_interest,
        )

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
