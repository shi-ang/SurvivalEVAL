"""Evaluation of cumulative incidence predictions for competing risks."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import PchipInterpolator

from SurvivalEVAL.evaluations.calibration import (
    aj_calibration,
    aj_calibration_per_event,
    cr_d_calibration,
    integrated_calibration_index_cr,
    one_calibration_cr,
)
from SurvivalEVAL.evaluations.calibration.single_time.util import (
    _calibration_curve_details,
)
from SurvivalEVAL.evaluations.custom_types import NumericArrayLike
from SurvivalEVAL.evaluations.discrimination import (
    auc_cr,
    auc_cr_per_event,
    concordance_cr,
    concordance_cr_per_event,
)
from SurvivalEVAL.evaluations.prediction_error import (
    brier_multiple_points_cr,
    brier_multiple_points_cr_per_event,
)
from SurvivalEVAL.evaluations.prediction_error.brier_score import (
    _brier_time_grid,
    _integrate_brier_scores,
)
from SurvivalEVAL.evaluations.util import (
    check_and_convert,
    check_and_convert_event_data,
    check_and_convert_probabilities,
    interpolated_curve,
    validate_time_point,
    validate_time_points,
    zero_padding,
)
from SurvivalEVAL.evaluations.util_plots import _plot_curves, _validate_plot_indices


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
        """
        Initialize the evaluator with CIF curves on a shared time grid.

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
            Matching 1D training outcomes, required for IPCW concordance, AUC, and Brier scores.
            Times must be finite and non-negative; labels are non-negative
            integers with 0 for censoring. Training times may extend beyond
            the prediction horizon. Supply both arrays together.
        """
        event_times, event_indicators = check_and_convert_event_data(
            event_times, event_indicators, competing_risks=True
        )
        if (train_event_times is None) != (train_event_indicators is None):
            raise ValueError("Supply both training event times and indicators.")
        if train_event_times is not None:
            train_event_times, train_event_indicators = check_and_convert_event_data(
                train_event_times, train_event_indicators, competing_risks=True
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
        """
        Validate and replace CIF curves and their shared grid together.

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
        pred_cifs = check_and_convert_probabilities(pred_cifs, "CIFs")
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
        """
        Read-only CIF curves; assignment validates against the current grid.

        Returns
        -------
        ndarray
            Stored cumulative incidence probabilities, with shape (n_samples, n_causes, n_times).
        """
        return self._pred_cifs

    @pred_cifs.setter
    def pred_cifs(self, value: NumericArrayLike) -> None:
        """
        Replace and validate the stored cumulative incidence curves.

        Parameters
        ----------
        value : NumericArrayLike
            Replacement cumulative incidence curves matching the stored time grid.
        """
        self.set_prediction_inputs(value, self.time_coordinates)

    @property
    def time_coordinates(self) -> np.ndarray:
        """
        Read-only shared time grid; assignment validates against current CIFs.

        Returns
        -------
        ndarray
            Stored time coordinates corresponding to the predicted curves.
        """
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, value: NumericArrayLike) -> None:
        """
        Replace and validate the time coordinates for the stored curves.

        Parameters
        ----------
        value : NumericArrayLike
            Replacement time grid matching the stored curves.
        """
        self.set_prediction_inputs(self.pred_cifs, value)

    @property
    def interpolation(self) -> str:
        """
        Interpolation method used for CIF probabilities and RMTL.

        Returns
        -------
        str
            Interpolation method used when evaluating curves between their time coordinates.
        """
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: str) -> None:
        """
        Set the curve interpolation method and invalidate cached predictions.

        Parameters
        ----------
        value : str
            Curve interpolation method, "Linear" or "Pchip".
        """
        if not isinstance(value, str) or value.lower() not in {"linear", "pchip"}:
            raise ValueError("interpolation must be 'Linear' or 'Pchip'.")
        self._interpolation = value.lower()

    @property
    def default_target_time(self) -> float:
        """
        Median observed time across the available training and test outcomes.

        Returns
        -------
        float
            Median observed time in the combined training and test outcomes, or the test outcomes alone.
        """
        times = self.event_times
        if self.train_event_times is not None:
            times = np.concatenate((times, self.train_event_times))
        return float(np.median(times))

    def predict_cifs(
        self,
        target_times: np.ndarray,
        event_of_interest: int | None = None,
    ) -> np.ndarray:
        """
        Predict cumulative incidence probabilities at the requested times.

        Parameters
        ----------
        target_times : ndarray
            One-dimensional array of finite, non-negative times
            within the prediction horizon. Order and repeated times are preserved.
        event_of_interest : int | None
            Integer cause label from 1 through n_causes.
            None returns predictions for all causes.

        Returns
        -------
        ndarray
            Array of shape (n_samples, n_causes, n_times), or (n_samples, n_times)
            when selecting one cause. Uses the evaluator's Linear or Pchip interpolation.

        Raises
        ------
        ValueError
            If times or the cause label are invalid.
        """
        target_times = validate_time_points(target_times, input_name="target_times")
        if not np.all(np.isfinite(target_times)) or np.any(
            target_times > self.time_coordinates[-1]
        ):
            raise ValueError(
                "target_times must be finite and within the prediction horizon."
            )
        cifs = self.pred_cifs
        if event_of_interest is not None:
            self._check_event_of_interest(event_of_interest)
            cifs = cifs[:, event_of_interest - 1, :]
        probabilities = np.empty((*cifs.shape[:-1], target_times.size), dtype=float)
        for index in np.ndindex(cifs.shape[:-1]):
            probabilities[index] = interpolated_curve(
                self.time_coordinates, cifs[index], self.interpolation
            )(target_times)
        np.clip(probabilities, 0, cifs[..., -1, None], out=probabilities)
        return probabilities

    def predict_rmtl(self, tau: float | None = None) -> np.ndarray:
        """
        Return cause-specific restricted mean time lost, integral_0^tau F_k(t).

        Parameters
        ----------
        tau : float, optional
            Finite, non-negative integration horizon within the prediction
            grid. Defaults to the final prediction time.

        Returns
        -------
        ndarray, shape (n_samples, n_causes)
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

    def predict_risks(
        self,
        predict_risk_method: str = "RMTL",
        tau: float | None = None,
        target_time: float | None = None,
        event_of_interest: int | None = None,
    ) -> np.ndarray:
        """
        Compute per-cause risk scores from RMTL or fixed-time CIF probabilities.

        Parameters
        ----------
        predict_risk_method : str
            "RMTL" (default) or "CIF", case-insensitive.
        tau : float | None
            Finite, non-negative RMTL integration horizon within the
            prediction grid. None uses the final prediction time. Unused for CIF.
        target_time : float | None
            Finite, non-negative prediction time within the grid.
            Required for CIF; must be omitted for RMTL.
        event_of_interest : int | None
            Integer cause label from 1 through n_causes.
            None returns risk scores for all causes.

        Returns
        -------
        ndarray
            Array of shape (n_samples, n_causes), or (n_samples,) when selecting
            one cause. Higher scores indicate greater risk. RMTL scores are
            areas under CIF curves; CIF scores are fixed-time probabilities.

        Raises
        ------
        ValueError
            If the method, horizon, or cause label is invalid.
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
            if event_of_interest is not None:
                self._check_event_of_interest(event_of_interest)
            risks = self.predict_rmtl(tau)
            return (
                risks if event_of_interest is None else risks[:, event_of_interest - 1]
            )
        if target_time is None:
            raise ValueError("target_time is required for predict_risk_method='CIF'.")
        target_time = validate_time_point(target_time, "target_time")
        return self.predict_cifs(np.array([target_time]), event_of_interest)[..., 0]

    def concordance(
        self,
        ties: str = "Risk",
        method: str = "Naive",
        control_definition: str = "NotCase",
        tau: float | None = None,
        reduction: str = "mean",
        return_details: bool = False,
        *,
        predict_risk_method: str = "RMTL",
        target_time: float | None = None,
    ) -> float | tuple[float, dict]:
        """
        Calculate competing-risks concordance across all predicted causes.

        The ``NotCase`` definition follows Wolbers et al. [1]_. ``EventFree``
        restricts controls to subjects still event-free at the anchor time.

        Parameters
        ----------
        ties : {"None", "Time", "Risk", "All"}, default="Risk"
            Tie policy from ``concordance_cr_per_event``. Risk ties receive
            half credit by default; selected-cause event-time ties are excluded.
        method : {"Naive", "IPCW"}, default="Naive"
            Unit pair weights or inverse censoring weights estimated from
            the training outcomes supplied at initialization.
        control_definition : {"NotCase", "EventFree"}, default="NotCase"
            NotCase includes competing events before or at each anchor time
            as controls; EventFree excludes them. Both include later subjects
            and censorings exactly at the anchor. Case-insensitive.
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

        References
        ----------
        .. [1] Wolbers, M.; Blanche, P.; Koller, M. T.; Witteman, J. C. M.;
           Gerds, T. A. (2014). "Concordance for prognostic models with competing risks".
           Biostatistics, 15(3), 526-539. https://doi.org/10.1093/biostatistics/kxt059
        """
        risks = self.predict_risks(predict_risk_method, tau, target_time)
        return concordance_cr(
            predicted_risks=risks,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            control_definition=control_definition,
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
        control_definition: str = "NotCase",
        tau: float | None = None,
        *,
        predict_risk_method: str = "RMTL",
        target_time: float | None = None,
    ) -> tuple[float, float, float]:
        """
        Calculate concordance for one cause using positive RMTL by default.

        The ``NotCase`` definition follows Wolbers et al. [1]_. ``EventFree``
        restricts controls to subjects still event-free at the anchor time.

        Parameters
        ----------
        event_of_interest : int, default=1
            Integer cause label from 1 through n_causes.
        ties : {"None", "Time", "Risk", "All"}, default="Risk"
            Tie policy; see ``concordance``.
        method : {"Naive", "IPCW"}, default="Naive"
            Unit pair weights or training-based inverse censoring weights.
        control_definition : {"NotCase", "EventFree"}, default="NotCase"
            Control definition at each selected-cause anchor time; see
            ``concordance``. Independent of the prediction target_time.
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

        References
        ----------
        .. [1] Wolbers, M.; Blanche, P.; Koller, M. T.; Witteman, J. C. M.;
           Gerds, T. A. (2014). "Concordance for prognostic models with competing risks".
           Biostatistics, 15(3), 526-539. https://doi.org/10.1093/biostatistics/kxt059
        """
        self._check_event_of_interest(event_of_interest)
        risks = self.predict_risks(
            predict_risk_method,
            tau,
            target_time,
            event_of_interest=event_of_interest,
        )
        return concordance_cr_per_event(
            predicted_risks=risks,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            control_definition=control_definition,
            ties=ties,
            tau=tau,
            event_of_interest=event_of_interest,
        )

    def auc(
        self,
        target_time: float | None = None,
        method: str = "Naive",
        control_definition: str = "NotCase",
        return_details: bool = False,
    ) -> float | tuple[float, dict]:
        """
        Calculate the equal-weight mean of AUCs from cause-specific CIFs.

        The competing-risk AUROC definitions and IPCW estimation follow [1]_.

        Parameters
        ----------
        target_time : float | None
            Finite, non-negative horizon within the prediction grid.
            Defaults to the median of available training and test outcomes.
        method : str
            "Naive" (default) or "IPCW", case-insensitive. IPCW requires
            training outcomes and assumes independent censoring.
        control_definition : str
            "NotCase" (default) includes competing events
            by the target as controls; "EventFree" excludes them. Both
            include subjects observed after the target and censorings
            exactly at the target. Names are case-insensitive.
        return_details : bool
            Also return per_event_scores, keyed by cause label.

        Returns
        -------
        float | tuple[float, dict]
            Mean AUC, optionally with per-cause details. Undefined causes have
            NaN scores and make the mean NaN; they are not silently omitted.

        References
        ----------
        .. [1] Blanche, P.; Dartigues, J.-F.; Jacqmin-Gadda, H. (2013). "Estimating
           and comparing time-dependent areas under receiver operating characteristic
           curves for censored event times with competing risks".
           Statistics in Medicine, 32(30), 5381-5397. https://doi.org/10.1002/sim.5958
        """
        if target_time is None:
            target_time = self.default_target_time
        probabilities = self.predict_risks("CIF", None, target_time)
        return auc_cr(
            predict_probs=probabilities,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            target_time=target_time,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            control_definition=control_definition,
            return_details=return_details,
        )

    def auc_per_event(
        self,
        event_of_interest: int = 1,
        target_time: float | None = None,
        method: str = "Naive",
        control_definition: str = "NotCase",
    ) -> float:
        """
        Calculate AUC for one cause using its CIF probability at the target.

        The competing-risk AUROC definitions and IPCW estimation follow [1]_.

        Parameters
        ----------
        event_of_interest : int
            Integer cause label from 1 through n_causes.
        target_time : float | None
            Evaluation horizon, with the same default as auc.
        method : str
            "Naive" (default) or "IPCW", case-insensitive.
        control_definition : str
            "NotCase" (default) or "EventFree"
            case-insensitive. See auc_cr_per_event for definitions and weights.

        Returns
        -------
        float
            Selected cause's AUC, with half credit for exactly tied predictions.

        Raises
        ------
        ValueError
            If the cause is invalid, cases or controls are absent,
            or inputs do not satisfy auc_cr_per_event's requirements.

        References
        ----------
        .. [1] Blanche, P.; Dartigues, J.-F.; Jacqmin-Gadda, H. (2013). "Estimating
           and comparing time-dependent areas under receiver operating characteristic
           curves for censored event times with competing risks".
           Statistics in Medicine, 32(30), 5381-5397. https://doi.org/10.1002/sim.5958
        """
        self._check_event_of_interest(event_of_interest)
        if target_time is None:
            target_time = self.default_target_time
        probabilities = self.predict_risks(
            "CIF", None, target_time, event_of_interest=event_of_interest
        )
        return auc_cr_per_event(
            predict_probs=probabilities,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            target_time=target_time,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            event_of_interest=event_of_interest,
            control_definition=control_definition,
        )

    def auroc(
        self,
        target_time: float | None = None,
        method: str = "Naive",
        control_definition: str = "NotCase",
        return_details: bool = False,
    ) -> float | tuple[float, dict]:
        """
        Alias for auc, with the same arguments, defaults and return values.

        The competing-risk AUROC definitions and IPCW estimation follow [1]_.

        Parameters
        ----------
        target_time : float | None
            Non-negative evaluation horizon; None selects the evaluator's default target time.
        method : str
            Scoring method, "Naive" or "IPCW", case-insensitive; single-event AUROC also accepts "Uno".
        control_definition : str
            "EventFree" excludes competing events from controls; "NotCase" includes them.
        return_details : bool
            Whether to return diagnostic details along with the score.

        Returns
        -------
        float | tuple[float, dict]
            AUROC at the target time; when return_details is True, return (score, per-cause details).

        References
        ----------
        .. [1] Blanche, P.; Dartigues, J.-F.; Jacqmin-Gadda, H. (2013). "Estimating
           and comparing time-dependent areas under receiver operating characteristic
           curves for censored event times with competing risks".
           Statistics in Medicine, 32(30), 5381-5397. https://doi.org/10.1002/sim.5958
        """
        return self.auc(target_time, method, control_definition, return_details)

    def auroc_per_event(
        self,
        event_of_interest: int = 1,
        target_time: float | None = None,
        method: str = "Naive",
        control_definition: str = "NotCase",
    ) -> float:
        """
        Alias for auc_per_event, with the same arguments and return value.

        The competing-risk AUROC definitions and IPCW estimation follow [1]_.

        Parameters
        ----------
        event_of_interest : int
            Positive cause label to evaluate; None requests the mean across all predicted causes.
        target_time : float | None
            Non-negative evaluation horizon; None selects the evaluator's default target time.
        method : str
            Scoring method, "Naive" or "IPCW", case-insensitive; single-event AUROC also accepts "Uno".
        control_definition : str
            "EventFree" excludes competing events from controls; "NotCase" includes them.

        Returns
        -------
        float
            AUROC for the selected event cause at the target time.

        References
        ----------
        .. [1] Blanche, P.; Dartigues, J.-F.; Jacqmin-Gadda, H. (2013). "Estimating
           and comparing time-dependent areas under receiver operating characteristic
           curves for censored event times with competing risks".
           Statistics in Medicine, 32(30), 5381-5397. https://doi.org/10.1002/sim.5958
        """
        return self.auc_per_event(
            event_of_interest, target_time, method, control_definition
        )

    def brier_score(
        self,
        target_time: float | None = None,
        IPCW_weighted: bool = True,
    ) -> float:
        """
        Calculate the mean cause-specific Brier score at one time.

        Each component uses the competing-risk prediction error in [1]_.
        The evaluator averages these scores equally across predicted causes.

        Parameters
        ----------
        target_time : float | None
            Finite time within the CIF horizon. None uses
            default_target_time, the median available train/test observed time.
        IPCW_weighted : bool
            Use training-based IPCW. Defaults to True. Fully
            observed test data automatically use plain scoring.

        Returns
        -------
        float
            Equal-weight mean across every predicted cause.

        References
        ----------
        .. [1] Schoop, R.; Beyersmann, J.; Schumacher, M.; Binder, H. (2011).
           "Quantifying the predictive accuracy of time-to-event models in the
           presence of competing risks". Biometrical Journal, 53(1), 88-112.
           https://doi.org/10.1002/bimj.201000073
        """
        if target_time is None:
            target_time = self.default_target_time
        target_time = validate_time_point(target_time, "target_time")
        return float(
            self.brier_score_multiple_points(np.array([target_time]), IPCW_weighted)[0]
        )

    def brier_score_per_event(
        self,
        event_of_interest: int = 1,
        target_time: float | None = None,
        IPCW_weighted: bool = True,
    ) -> float:
        """
        Calculate a cause-specific Brier score at one time.

        The competing-risk prediction error follows [2]_, extending the IPCW
        approach in [1]_.

        Parameters
        ----------
        event_of_interest : int
            Integer cause label from 1 through n_causes.
        target_time : float | None
            Finite time within the CIF horizon. None uses
            default_target_time, the median available train/test observed time.
        IPCW_weighted : bool
            Use training-based IPCW, defaulting to True.
            Fully observed test data automatically use plain scoring.

        Returns
        -------
        float
            Brier score for the selected cause, including competing-event
            negative outcomes. An absent cause still has a defined score.

        References
        ----------
        .. [1] Gerds, T. A.; Schumacher, M. (2006). "Consistent Estimation of the
           Expected Brier Score in General Survival Models with Right-Censored
           Event Times". Biometrical Journal, 48(6), 1029-1040.
           https://doi.org/10.1002/bimj.200610301
        .. [2] Schoop, R.; Beyersmann, J.; Schumacher, M.; Binder, H. (2011).
           "Quantifying the predictive accuracy of time-to-event models in the
           presence of competing risks". Biometrical Journal, 53(1), 88-112.
           https://doi.org/10.1002/bimj.201000073
        """
        if target_time is None:
            target_time = self.default_target_time
        target_time = validate_time_point(target_time, "target_time")
        return float(
            self.brier_score_multiple_points_per_event(
                np.array([target_time]), event_of_interest, IPCW_weighted
            )[0]
        )

    def brier_score_multiple_points(
        self,
        target_times: np.ndarray,
        IPCW_weighted: bool = True,
    ) -> np.ndarray:
        """
        Calculate mean cause-specific Brier scores at multiple times.

        Each component uses the competing-risk prediction error in [1]_.
        The evaluator averages these scores equally across predicted causes.

        Parameters
        ----------
        target_times : ndarray
            Finite, non-negative times within the CIF horizon.
            Scores follow this order, including repeated times.
        IPCW_weighted : bool
            Use training-based IPCW, defaulting to True.
            Fully observed test data automatically use plain scoring.

        Returns
        -------
        ndarray
            A vector of equal-weight means across all predicted causes. Earlier
            censorings contribute zero; the denominator is the full test size.
            IPCW fits one censoring model for all causes and times.

        References
        ----------
        .. [1] Schoop, R.; Beyersmann, J.; Schumacher, M.; Binder, H. (2011).
           "Quantifying the predictive accuracy of time-to-event models in the
           presence of competing risks". Biometrical Journal, 53(1), 88-112.
           https://doi.org/10.1002/bimj.201000073
        """
        return self._brier_score_multiple_points(target_times, IPCW_weighted, None)

    def brier_score_multiple_points_per_event(
        self,
        target_times: np.ndarray,
        event_of_interest: int = 1,
        IPCW_weighted: bool = True,
    ) -> np.ndarray:
        """
        Calculate a selected cause's Brier scores at multiple times.

        The competing-risk prediction error follows [2]_, extending the IPCW
        approach in [1]_.

        Parameters
        ----------
        target_times : ndarray
            Finite, non-negative times within the CIF horizon.
        event_of_interest : int
            Integer cause label from 1 through n_causes.
        IPCW_weighted : bool
            Use training-based IPCW, defaulting to True.
            Fully observed test data automatically use plain scoring.

        Returns
        -------
        ndarray
            One score per target time, including competing events as negative
            outcomes. Earlier censorings contribute zero; divide by full test size.

        References
        ----------
        .. [1] Gerds, T. A.; Schumacher, M. (2006). "Consistent Estimation of the
           Expected Brier Score in General Survival Models with Right-Censored
           Event Times". Biometrical Journal, 48(6), 1029-1040.
           https://doi.org/10.1002/bimj.200610301
        .. [2] Schoop, R.; Beyersmann, J.; Schumacher, M.; Binder, H. (2011).
           "Quantifying the predictive accuracy of time-to-event models in the
           presence of competing risks". Biometrical Journal, 53(1), 88-112.
           https://doi.org/10.1002/bimj.201000073
        """
        self._check_event_of_interest(event_of_interest)
        return self._brier_score_multiple_points(
            target_times, IPCW_weighted, event_of_interest
        )

    def integrated_brier_score(
        self,
        num_points: int | None = None,
        target_times: np.ndarray | None = None,
        IPCW_weighted: bool = True,
        integration_method: str = "trapz",
        draw_figure: bool = False,
    ) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
        """
        Integrate the mean cause-specific Brier score over a time grid.

        This integrates the equal-weight mean of the cause-specific prediction
        errors in [1]_ over the selected time interval.

        Parameters
        ----------
        num_points : int | None
            Generate this many times from zero through the maximum
            train/test observed time. Mutually exclusive with target_times.
        target_times : ndarray | None
            Explicit integration grid within the CIF horizon.
            Unsorted times are sorted with a warning. If neither grid option
            is supplied, use unique test censoring times; at least two are needed.
        IPCW_weighted : bool
            Use training-based IPCW, defaulting to True.
            Fully observed test data automatically use plain scoring.
        integration_method : str
            Case-insensitive 'trapz' (default) or 'simpson'.
        draw_figure : bool
            Also return the Brier curve figure and axes.

        Returns
        -------
        float | tuple[float, tuple[plt.Figure, plt.Axes]]
            The integral divided by the time span, or (ibs, (fig, ax)) when
            draw_figure is True. Every predicted cause receives equal weight.

        References
        ----------
        .. [1] Schoop, R.; Beyersmann, J.; Schumacher, M.; Binder, H. (2011).
           "Quantifying the predictive accuracy of time-to-event models in the
           presence of competing risks". Biometrical Journal, 53(1), 88-112.
           https://doi.org/10.1002/bimj.201000073
        """
        return self._integrated_brier_score(
            None,
            num_points,
            target_times,
            IPCW_weighted,
            integration_method,
            draw_figure,
        )

    def integrated_brier_score_per_event(
        self,
        event_of_interest: int = 1,
        num_points: int | None = None,
        target_times: np.ndarray | None = None,
        IPCW_weighted: bool = True,
        integration_method: str = "trapz",
        draw_figure: bool = False,
    ) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
        """
        Integrate a cause-specific Brier score over a time grid.

        This integrates the competing-risk prediction error in [2]_ over the
        selected time interval, using the IPCW approach in [1]_ when enabled.

        Parameters
        ----------
        event_of_interest : int
            Integer cause label from 1 through n_causes.
        num_points : int | None
            Generate times from zero through the maximum train/test
            observed time. Mutually exclusive with target_times.
        target_times : ndarray | None
            Explicit grid within the CIF horizon, sorted with a
            warning if needed. With neither grid option, infer unique test
            censoring times; at least two are needed.
        IPCW_weighted : bool
            Use training-based IPCW, defaulting to True.
            Fully observed test data automatically use plain scoring.
        integration_method : str
            Case-insensitive 'trapz' (default) or 'simpson'.
        draw_figure : bool
            Also return the Brier curve figure and axes.

        Returns
        -------
        float | tuple[float, tuple[plt.Figure, plt.Axes]]
            The cause-specific integral divided by the grid's time span, or
            (ibs, (fig, ax)) when draw_figure is True.

        References
        ----------
        .. [1] Gerds, T. A.; Schumacher, M. (2006). "Consistent Estimation of the
           Expected Brier Score in General Survival Models with Right-Censored
           Event Times". Biometrical Journal, 48(6), 1029-1040.
           https://doi.org/10.1002/bimj.200610301
        .. [2] Schoop, R.; Beyersmann, J.; Schumacher, M.; Binder, H. (2011).
           "Quantifying the predictive accuracy of time-to-event models in the
           presence of competing risks". Biometrical Journal, 53(1), 88-112.
           https://doi.org/10.1002/bimj.201000073
        """
        self._check_event_of_interest(event_of_interest)
        return self._integrated_brier_score(
            event_of_interest,
            num_points,
            target_times,
            IPCW_weighted,
            integration_method,
            draw_figure,
        )

    def one_calibration(
        self,
        target_time: float | None = None,
        event_of_interest: int = 1,
        num_bins: int = 10,
        binning_strategy: str = "C",
        return_details: bool = False,
        adjust_variance: bool = True,
    ) -> tuple[float, list, list] | tuple[float, dict]:
        """
        Test grouped predicted CIFs against observed incidence at one time.

        The Greenwood-type Aalen-Johansen variance used when
        ``adjust_variance=True`` is described in [1]_.

        Parameters
        ----------
        target_time : float | None
            Finite, non-negative time within the prediction grid.
            None uses default_target_time, the median available observed time.
        event_of_interest : int
            Integer cause label from 1 through n_causes.
        num_bins : int
            Positive integer number of groups. Empty groups are omitted.
        binning_strategy : str
            "C" (default) for approximately equal-sized risk
            groups, or "H" for equal-width probability groups; case-insensitive.
        return_details : bool
            Return (p_value, details), with observed/expected probabilities
            anchored least-squares slope and intercept, maximum local deviation,
            statistic, variances, bin sizes, degrees of freedom, and histogram_plot
            and pp_plot figure/axes pairs. Figures are not shown.
        adjust_variance : bool
            Use Greenwood-type AJ variance and G degrees of
            freedom, defaulting to True. False uses expected binomial variance
            and the single-event DN degrees-of-freedom convention.

        Returns
        -------
        tuple[float, list, list] | tuple[float, dict]
            (p_value, observed, expected), or (p_value, details). Observed
            probabilities are within-group Aalen–Johansen estimates for the
            selected cause, including events at target_time. The adjusted test
            is a grouped Wald test with asymptotic p-values, assuming independent
            censoring within groups and an independently trained model. P-values
            are NaN if any adjusted variance is zero or no groups remain.

        References
        ----------
        .. [1] Allignol, Schumacher and Beyersmann (2010). "A Note on Variance Estimation
           of the Aalen-Johansen Estimator ...". Biometrical Journal, 52(1), 126-137.
           https://doi.org/10.1002/bimj.200900039
        """
        self._check_event_of_interest(event_of_interest)
        if target_time is None:
            target_time = self.default_target_time
        probabilities = self.predict_risks(
            "CIF", target_time=target_time, event_of_interest=event_of_interest
        )
        p_value, details = one_calibration_cr(
            probabilities,
            self.event_times,
            self.event_indicators,
            target_time,
            event_of_interest,
            num_bins,
            binning_strategy,
            adjust_variance=adjust_variance,
            return_details=True,
        )
        if return_details:
            details.update(
                _calibration_curve_details(
                    details["observed_probabilities"], details["expected_probabilities"]
                )
            )
            return p_value, details
        return (
            p_value,
            details["observed_probabilities"],
            details["expected_probabilities"],
        )

    def integrated_calibration_index(
        self,
        target_time: float | None = None,
        event_of_interest: int = 1,
        knots: int = 3,
        draw_figure: bool = True,
        figure_range: tuple | None = None,
    ) -> dict | tuple[dict, tuple[plt.Figure, plt.Axes]]:
        """
        Estimate a selected cause's smoothed calibration curve and ICI.

        The Fine-Gray calibration model follows Austin et al. [1]_. It uses
        evaluation outcomes to estimate censoring weights; training outcomes
        are not needed for the fit.

        Parameters
        ----------
        target_time : float | None, default=None
            Finite, non-negative time within the CIF prediction grid.
            None uses default_target_time.
        event_of_interest : int, default=1
            Integer cause label from 1 through n_causes.
        knots : int, default=3
            Total restricted cubic spline knot count, at least three.
            Nonconstant risks must support distinct knots and a full-rank fit.
        draw_figure : bool, default=True
            Return a calibration plot without displaying it.
        figure_range : tuple, optional
            Plot limits (x_min, x_max, y_min, y_max).

        Returns
        -------
        dict or tuple[dict, tuple[plt.Figure, plt.Axes]]
            ICI, E50, E90, E_max, and curve data; optionally figure and axes.
            An unobserved cause has a zero calibrated curve. Constant risks
            use the selected-cause Aalen-Johansen incidence estimate.

        References
        ----------
        .. [1] Austin et al. (2022). "Graphical calibration curves and the
           integrated calibration index (ICI) for competing risk models".
           Diagnostic and Prognostic Research, 6, 2.
           https://doi.org/10.1186/s41512-021-00114-6
        """
        self._check_event_of_interest(event_of_interest)
        if target_time is None:
            target_time = self.default_target_time
        probabilities = self.predict_risks(
            "CIF", target_time=target_time, event_of_interest=event_of_interest
        )
        return integrated_calibration_index_cr(
            probabilities,
            self.event_times,
            self.event_indicators,
            target_time,
            event_of_interest=event_of_interest,
            knots=knots,
            draw_figure=draw_figure,
            figure_range=figure_range,
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
        """
        Calculate CR-D-calibration from interpolated CIF predictions.

        CR-D-calibration follows [1]_. The ``sum`` reduction matches the
        paper's definition; ``mean`` and ``max`` provide alternative reductions.

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

        References
        ----------
        .. [1] Alberge, J.; Haugomat, T.; Varoquaux, G.; Abécassis, J. (2026).
           "On the Calibration of Survival Models with Competing Risks".
           Proceedings of the 29th International Conference on Artificial Intelligence
           and Statistics, Proceedings of Machine Learning Research, 300, 1837-1845.
           https://proceedings.mlr.press/v300/alberge26a.html
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

    def aj_calibration(
        self,
        alpha: float = 2,
        reduction: str = "mean",
        min_prop_at_risk: float = 0.05,
        return_details: bool = False,
    ) -> float | tuple[float, dict]:
        """
        Compare mean predicted CIFs with marginal Aalen–Johansen estimates.

        AJ calibration of competing-risk survival models is described in [1]_.

        Parameters
        ----------
        alpha : float
            Finite exponent at least 1. The default gives time-averaged
            squared error; 1 gives time-averaged absolute error.
        reduction : str
            Case-insensitive 'mean' (default), 'sum', or 'max' of the
            per-cause errors. Includes all predicted causes, even if absent.
        min_prop_at_risk : float
            Retain prediction times where at least this
            fraction of subjects has an observed time greater than or equal
            to the horizon. Defaults to 0.05; 0 uses the full grid.
        return_details : bool
            Include per_event_scores, retained time_coordinates
            mean_pred_cifs, and aj_cifs with shape (n_causes, n_times).

        Returns
        -------
        float | tuple[float, dict]
            Calibration error, or (score, details). Lower is better. Integrates
            absolute curve differences raised to alpha with the trapezoidal rule
            on the retained prediction grid, divided by its final time. Requires
            at least two retained times and independent censoring for marginal
            AJ estimation. This is a marginal error, not a hypothesis-test p-value.

        References
        ----------
        .. [1] Alberge, J.; Haugomat, T.; Varoquaux, G.; Abécassis, J. (2026).
           "On the Calibration of Survival Models with Competing Risks".
           Proceedings of the 29th International Conference on Artificial Intelligence
           and Statistics, Proceedings of Machine Learning Research, 300, 1837-1845.
           https://proceedings.mlr.press/v300/alberge26a.html
        """
        return aj_calibration(
            self.pred_cifs,
            self.time_coordinates,
            self.event_times,
            self.event_indicators,
            alpha=alpha,
            reduction=reduction,
            min_prop_at_risk=min_prop_at_risk,
            return_details=return_details,
        )

    def aj_calibration_per_event(
        self,
        event_of_interest: int = 1,
        alpha: float = 2,
        min_prop_at_risk: float = 0.05,
        draw_figure: bool = False,
    ) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
        """
        Compare one cause's mean predicted CIF with its AJ reference curve.

        AJ calibration follows [1]_; [2]_ provides the reference implementation.

        Parameters
        ----------
        event_of_interest : int
            Integer cause label from 1 through n_causes.
        alpha : float
            Finite exponent at least 1; defaults to squared error.
        min_prop_at_risk : float
            Minimum observed fraction at risk at retained
            prediction times. Defaults to 0.05; 0 uses the full grid.
        draw_figure : bool
            Also return the comparison figure and axes, without
            displaying the plot. Curves are joined on the retained grid.

        Returns
        -------
        float | tuple[float, tuple[plt.Figure, plt.Axes]]
            The time-normalized calibration error, or (score, (fig, ax)) when
            draw_figure is True, matching km_calibration's plotting convention.
            Uses the same integration and censoring assumptions as aj_calibration.
            Competing events remain observed events in the reference estimate.

        References
        ----------
        .. [1] Alberge, J.; Haugomat, T.; Varoquaux, G.; Abécassis, J. (2026).
           "On the Calibration of Survival Models with Competing Risks".
           Proceedings of the 29th International Conference on Artificial Intelligence
           and Statistics, Proceedings of Machine Learning Research, 300, 1837-1845.
           https://proceedings.mlr.press/v300/alberge26a.html
        .. [2] Hazardous reference implementation.
           https://github.com/soda-inria/hazardous/blob/main/hazardous/metrics/_aj_calibration.py
        """
        self._check_event_of_interest(event_of_interest)
        result = aj_calibration_per_event(
            self.pred_cifs[:, event_of_interest - 1],
            self.time_coordinates,
            self.event_times,
            self.event_indicators,
            event_of_interest,
            alpha=alpha,
            min_prop_at_risk=min_prop_at_risk,
            return_details=draw_figure,
        )
        if not draw_figure:
            return result
        score, details = result
        fig, ax = _plot_curves(
            details["time_coordinates"],
            np.stack((details["mean_pred_cifs"], details["aj_cifs"])),
            ["Average prediction", "Aalen–Johansen estimate"],
            title=f"AJ calibration for cause {event_of_interest} = {score:.3f}",
            x_label="Time",
            y_label="Cumulative incidence",
            y_lim=(0, 1.02),
        )
        ax.fill_between(
            details["time_coordinates"],
            details["mean_pred_cifs"],
            details["aj_cifs"],
            alpha=0.2,
        )
        return score, (fig, ax)

    def plot_cif_curves(
        self,
        curve_indices: int | list[int],
        event_of_interest: int | list[int] | None = None,
        color: str | list[str] | None = None,
        x_lim: float | tuple[float, float] | None = None,
        y_lim: float | tuple[float, float] | None = None,
        x_label: str = "Time",
        y_label: str = "Cumulative incidence",
        *,
        ax: plt.Axes | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot selected samples' cause-specific cumulative incidence curves.

        Parameters
        ----------
        curve_indices : int or list of int
            Zero-based sample indices, plotted in the supplied order.
        event_of_interest : int or list of int, optional
            One-based cause labels. None includes all predicted causes.
        color : str or list of str, optional
            One shared color or one color per plotted sample/cause combination,
            ordered by sample, then cause. By default, colors identify causes
            and line styles identify samples, cycling through four styles.
        x_lim, y_lim : float or tuple of float, optional
            Axis limits. Probabilities default to [0, 1.02].
        x_label, y_label : str
            Axis labels, defaulting to Time and Cumulative incidence.
        ax : matplotlib.axes.Axes, optional
            Existing axes to reuse. Otherwise create a new figure.
        **kwargs : Any
            Matplotlib line options. An explicit linestyle overrides the
            sample styles; label overrides the generated sample/cause labels.

        Returns
        -------
        fig, ax : tuple
            Figure and axes, without displaying the plot. Stored CIF values
            are connected directly, without interpolation or extrapolation.
        """
        indices = _validate_plot_indices(curve_indices, self.pred_cifs.shape[0])
        if event_of_interest is None:
            causes = np.arange(self.pred_cifs.shape[1])
        else:
            causes = (
                _validate_plot_indices(
                    event_of_interest,
                    self.pred_cifs.shape[1],
                    "event_of_interest",
                    one_based=True,
                )
                - 1
            )
        curves = self.pred_cifs[indices[:, None], causes[None, :], :].reshape(
            -1, self.time_coordinates.size
        )
        labels = [f"Sample {i} · Cause {k + 1}" for i in indices for k in causes]
        if color is None:
            color = [f"C{k % 10}" for k in causes] * indices.size
        styles = ("-", "--", ":", "-.")
        line_styles = [
            styles[i % len(styles)] for i in range(indices.size) for _ in causes
        ]
        return _plot_curves(
            self.time_coordinates,
            curves,
            labels,
            title="Predicted cumulative incidence",
            x_label=x_label,
            y_label=y_label,
            color=color,
            x_lim=(0, float(self.time_coordinates[-1])) if x_lim is None else x_lim,
            y_lim=(0, 1.02) if y_lim is None else y_lim,
            ax=ax,
            line_styles=line_styles,
            **kwargs,
        )

    def plot_survival_curves(
        self,
        curve_indices: int | list[int],
        color: str | list[str] | None = None,
        x_lim: float | tuple[float, float] | None = None,
        y_lim: float | tuple[float, float] | None = None,
        x_label: str = "Time",
        y_label: str = "Survival probability",
        *,
        ax: plt.Axes | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot overall event-free survival, 1 minus the sum of cause-specific CIFs.

        Parameters
        ----------
        curve_indices : int or list of int
            Zero-based sample indices, plotted in the supplied order.
        color : str or list of str, optional
            One shared color or one color per selected curve.
        x_lim, y_lim : float or tuple of float, optional
            Axis limits. Survival probabilities default to [0, 1.02].
        x_label, y_label : str
            Axis labels, defaulting to Time and Survival probability.
        ax : matplotlib.axes.Axes, optional
            Existing axes to reuse. Otherwise create a new figure.
        **kwargs : Any
            Matplotlib line options, including label overrides.

        Returns
        -------
        fig, ax : tuple
            Figure and axes, without displaying the plot. Predictions are
            connected on their stored grid without extrapolation.
        """
        indices = _validate_plot_indices(curve_indices, self.pred_cifs.shape[0])
        survs = np.clip(1 - self.pred_cifs[indices].sum(axis=1, dtype=float), 0, 1)
        return _plot_curves(
            self.time_coordinates,
            survs,
            [f"Sample {i}" for i in indices],
            title="Predicted event-free survival",
            x_label=x_label,
            y_label=y_label,
            color=color,
            x_lim=(0, float(self.time_coordinates[-1])) if x_lim is None else x_lim,
            y_lim=(0, 1.02) if y_lim is None else y_lim,
            ax=ax,
            **kwargs,
        )

    def _check_event_of_interest(self, event_of_interest: int) -> None:
        """
        Require a cause label from 1 through the number of predicted causes.

        Parameters
        ----------
        event_of_interest : int
            Positive cause label to evaluate; None requests the mean across all predicted causes.
        """
        if (
            isinstance(event_of_interest, (bool, np.bool_))
            or not isinstance(event_of_interest, (int, np.integer))
            or not 1 <= event_of_interest <= self.pred_cifs.shape[1]
        ):
            raise ValueError("event_of_interest must be an integer from 1 to n_causes.")

    def _integrated_brier_score(
        self,
        event_of_interest: int | None,
        num_points: int | None,
        target_times: np.ndarray | None,
        ipcw: bool,
        integration_method: str,
        draw_figure: bool,
    ) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
        """
        Resolve the common grid, evaluate once, then integrate and optionally plot.

        Parameters
        ----------
        event_of_interest : int | None
            Positive cause label to evaluate; None requests the mean across all predicted causes.
        num_points : int | None
            Number of equally spaced evaluation times; mutually exclusive with target_times.
        target_times : ndarray | None
            One-dimensional array of non-negative evaluation times.
        ipcw : bool
            Whether to apply inverse probability of censoring weights estimated from the training outcomes.
        integration_method : str
            Numerical integration rule, "trapz" or "simpson".
        draw_figure : bool
            Whether to create a Matplotlib figure and return it with the score.

        Returns
        -------
        float | tuple[float, tuple[plt.Figure, plt.Axes]]
            Integrated Brier score; when draw_figure is True, return (score, (figure, axes)).
        """
        target_times = _brier_time_grid(
            self.event_times,
            self.event_indicators,
            self.train_event_times,
            num_points,
            target_times,
        )
        scores = self._brier_score_multiple_points(
            target_times, ipcw, event_of_interest
        )
        ibs = _integrate_brier_scores(scores, target_times, integration_method)
        if not draw_figure:
            return ibs
        fig, ax = _plot_curves(
            target_times,
            scores,
            [f"IBS = {ibs:.3f}"],
            title="Integrated Brier Score",
            x_label="Time",
            y_label="Brier Score",
            marker="o",
        )
        return ibs, (fig, ax)

    def _brier_score_multiple_points(
        self,
        target_times: np.ndarray,
        ipcw: bool,
        event_of_interest: int | None,
    ) -> np.ndarray:
        """
        Predict CIFs and dispatch to aggregate or selected-cause scoring.

        Parameters
        ----------
        target_times : ndarray
            One-dimensional array of non-negative evaluation times.
        ipcw : bool
            Whether to apply inverse probability of censoring weights estimated from the training outcomes.
        event_of_interest : int | None
            Positive cause label to evaluate; None requests the mean across all predicted causes.

        Returns
        -------
        ndarray
            Mean Brier score across causes, or the selected-cause score, at each target time.
        """
        probabilities = self.predict_cifs(target_times, event_of_interest)
        if np.all(self.event_indicators > 0):
            ipcw = False
        metric = (
            brier_multiple_points_cr
            if event_of_interest is None
            else brier_multiple_points_cr_per_event
        )
        options = (
            {}
            if event_of_interest is None
            else {"event_of_interest": event_of_interest}
        )
        return metric(
            pred_mat=probabilities,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            target_times=target_times,
            ipcw=ipcw,
            **options,
        )
