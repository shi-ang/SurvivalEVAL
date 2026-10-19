"""Evaluation of survival probabilities at a single time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from SurvivalEVAL.evaluations.calibration import (
    integrated_calibration_index,
    one_calibration,
)
from SurvivalEVAL.evaluations.calibration.single_time.util import (
    _calibration_curve_details,
)
from SurvivalEVAL.evaluations.custom_types import NumericArrayLike
from SurvivalEVAL.evaluations.discrimination import auc
from SurvivalEVAL.evaluations.prediction_error import brier_multiple_points
from SurvivalEVAL.evaluations.util import (
    check_and_convert,
    check_and_convert_event_data,
)


class SingleTimeEvaluator:
    def __init__(
        self,
        pred_probs: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        target_time: float | None = None,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
    ) -> None:
        """
        Initialize the SingleTimeEvaluator with predictions and observed outcomes.

        Parameters
        ----------
        pred_probs : NumericArrayLike, shape = (n_samples, )
            Predicted survival probability at the target time for the testing samples.
        event_times : NumericArrayLike, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        event_indicators : NumericArrayLike, shape = (n_samples, )
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        target_time : float, int, or None, default=None
            Time point at which the evaluation is to be performed. If None, the target time is set to the median time
        train_event_times : NumericArrayLike, shape = (n_train_samples, )
            Actual event/censor time for the training samples.
        train_event_indicators : NumericArrayLike, shape = (n_train_samples, )
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        """
        self.event_times, self.event_indicators = check_and_convert_event_data(
            event_times, event_indicators
        )
        self.pred_probs = pred_probs

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert_event_data(
                train_event_times, train_event_indicators
            )
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        if target_time is None:
            # set to the median time of all the event/censor times from the training and test sets
            # if train set is not provided, use test set only
            event_times = (
                np.concatenate((self.event_times, self.train_event_times))
                if self.train_event_times is not None
                else self.event_times
            )
            target_time = np.quantile(event_times, 0.5)
        self.target_time = target_time
        self._NO_CENSOR = np.all(self.event_indicators == 1)

    @property
    def pred_probs(self) -> np.ndarray:
        """
        Return the stored single-time survival probabilities.

        Returns
        -------
        ndarray
            Stored predicted survival probabilities with shape (n_samples,).
        """
        return self._pred_probs

    @pred_probs.setter
    def pred_probs(self, pred_probs: NumericArrayLike) -> None:
        """
        Replace the stored single-time survival probabilities.

        Parameters
        ----------
        pred_probs : NumericArrayLike
            Predicted survival probabilities at the target time, one per testing sample.
        """
        pred_probs = check_and_convert(pred_probs)
        if pred_probs.ndim != 1:
            raise ValueError(
                "predicted_probs should be a 1D array-like object, "
                f"but got a {pred_probs.ndim}D array-like object"
            )
        if pred_probs.shape != self.event_times.shape:
            raise ValueError("pred_probs and event_times must have the same shape.")
        self._pred_probs = pred_probs

    def auc(self, method: str = "Naive") -> float:
        """
        Calculate cumulative/dynamic AUROC at this evaluator's target time.

        Cases are observed events at or before the target. Controls are
        observations after the target and censorings exactly at the target.
        Earlier censorings are excluded.

        The cumulative/dynamic AUROC definition follows [1]_. For ``IPCW``
        and its alias ``Uno``, censoring-weighted estimation follows [2]_.

        Parameters
        ----------
        method : str
            "Naive" (default), "IPCW", or its alias "Uno", case-insensitive. IPCW requires
            training outcomes, assumes independent censoring, and weights
            cases by 1/G(event_time-) from reverse Kaplan-Meier.

        Returns
        -------
        float
            AUROC at the target time.

        Raises
        ------
        TypeError
            If IPCW is requested without training outcomes.
        ValueError
            If inputs or the method are invalid, either class is
            absent, or IPCW cannot be estimated at the target.

        References
        ----------
        .. [1] Heagerty, P. J.; Lumley, T.; Pepe, M. S. (2000). "Time-dependent ROC
           curves for censored survival data and a diagnostic marker".
           Biometrics, 56(2), 337-344.
           https://doi.org/10.1111/j.0006-341X.2000.00337.x
        .. [2] Uno, H.; Cai, T.; Tian, L.; Wei, L. J. (2007). "Evaluating prediction
           rules for t-year survivors with censored regression models".
           Journal of the American Statistical Association, 102(478), 527-537.
           https://doi.org/10.1198/016214507000000149
        """
        if method.lower() in {"ipcw", "uno"}:
            self._error_trainset("IPCW AUROC")

        return auc(
            predict_probs=self._pred_probs,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            target_time=self.target_time,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
        )

    def brier_score(self, IPCW_weighted: bool = True) -> float:
        """
        Calculate the Brier score at a given time point from the predicted survival curve.

        The original Brier score is described in [1]_. IPCW estimation for
        right-censored outcomes follows Graf et al. [2]_ and Gerds and
        Schumacher [3]_.

        Parameters
        ----------
        IPCW_weighted : bool, default=True
            Whether to use IPCW weighting for the Brier score.

        Returns
        -------
        brier_score : float
            The Brier score at the target time point.

        References
        ----------
        .. [1] Brier, G. W. (1950). "Verification of forecasts expressed in terms
           of probability". Monthly Weather Review, 78(1), 1-3.
           https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
        .. [2] Graf, E.; Schmoor, C.; Sauerbrei, W.; Schumacher, M. (1999).
           "Assessment and comparison of prognostic classification schemes for
           survival data". Statistics in Medicine, 18(17-18), 2529-2545.
           https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5
        .. [3] Gerds, T. A.; Schumacher, M. (2006). "Consistent Estimation of the
           Expected Brier Score in General Survival Models with Right-Censored
           Event Times". Biometrical Journal, 48(6), 1029-1040.
           https://doi.org/10.1002/bimj.200610301
        """
        # Fully observed test outcomes use the plain Brier score.
        if self._NO_CENSOR:
            IPCW_weighted = False

        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Brier score (BS)")
        return float(
            brier_multiple_points(
                pred_mat=self._pred_probs[:, None],
                event_times=self.event_times,
                event_indicators=self.event_indicators,
                train_event_times=self.train_event_times,
                train_event_indicators=self.train_event_indicators,
                target_times=np.asarray([self.target_time]),
                ipcw=IPCW_weighted,
            )[0]
        )

    def one_calibration(
        self,
        num_bins: int = 10,
        binning_strategy: str = "C",
        method: str = "DN",
        return_details: bool = False,
        adjust_variance: bool = True,
    ) -> tuple[float, list, list] | tuple[float, dict]:
        """
        Calculate the one calibration score at a given time point from the predicted survival curve.

        The ``DN`` method follows D'Agostino and Nam [1]_; its Greenwood
        variance modification is discussed in [2]_.

        Parameters
        ----------
        num_bins : int, default=10
            Number of bins used to calculate the one calibration score.
        binning_strategy : str, default="C"
            The strategy to bin the predictions. The options are: "C" (default), and "H".
            C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
            H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
        method : str, default="DN"
            The method used to calculate the one calibration score.
            Options: "Uncensored", or "DN" (default)
        return_details : bool, default=False
            Whether to return the detailed calibration information.

        adjust_variance : bool, default=True
            Use censoring-aware Greenwood variance for DN, or observed binomial
            variance for Uncensored, with G degrees of freedom. False uses
            the original expected binomial variance and HL/DN convention.
            Adjusted p-values are undefined if any group variance is zero.

        Returns
        -------
        p_value : float
            The approximate calibration p-value; NaN for an undefined adjusted test.
        observed_probabilities : list
            The observed probabilities in each bin.
        expected_probabilities : list
            The expected probabilities in each bin.
        details : dict, optional
            A dictionary containing detailed calibration information, including:
            - p_value: The approximate calibration p-value; NaN for an undefined adjusted test.
            - statistics: The grouped calibration statistic.
            - variances, bin_sizes, degrees_of_freedom: Test variance and group information.
            - observed_probabilities: The observed probabilities in each bin.
            - expected_probabilities: The expected probabilities in each bin.
            - slope: The slope of the calibration curve.
            - intercept: The intercept of the calibration curve.
            - max_local_deviation: The maximum local deviation between observed and expected probabilities.
            - histogram_plot: A tuple containing the figure and axes of the histogram plot.
            - pp_plot: A tuple containing the figure and axes of the P-P plot.
            Only returned when 'return_details' is set to True.

        References
        ----------
        .. [1] D'Agostino, R. B.; Nam, B.-H. (2003). "Evaluation of the performance
           of survival analysis models: discrimination and calibration measures".
           Handbook of Statistics, 23, 1-25.
           https://doi.org/10.1016/S0169-7161(03)23001-7
        .. [2] Demler, Paynter and Cook (2015). "Tests of Calibration and Goodness of Fit
           in the Survival Setting". Statistics in Medicine, 34(10), 1659-1680.
           https://doi.org/10.1002/sim.6428
        """
        if self._NO_CENSOR:
            method = "Uncensored"

        p_value, details = one_calibration(
            preds=1 - self._pred_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=self.target_time,
            num_bins=num_bins,
            binning_strategy=binning_strategy,
            method=method,
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
        knots: int = 3,
        draw_figure: bool | None = True,
        figure_range: tuple | None = None,
    ) -> dict | tuple[dict, tuple[plt.Figure, plt.Axes]]:
        """
        Calculate the integrated one calibration index (ICI) for a given set of predictions and true event times.

        The smoothed calibration curve and integrated calibration index follow [1]_.

        Parameters
        ----------
        knots : int, default=3
            Total restricted cubic spline knot count, at least three.
            Nonconstant risks must support distinct knots and an identifiable fit.
        draw_figure : bool, default=True
            Whether to create a figure showing the graphical calibration curve.
        figure_range : tuple, optional
            The range of the figure to be plotted. If None, the range is automatically determined.

        Returns
        -------
        dict or tuple[dict, tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]]
            ICI summary containing ICI, E50, E90, E_max, and curve data. When draw_figure is True
            return (summary, (figure, axes)); otherwise return the summary alone.

        References
        ----------
        .. [1] Austin et al. (2020). "Graphical calibration curves and the integrated
           calibration index (ICI) for survival models". Statistics in Medicine, 39(21),
           2714-2742. https://doi.org/10.1002/sim.8570
        """
        return integrated_calibration_index(
            preds=1 - self._pred_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=self.target_time,
            knots=knots,
            draw_figure=draw_figure,
            figure_range=figure_range,
        )

    def _error_trainset(self, method_name: str) -> None:
        """
        Require training outcomes for a metric that estimates censoring or event distributions.

        Parameters
        ----------
        method_name : str
            Metric name to include in the missing-training-data error.

        Raises
        ------
        TypeError
            If the training outcomes required by the metric are unavailable.
        """
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError(
                "Train set information is missing. "
                f"Evaluator cannot perform {method_name} evaluation."
            )
