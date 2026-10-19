"""Evaluation of predicted survival curves for right-censored outcomes."""

from __future__ import annotations

import warnings
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from lifelines.statistics import logrank_test

from SurvivalEVAL.Evaluations.Calibration import (
    d_calibration,
    integrated_calibration_index,
    km_calibration,
    ksd_calibration,
    one_calibration,
    residuals,
)
from SurvivalEVAL.Evaluations.Calibration.SingleTime.util import (
    _calibration_curve_details,
    _maximum_local_deviation,
)
from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.Discrimination import auc, concordance
from SurvivalEVAL.Evaluations.Discrimination._concordance_utils import _normalize_ties
from SurvivalEVAL.Evaluations.Discrimination.Distributional.TimeDependentConcordance import (
    _concordance_time_dependent,
    _normalize_time_dependent_method,
)
from SurvivalEVAL.Evaluations.Others import auprc_right_censor
from SurvivalEVAL.Evaluations.PredictionError import brier_multiple_points, mean_error
from SurvivalEVAL.Evaluations.PredictionError.BrierScore import (
    _brier_time_grid,
    _integrate_brier_scores,
)
from SurvivalEVAL.Evaluations.util import (
    check_and_convert_event_data,
    fit_least_squares,
    validate_time_points,
)
from SurvivalEVAL.Evaluations.util_plots import pp_plot

from ._BaseSurvivalEvaluator import _BaseSurvivalEvaluator


class SurvivalEvaluator(_BaseSurvivalEvaluator):
    def __init__(
        self,
        pred_survs: NumericArrayLike,
        time_coordinates: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
    ) -> None:
        """
        Initialize the SurvivalEvaluator with predictions and observed outcomes.

        Parameters
        ----------
        pred_survs : NumericArrayLike
            Accept shapes: (n_time_points,) or (n_samples, n_time_points).
            Predicted survival curves for the testing samples.
            At least one of `pred_survs` or `time_coordinates` must be a 2D array.
        time_coordinates : NumericArrayLike
            Accept shapes: (n_time_points,) or (n_samples, n_time_points).
            Time coordinates corresponding to the survival curves.
            At least one of `pred_survs` or `time_coordinates` must be a 2D array.
        event_times : NumericArrayLike, shape = (n_samples, )
            Actual event/censor time for the testing samples.
        event_indicators : NumericArrayLike, shape = (n_samples, )
            Binary event indicators for the testing samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        train_event_times : NumericArrayLike | None, shape = (n_train_samples, ), default=None
            Actual event/censor time for the training samples.
        train_event_indicators : NumericArrayLike | None, shape = (n_train_samples, ), default=None
            Binary event indicators for the training samples: 1 denotes an
            observed event and 0 denotes a censored observation.
        predict_time_method : str, default="Median"
            Method for calculating predicted survival time. Available options are "Median", "Mean" and "RMST".
        interpolation : str, default="Linear"
            Method for interpolation. Available options are ['Linear', 'Pchip'].
        """
        event_times, event_indicators = check_and_convert_event_data(
            event_times, event_indicators
        )
        self.event_times = event_times
        self.event_indicators = event_indicators
        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert_event_data(
                train_event_times, train_event_indicators
            )
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        self._NO_CENSOR = np.all(self.event_indicators == 1)
        super().__init__(
            pred_survs,
            time_coordinates,
            n_samples=self.event_times.shape[0],
            predict_time_method=predict_time_method,
            interpolation=interpolation,
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
        return float(np.quantile(times, 0.5))

    def concordance(
        self,
        ties: str = "Risk",
        method: str = "Harrell",
        tau: Numeric | None = None,
    ) -> tuple[float, float, float]:
        """
        Calculate the concordance index between the predicted survival times and the true survival times.

        The ``Harrell`` and ``Naive`` methods follow [1]_; ``Uno`` and ``IPCW``
        use the censoring-weighted approach in [2]_. The ``Margin`` method is
        described in [3]_.

        Parameters
        ----------
        ties : str, default="Risk"
            A string indicating the way ties should be handled.
            Options: "None", "Time", "Risk" (default), or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        method : str, default="Harrell"
            A string indicating the method for constructing the pairs of samples.
            Options are "Harrell" (default), "Naive", "Uno", "IPCW", or "Margin".
            "Harrell": comparable pairs are anchored by samples with observed
            events. If sample i has an observed event at time t_i, it is compared
            with samples whose observed event/censoring time is greater than t_i.
            "Naive": alias of "Harrell".
            "Uno": Harrell-style comparable pairs weighted by inverse probability
            of censoring weights from the training data.
            "IPCW": alias of "Uno".
            "Margin": the pairs are constructed between all samples. A best-guess
            time for the censored samples will be calculated and used to construct
            the pairs. "Uno", "IPCW", and "Margin" require training set
            information.
        tau : float, optional, default=None
            Truncation time. If provided, only pairs whose effective earlier or
            anchor time is strictly before ``tau`` are counted. If None, no
            truncation is applied.

        Returns
        -------
        concordance_index : float
            The concordance index.
        num_concordant_pairs : float
            The number of concordant pairs.
        num_total_pairs : float
            The number of total pairs.

        References
        ----------
        .. [1] Harrell, F. E., Jr.; Lee, K. L.; Mark, D. B. (1996).
           "Multivariable prognostic models: issues in developing models, evaluating
           assumptions and adequacy, and measuring and reducing errors".
           Statistics in Medicine, 15(4), 361-387.
           https://doi.org/10.1002/(SICI)1097-0258(19960229)15:4%3C361::AID-SIM168%3E3.0.CO;2-4
        .. [2] Uno, H.; Cai, T.; Pencina, M. J.; D'Agostino, R. B.; Wei, L. J. (2011).
           "On the C-statistics for evaluating overall adequacy of risk prediction
           procedures with censored survival data". Statistics in Medicine,
           30(10), 1105-1117. https://doi.org/10.1002/sim.4154
        .. [3] Kumar, N.; Qi, S.-A.; Kuan, L.-H.; Sun, W.; Zhang, J.; Greiner, R. (2022).
           "Learning accurate personalized survival models for predicting hospital
           discharge and mortality of COVID-19 patients". Scientific Reports,
           12, 4472. https://doi.org/10.1038/s41598-022-08601-6
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "harrell"

        if method in {"margin", "uno", "ipcw"}:
            self._error_trainset(f"{method} concordance")

        return concordance(
            predicted_times=self.predicted_event_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties,
            tau=tau,
        )

    def auc(self, target_time: Numeric | None = None, method: str = "Naive") -> float:
        """
        Calculate cumulative/dynamic AUROC from predicted survival curves.

        Cases are observed events at or before the target. Controls are
        observations after the target and censorings exactly at the target.
        Earlier censorings are excluded.

        The cumulative/dynamic AUROC definition follows [1]_. For ``IPCW``
        and its alias ``Uno``, censoring-weighted estimation follows [2]_.

        Parameters
        ----------
        target_time : Numeric | None
            Evaluation time. If None, use the median observed time
            of the combined training and test sets, or just the test set
            when training outcomes are unavailable.
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

        if target_time is None:
            target_time = self.default_target_time

        predict_probs = self.predict_probability_from_curve(target_time)

        return auc(
            predict_probs=predict_probs,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            target_time=target_time,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
        )

    def auroc(self, target_time: Numeric | None = None, method: str = "Naive") -> float:
        """
        Alias for :meth:`auc`, including its boundary and IPCW conventions.

        The cumulative/dynamic AUROC definition follows [1]_. For ``IPCW``
        and its alias ``Uno``, censoring-weighted estimation follows [2]_.

        Parameters
        ----------
        target_time : Numeric | None
            Evaluation time, defaulting to the median observed
            time of the available training and test outcomes.
        method : str
            "Naive" (default), "IPCW", or its alias "Uno", case-insensitive.

        Returns
        -------
        float
            AUROC at the target time.

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
        return self.auc(target_time, method=method)

    def concordance_time_dependent(
        self,
        ties: str = "Risk",
        method: str = "Antolini",
        risks: str = "Survival",
        tau: Numeric | None = None,
    ) -> tuple[float, float, float]:
        """
        Calculate the time-dependent concordance index.

        Risk scores are evaluated at observed-event anchor times, processing
        one sample's curve at a time. Each contributing time is predicted at
        most once per sample; observed events tied on time remain separate
        anchors for pair counting.

        The ``Antolini`` and ``Naive`` methods follow [1]_. For ``IPCW``, see
        the approaches in [2]_ and [3]_.

        Parameters
        ----------
        ties : str, default="Risk"
            A string indicating the way ties should be handled.
            Options: "None", "Time", "Risk" (default), or "All"
            "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
            "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
            "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
            "All" includes all ties.
            Note the concordance calculation is given by
            (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).
        method : str, default="Antolini"
            A string indicating the method for constructing the pairs of samples.
            Options are "Antolini" (default), "Naive", "IPCW".
            "Antolini": comparable pairs are anchored by samples with observed
            events. If sample i has an observed event at time t_i, it is compared
            with samples whose observed event/censoring time is greater than t_i.
            "Naive": alias of "Antolini".
            "IPCW": Antolini-style comparable pairs weighted by inverse probability
            of censoring weights from the training data.
        risks : str, default="Survival"
            A string indicating the type of risk scores to be used for concordance calculation.
            Options are "Survival" (default) and "Hazard".
            "Survival": predicted survival probabilities at anchor event times are converted to risk scores
            using -S(t | z).
            "Hazard": predicted hazard rates at anchor event times are used directly as risk scores.
        tau : float, optional, default=None
            Truncation time. If provided, only pairs whose effective earlier or
            anchor time is strictly before ``tau`` are counted. If None, no
            truncation is applied.

        Returns
        -------
        concordance_index : float
            The time-dependent concordance index.
        num_concordant_pairs : float
            The number of concordant pairs.
        num_total_pairs : float
            The number of total pairs.

        References
        ----------
        .. [1] Antolini, L.; Boracchi, P.; Biganzoli, E. (2005). "A time-dependent
           discrimination index for survival data". Statistics in Medicine,
           24(24), 3927-3944. https://doi.org/10.1002/sim.2427
        .. [2] Gandy, A.; Matcham, T. J. (2025). "A proper concordance index for
           models with crossing hazards". Scandinavian Journal of Statistics,
           52(4), 1479-1504. https://doi.org/10.1111/sjos.70000
        .. [3] Faruk, A.; Palczewski, J.; Aivaliotis, G. (2025). "A new discrimination
           measure for assessing predictive performance of non-linear survival models".
           arXiv preprint, arXiv:2504.05630. https://arxiv.org/abs/2504.05630
        """
        risks = risks.lower()

        if self._NO_CENSOR:
            method = "antolini"

        method = _normalize_time_dependent_method(method)
        ties = _normalize_ties(ties)

        if risks not in {"survival", "hazard"}:
            error = "Risks must be 'Survival' or 'Hazard', " f"got '{risks}' instead"
            raise ValueError(error)

        if method == "ipcw":
            self._error_trainset("IPCW time-dependent concordance")

        return _concordance_time_dependent(
            risk_scores=partial(self._predict_risks_from_curve, risks=risks),
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties,
            tau=tau,
        )

    def brier_score(
        self,
        target_time: Numeric | None = None,
        IPCW_weighted: bool = True,
    ) -> float:
        """
        Calculate the Brier score at a given time point from the predicted survival curve.

        The original Brier score is described in [1]_. IPCW estimation for
        right-censored outcomes follows Graf et al. [2]_ and Gerds and
        Schumacher [3]_.

        Parameters
        ----------
        target_time : float, int, or None, default=None
            Time point at which the Brier score is to be calculated. If None, the Brier score is calculated at the
            median time of the test event/censor times and, when available, the training event/censor times.
        IPCW_weighted : bool, default=True
            Whether to use IPCW weighting for the Brier score.

        Returns
        -------
        float
            Brier score at the target time.

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
        if target_time is None:
            target_time = self.default_target_time

        return float(
            self.brier_score_multiple_points(np.asarray([target_time]), IPCW_weighted)[
                0
            ]
        )

    def brier_score_multiple_points(
        self,
        target_times: np.ndarray,
        IPCW_weighted: bool = True,
    ) -> np.ndarray:
        """
        Calculate multiple Brier scores at multiple specific times.

        The original Brier score is described in [1]_. IPCW estimation for
        right-censored outcomes follows Graf et al. [2]_ and Gerds and
        Schumacher [3]_.

        Parameters
        ----------
        target_times : ndarray
            The specific time points for which to estimate the Brier scores.
        IPCW_weighted : bool, default=True
            Whether to use IPCW weighting for the Brier score.

        Returns
        -------
        brier_scores : ndarray
            Values of multiple Brier scores, in the same order as `target_times`.

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

        target_times = validate_time_points(target_times, input_name="target_times")

        pred_probs_mat = self.predict_multi_probabilities_from_curve(target_times)

        return brier_multiple_points(
            pred_mat=pred_probs_mat,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            target_times=target_times,
            ipcw=IPCW_weighted,
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
        Calculate the integrated Brier score (IBS) from the predicted survival curve.

        This averages the time-dependent Brier score over the selected interval,
        following the integrated prediction-error measure in [1]_. When enabled,
        IPCW accounts for right-censoring as described in [1]_ and [2]_.

        Parameters
        ----------
        num_points : int, optional, default=None
            Number of evaluation time points to generate automatically.
            If provided, `target_times` must be None.
            We generate `num_points` linearly spaced times between 0 and max_target_time,
            where max_target_time is the maximum observed (event or censoring) time
            across train + test.

            If both `num_points` and `target_times` are None:
            - Infer `target_times` from the unique censoring times in the test set.

        target_times : ndarray, shape = (m,), optional, default=None
            Explicit time grid at which to evaluate the Brier score.
            If provided, `num_points` must be None. We'll integrate over exactly these times.
            If unsorted, values are sorted ascending with a warning. You are
            responsible for making sure they lie within the support of the model.

        IPCW_weighted : bool, default=True
            Whether to use IPCW weighting.

        integration_method : str, default="trapz"
            Numerical integration method. Options: "trapz" (trapezoidal), "simpson".

        draw_figure : bool, default=False
            Whether to draw the figure of the IBS.

        Returns
        -------
        ibs : float
            The integrated Brier score when `draw_figure` is False.
        result : tuple[float, tuple[plt.Figure, plt.Axes]]
            When `draw_figure` is True, returns `(ibs, (fig, ax))`.

        References
        ----------
        .. [1] Graf, E.; Schmoor, C.; Sauerbrei, W.; Schumacher, M. (1999).
           "Assessment and comparison of prognostic classification schemes for
           survival data". Statistics in Medicine, 18(17-18), 2529-2545.
           https://doi.org/10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5
        .. [2] Gerds, T. A.; Schumacher, M. (2006). "Consistent Estimation of the
           Expected Brier Score in General Survival Models with Right-Censored
           Event Times". Biometrical Journal, 48(6), 1029-1040.
           https://doi.org/10.1002/bimj.200610301
        """
        # Fully observed test outcomes use the plain Brier score.
        if self._NO_CENSOR:
            IPCW_weighted = False

        if IPCW_weighted:
            self._error_trainset("IPCW-weighted Integrated Brier Score (IBS)")

        target_times = _brier_time_grid(
            self.event_times,
            self.event_indicators,
            self.train_event_times,
            num_points,
            target_times,
        )
        b_scores = self.brier_score_multiple_points(target_times, IPCW_weighted)
        if np.isnan(b_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(target_times, b_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are:\n", bs_dict)

        ibs_score = _integrate_brier_scores(b_scores, target_times, integration_method)

        # Draw the Brier score graph
        if draw_figure:
            fig, ax = plt.subplots()
            ax.plot(target_times, b_scores, "bo-")
            score_text = rf"IBS$= {ibs_score:.3f}$"
            ax.plot([], [], " ", label=score_text)
            ax.legend()
            ax.set_xlabel("Time")
            ax.set_ylabel("Brier Score")
            ax.set_title("Integrated Brier Score")
            plt.show()
            return ibs_score, (fig, ax)
        return ibs_score

    def mae(
        self,
        method: str = "Hinge",
        weighted: bool | None = None,
        log_scale: bool = False,
        verbose: bool = False,
        truncated_time: float | None = None,
    ) -> float:
        """
        Calculate the MAE score for the test set.

        The ``IPCW-D``, ``IPCW-T``, and ``Pseudo_obs`` MAE formulations are
        described in [1]_. For truncated ``Margin`` error, see [2]_.

        Parameters
        ----------
        method : str, default="Hinge"
            The method used to calculate the MAE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted : bool, default=None
            Whether to use weighting scheme for MAE.
            If None, the default value is False for "Uncensored" and "Hinge" methods, and True for the rest.
        log_scale : bool, default=False
            Whether to use log scale for the time axis.
        verbose : bool, default=False
            Whether to show the progress bar.
        truncated_time : float, default=None
            Optional upper limit for the event-time errors in methods that support truncation.

        Returns
        -------
        mae_score : float
            The MAE score for the test set.

        References
        ----------
        .. [1] Qi, S.-A.; Kumar, N.; Farrokh, M.; Sun, W.; Kuan, L.-H.;
           Ranganath, R.; Henao, R.; Greiner, R. (2023).
           "An Effective Meaningful Way to Evaluate Survival Models".
           Proceedings of the 40th International Conference on Machine Learning,
           Proceedings of Machine Learning Research, 202, 28244-28276.
           https://proceedings.mlr.press/v202/qi23b.html
        .. [2] Sun, W.; Kalmady, S. V.; Qi, S.-A.; Sepehrvand, N.; Hindle, A.;
           Greiner, R.; Kaul, P. (2024). "Predicting Individual Survival Distributions
           Using ECG: A Deep Learning Approach Utilizing Features Extracted by a
           Learned Diagnostic Model". Proceedings of the AAAI Symposium Series,
           2(1), 475-481. https://doi.org/10.1609/aaaiss.v2i1.27716
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "uncensored"

        if weighted is None:
            weighted = method not in ("uncensored", "hinge")

        return mean_error(
            predicted_times=self.predicted_event_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="absolute",
            method=method,
            weighted=weighted,
            log_scale=log_scale,
            verbose=verbose,
            truncation_time=truncated_time,
        )

    def mse(
        self,
        method: str = "Hinge",
        weighted: bool | None = None,
        log_scale: bool = False,
        verbose: bool = False,
        truncated_time: float | None = None,
    ) -> float:
        """
        Calculate the MSE score for the test set.

        The censoring strategies adapt the MAE formulations in [1]_. The
        truncated ``Margin`` approach is described for absolute error in [2]_.

        Parameters
        ----------
        method : str, default="Hinge"
            The method used to calculate the MSE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted : bool, default=None
            Whether to use weighting scheme for MSE.
            If None, the default value is False for "Uncensored" and "Hinge" methods, and True for the rest.
        log_scale : bool, default=False
            Whether to use log scale for the time axis.
        verbose : bool, default=False
            Whether to show the progress bar.
        truncated_time : float, default=None
            Optional upper limit for the event-time errors in methods that support truncation.

        Returns
        -------
        mse_score : float
            The MSE score for the test set.

        References
        ----------
        .. [1] Qi, S.-A.; Kumar, N.; Farrokh, M.; Sun, W.; Kuan, L.-H.;
           Ranganath, R.; Henao, R.; Greiner, R. (2023).
           "An Effective Meaningful Way to Evaluate Survival Models".
           Proceedings of the 40th International Conference on Machine Learning,
           Proceedings of Machine Learning Research, 202, 28244-28276.
           https://proceedings.mlr.press/v202/qi23b.html
        .. [2] Sun, W.; Kalmady, S. V.; Qi, S.-A.; Sepehrvand, N.; Hindle, A.;
           Greiner, R.; Kaul, P. (2024). "Predicting Individual Survival Distributions
           Using ECG: A Deep Learning Approach Utilizing Features Extracted by a
           Learned Diagnostic Model". Proceedings of the AAAI Symposium Series,
           2(1), 475-481. https://doi.org/10.1609/aaaiss.v2i1.27716
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = "uncensored"

        if weighted is None:
            weighted = method not in ("uncensored", "hinge")

        return mean_error(
            predicted_times=self.predicted_event_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            error_type="squared",
            method=method,
            weighted=weighted,
            log_scale=log_scale,
            verbose=verbose,
            truncation_time=truncated_time,
        )

    def rmse(
        self,
        method: str = "Hinge",
        weighted: bool | None = None,
        log_scale: bool = False,
        verbose: bool = False,
        truncated_time: float | None = None,
    ) -> float:
        """
        Calculate the root mean squared error (RMSE) score for the test set.

        The censoring strategies adapt the MAE formulations in [1]_. The
        truncated ``Margin`` approach is described for absolute error in [2]_.

        Parameters
        ----------
        method : str, default="Hinge"
            The method used to calculate the RMSE score.
            Options: "Uncensored", "Hinge" (default), "Margin", "IPCW-T", "IPCW-D", or "Pseudo_obs"
        weighted : bool, default=None
            Whether to use weighting scheme for RMSE.
            If None, the default value is False for "Uncensored" and "Hinge" methods, and True for the rest.
        log_scale : bool, default=False
            Whether to use log scale for the time axis.
        verbose : bool, default=False
            Whether to show the progress bar.
        truncated_time : float, default=None
            Optional upper limit for the event-time errors in methods that support truncation.

        Returns
        -------
        rmse_score : float
            The RMSE score for the test set.

        References
        ----------
        .. [1] Qi, S.-A.; Kumar, N.; Farrokh, M.; Sun, W.; Kuan, L.-H.;
           Ranganath, R.; Henao, R.; Greiner, R. (2023).
           "An Effective Meaningful Way to Evaluate Survival Models".
           Proceedings of the 40th International Conference on Machine Learning,
           Proceedings of Machine Learning Research, 202, 28244-28276.
           https://proceedings.mlr.press/v202/qi23b.html
        .. [2] Sun, W.; Kalmady, S. V.; Qi, S.-A.; Sepehrvand, N.; Hindle, A.;
           Greiner, R.; Kaul, P. (2024). "Predicting Individual Survival Distributions
           Using ECG: A Deep Learning Approach Utilizing Features Extracted by a
           Learned Diagnostic Model". Proceedings of the AAAI Symposium Series,
           2(1), 475-481. https://doi.org/10.1609/aaaiss.v2i1.27716
        """
        return self.mse(method, weighted, log_scale, verbose, truncated_time) ** 0.5

    def one_calibration(
        self,
        target_time: Numeric | None = None,
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
        target_time : Numeric | None, default=None
            Evaluation time. None uses default_target_time, the median observed
            time across available training and test outcomes.
        num_bins : int, default=10
            Number of bins used to calculate the one calibration score.
        binning_strategy : str
            The strategy to bin the predictions. The options are: "C" (default), and "H".
            C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
            H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
        method : str, default="DN"
            The method used to calculate the one calibration score.
            Options: "Uncensored", or "DN" (default)
        return_details : bool, default=False
            Whether to return detailed calibration information, including plots and statistics.

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
        if target_time is None:
            target_time = self.default_target_time

        if self._NO_CENSOR:
            method = "Uncensored"

        predict_probs = self.predict_probability_from_curve(target_time)
        p_value, details = one_calibration(
            preds=1 - predict_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=target_time,
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
        target_time: Numeric | None = None,
        knots: int = 3,
        draw_figure: bool | None = True,
        figure_range: tuple | None = None,
    ) -> dict | tuple[dict, tuple[plt.Figure, plt.Axes]]:
        """
        Calculate the integrated one calibration index (ICI) for a given set of predictions and true event times.

        The smoothed calibration curve and integrated calibration index follow [1]_.

        Parameters
        ----------
        target_time : Numeric | None, default=None
            Evaluation time. None uses default_target_time, the median observed
            time across available training and test outcomes.
        knots : int, default=3
            The number of knots to use for the spline fit in the ICI calculation.
        draw_figure : bool, default=True
            Whether to draw the figure of the ICI.
        figure_range : tuple, default=None
            The range of the figure to be drawn. If None, the range will be automatically determined based on the
            predicted probabilities and the target time.

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
        if target_time is None:
            target_time = self.default_target_time

        predict_probs = self.predict_probability_from_curve(target_time)
        return integrated_calibration_index(
            preds=1 - predict_probs,
            event_time=self.event_times,
            event_indicator=self.event_indicators,
            target_time=target_time,
            knots=knots,
            draw_figure=draw_figure,
            figure_range=figure_range,
        )

    def d_calibration(
        self,
        num_bins: int = 10,
        return_details: bool = False,
    ) -> tuple[float, np.ndarray] | tuple[float, dict]:
        """
        Calculate the D calibration score from the predicted survival curve.

        D-calibration of individual survival distributions is described in [1]_.

        Parameters
        ----------
        num_bins : int, default=10
            Number of bins used to calculate the D calibration score.
        return_details : bool, default=False
            Whether to return detailed calibration information, including plots and statistics.

        Returns
        -------
        p_value : float
            The p-value of the calibration test.
        hist : ndarray
            The histogram of the predicted probabilities in each bin.
        details : dict, optional
            A dictionary containing detailed calibration information, including:
            - statistics: The D calibration statistics.
            - p_value: The p-value of the calibration test.
            - histogram: The histogram of the predicted probabilities in each bin.
            - x_calibration: The x-calibration score.
            - linear_slope_intercept: A dictionary containing the slope and intercept of the linear fit.
            - max_local_slope: The maximum local slope deviation.
            - histogram_plot: A tuple containing the figure and axes of the histogram plot.
            - pp_plot: A tuple containing the figure and axes of the P-P plot.

        References
        ----------
        .. [1] Haider, H.; Hoehn, B.; Davis, S.; Greiner, R. (2020).
           "Effective Ways to Build and Evaluate Individual Survival Distributions".
           Journal of Machine Learning Research, 21(85), 1-63.
           https://www.jmlr.org/papers/v21/18-772.html
        """
        predict_probs = self.predict_probability_from_curve(self.event_times)
        statistics, p_value, hist = d_calibration(
            pred_probs=predict_probs,
            event_indicators=self.event_indicators,
            num_bins=num_bins,
        )
        if return_details:
            # normalize the histogram
            N = hist.sum()
            d_cal_pdf = hist / N
            # compute the x-calibration score
            optimal = np.ones_like(d_cal_pdf) / num_bins
            x_cal = np.sum(np.square(d_cal_pdf - optimal))

            d_cal_cdf = np.cumsum(
                d_cal_pdf[::-1]
            )  # so that the first number belongs to the lowest prob bin
            d_cal_cdf = np.insert(d_cal_cdf, 0, 0)
            optimal_cdf = np.linspace(0, 1, num_bins + 1)

            # LS fit of observed CDF vs expected CDF
            slope, intercept = fit_least_squares(
                optimal_cdf, d_cal_cdf, left_anchor=False, right_anchor=False
            )

            # Local slope deviation, max slope with the highest ratio difference from 1
            max_slope = _maximum_local_deviation(d_cal_cdf, optimal_cdf)

            # horizontal histograms
            fig1, ax1 = plt.subplots()
            widths = hist
            gap_fraction = 0.95

            # label the y positions
            y_labels = [
                f"[{i/num_bins:.2f}, {(i+1)/num_bins:.2f}{')' if i < num_bins - 1 else ']'}"
                for i in range(num_bins)
            ]
            y_positions = (optimal_cdf[:-1] + optimal_cdf[1:]) / 2  # midpoints of bins

            ax1.barh(y_positions, widths, height=gap_fraction / num_bins, alpha=0.7)
            ax1.axvline(N / num_bins, ls="dashed", c="grey", label="Ideal Calibration")
            ax1.set_xlabel("Frequency")
            ax1.set_ylabel("Probability Bins")
            ax1.set_yticks(y_positions)
            ax1.set_yticklabels(y_labels)
            ax1.legend()
            fig1.tight_layout()

            # P-P plot
            fig2, ax2 = pp_plot(d_cal_cdf, optimal_cdf)

            details = {
                "statistics": statistics,
                "p_value": p_value,
                "histogram": hist,
                "x_calibration": x_cal,
                "linear_slope_intercept": {"slope": slope, "intercept": intercept},
                "max_local_slope": max_slope,
                "histogram_plot": (fig1, ax1),
                "pp_plot": (fig2, ax2),
            }
            return p_value, details

        return p_value, hist

    def ksd_calibration(
        self,
        return_details: bool = False,
    ) -> tuple[float, float] | tuple[float, dict]:
        """
        Calculate the KSD calibration score from the predicted survival curve.

        The KS calibration metric is motivated by [1]_. Here, the distribution
        of predicted probabilities is estimated with Kaplan-Meier.

        Parameters
        ----------
        return_details : bool, default=False
            Whether to return detailed calibration information, including plots and statistics.

        Returns
        -------
        p_value : float
            Asymptotic Kolmogorov-Smirnov calibration p-value.
        statistic_or_details : float or dict
            Kolmogorov-Smirnov statistic when return_details is False. Otherwise, a dictionary
            containing statistics, p_value, empirical_distribution, and figure.

        References
        ----------
        .. [1] Park, J.; Kim, D.; Kim, C.; Park, H.; Kang, S.; Kim, G. (2025).
           "KSP: Kolmogorov-Smirnov metric-based Post-Hoc Calibration for
           Survival Analysis". Advances in Neural Information Processing Systems, 38.
           https://doi.org/10.52202/085713-0827
        """
        predict_probs = self.predict_probability_from_curve(self.event_times)
        return ksd_calibration(
            pred_probs=predict_probs,
            event_indicators=self.event_indicators,
            return_details=return_details,
        )

    def residuals(
        self,
        method: str = "CoxSnell",
        draw_figure: bool = False,
    ) -> np.ndarray:
        """
        Calculate the residuals from the predicted survival (cumulative hazard) function.

        Cox-Snell residuals follow [1]_, and both modified Cox-Snell variants
        use the adjustments in [2]_. Martingale and deviance residuals are
        described in [3]_ and [4]_, respectively.

        Parameters
        ----------
        method : str, default="CoxSnell"
            The method used to calculate the residuals.
            Options: "CoxSnell" (default), "Modified CoxSnell-v1", "Modified CoxSnell-v2",
            "Martingale", "Deviance".
        draw_figure : bool, default=False
            Whether to draw the figure of the residuals.

        Returns
        -------
        residuals : ndarray
            The residuals calculated from the predicted survival curve.

        References
        ----------
        .. [1] Cox, D. R.; Snell, E. J. (1968). "A General Definition of Residuals".
           Journal of the Royal Statistical Society: Series B (Methodological),
           30(2), 248-265. https://doi.org/10.1111/j.2517-6161.1968.tb00724.x
        .. [2] Collett, D. (2015). "Modelling Survival Data in Medical Research".
           3rd ed., Chapter 4. Chapman and Hall/CRC.
           https://doi.org/10.1201/b18041
        .. [3] Barlow, W. E.; Prentice, R. L. (1988). "Residuals for relative risk
           regression". Biometrika, 75(1), 65-74.
           https://doi.org/10.1093/biomet/75.1.65
        .. [4] Therneau, T. M.; Grambsch, P. M.; Fleming, T. R. (1990).
           "Martingale-based residuals for survival models". Biometrika, 77(1),
           147-160. https://doi.org/10.1093/biomet/77.1.147
        """
        method = method.lower()
        if self._NO_CENSOR:
            method = (
                "coxsnell"
                if method
                in ["coxsnell", "modified coxsnell-v1", "modified coxsnell-v2"]
                else method
            )

        predict_probs = self.predict_probability_from_curve(self.event_times)
        return residuals(
            pred_probs=predict_probs,
            event_indicators=self.event_indicators,
            method=method,
            draw_figure=draw_figure,
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
            Whether to draw the figure of the KM calibration curve.

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
        if self.ndim_time == 1:
            average_survival_curve = np.mean(self._pred_survs, axis=0, dtype=float)
            calibration_times = self._time_coordinates
        else:
            calibration_times = np.unique(self.event_times[self.event_indicators == 1])
            sample_survival_curves = self.predict_multi_probabilities_from_curve(
                calibration_times
            )
            average_survival_curve = np.mean(
                sample_survival_curves, axis=0, dtype=float
            )

        return km_calibration(
            average_survival_curve=average_survival_curve,
            time_coordinates=calibration_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            interpolation_method=self.interpolation,
            draw_figure=draw_figure,
        )

    def log_rank(
        self,
        weightings: str | None = None,
        p: float | None = 0,
        q: float | None = 0,
    ) -> tuple[float, float]:
        """
        Calculate the log-rank test statistic and p-value for the predicted survival curve.

        Parameters
        ----------
        weightings : str, optional
            The weighting method is for weighted log-rank test.
            Options: "None" (default), "wilcoxon", "tarone-ware", "peto", "fleming-harrington".
            None means unweighted log-rank test.
            Wilcoxon uses the number of at-risk population at each time point as the weight.
            Tarone-Ware uses the square root of the number of at-risk population at each time point as the weight.
            Peto uses the estimated survival probability as the weight.
            Fleming-Harrington uses
               w_i = S(t_i) ** p * (1 - S(t_i)) ** q
        p : float, default=0
            The p parameter for the Fleming-Harrington weighting method.
        q : float, default=0
            The q parameter for the Fleming-Harrington weighting method.

        Returns
        -------
        p_value : float
            The p-value of the log-rank test.
        test_statistic : float
            The test statistic of the log-rank test.
        """
        results = logrank_test(
            durations_A=self.event_times,
            durations_B=self.predicted_event_times,
            event_observed_A=self.event_indicators,
            event_observed_B=np.ones_like(self.event_indicators, dtype=bool),
            weightings=weightings,
            p=p,
            q=q,
        )
        return results.p_value, results.test_statistic

    def auprc(self, n_quad: int = 256) -> float:
        """
        Calculate the survival AUPRC for right-censored data.

        The survival precision-recall measure follows [1]_.

        Parameters
        ----------
        n_quad : int, default=256
            Number of quadrature intervals used for numerical integration.

        Returns
        -------
        auprc : float
            The survival AUPRC.

        References
        ----------
        .. [1] Avati et al. (2020). "Countdown Regression: Sharp and Calibrated Survival Predictions".
           Proceedings of Machine Learning Research, 115, 145-155.
           https://proceedings.mlr.press/v115/avati20a.html
        """
        predictions_cdf = 1 - self._pred_survs
        return auprc_right_censor(
            pred_cdf=predictions_cdf,
            time_grid=self._time_coordinates,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            n_quad=n_quad,
            return_details=False,
        )
