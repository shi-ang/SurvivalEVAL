"""Evaluation of event-time point predictions."""

from __future__ import annotations

import numpy as np
from lifelines.statistics import logrank_test

from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.Discrimination import concordance
from SurvivalEVAL.Evaluations.PredictionError import mean_error
from SurvivalEVAL.Evaluations.util import (
    check_and_convert,
    check_and_convert_event_data,
)


class PointEvaluator:
    def __init__(
        self,
        pred_times: NumericArrayLike,
        event_times: NumericArrayLike,
        event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike | None = None,
        train_event_indicators: NumericArrayLike | None = None,
    ) -> None:
        """
        Initialize the PointEvaluator with predictions and observed outcomes.

        Parameters
        ----------
        pred_times : NumericArrayLike, shape = (n_samples, )
            Predicted survival times for the testing samples.
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
        """
        self.event_times, self.event_indicators = check_and_convert_event_data(
            event_times, event_indicators
        )
        self.pred_times = pred_times

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert_event_data(
                train_event_times, train_event_indicators
            )
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        self._NO_CENSOR = np.all(self.event_indicators == 1)

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
    def pred_times(self) -> np.ndarray:
        """
        Return the stored event-time point predictions.

        Returns
        -------
        ndarray
            Stored predicted event times with shape (n_samples,).
        """
        return self._pred_times

    @pred_times.setter
    def pred_times(self, pred_times: NumericArrayLike) -> None:
        """
        Replace the stored event-time point predictions.

        Parameters
        ----------
        pred_times : NumericArrayLike
            Predicted event times, one per testing sample.
        """
        pred_times = check_and_convert(pred_times)
        if pred_times.shape != self.event_times.shape:
            raise ValueError("pred_times and event_times must have the same shape.")
        self._pred_times = pred_times

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
            The concordance index, which is the proportion of concordant pairs among all pairs.
        concordant_pairs : float
            The number of concordant pairs.
        total_pairs : float
            The total number of comparable pairs considered in the concordance calculation.

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
            predicted_times=self._pred_times,
            event_times=self.event_times,
            event_indicators=self.event_indicators,
            train_event_times=self.train_event_times,
            train_event_indicators=self.train_event_indicators,
            method=method,
            ties=ties,
            tau=tau,
        )

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
            predicted_times=self._pred_times,
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
            predicted_times=self._pred_times,
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
            Whether to use a weighting scheme for RMSE.
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
            durations_B=self._pred_times,
            event_observed_A=self.event_indicators,
            event_observed_B=np.ones_like(self.event_indicators, dtype=bool),
            weightings=weightings,
            p=p,
            q=q,
        )
        return results.p_value, results.test_statistic
