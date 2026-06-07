from __future__ import annotations

from dataclasses import dataclass, InitVar, field

import numpy as np

from SurvivalEVAL.NonparametricEstimator.SingleEvent.util import infer_survival_probabilities


@dataclass
class CopulaGraphic:
    """
    Implementation of the Copula Graphic estimator for survival function, under the dependent censoring assumption.

    This implementation supports three types of copulas:
    - Clayton
    - Gumbel
    - Frank
    The estimator is based on the R code in the package 'compound.Cox' by Takeshi Emura.
    To see the original R code, visit:
    https://github.com/cran/compound.Cox/tree/master/R/CG.Clayton.R
    https://github.com/cran/compound.Cox/tree/master/R/CG.Gumbel.R
    https://github.com/cran/compound.Cox/tree/master/R/CG.Frank.R

    However, the original R code (and also the math derivation) cannot handle ties --
    e.g., when a censored instance and an event instance have the same time point.
    This implementation correctly handling ties.
    based on the derivation in paper: https://arxiv.org/abs/2502.19460
    """

    event_times: InitVar[np.ndarray]
    event_indicators: InitVar[np.ndarray]
    alpha: InitVar[float]
    type: InitVar[str] = "Clayton"
    n_samples: int = field(init=False)
    survival_times: np.ndarray = field(init=False)
    population_count: np.ndarray = field(init=False)
    events: np.ndarray = field(init=False)
    survival_probabilities: np.ndarray = field(init=False)

    def __post_init__(self, event_times, event_indicators, alpha, type):
        alpha = max(alpha, 1e-9)
        self.n_samples = len(event_times)
        index = np.lexsort((event_indicators, event_times))
        unique_times = np.unique(event_times[index], return_counts=True)
        self.survival_times = unique_times[0]
        self.population_count = np.flip(np.flip(unique_times[1]).cumsum())

        event_counter = np.append(0, unique_times[1].cumsum()[:-1])
        event_ind = list()
        for i in range(np.size(event_counter[:-1])):
            event_ind.append(event_counter[i])
            event_ind.append(event_counter[i + 1])
        event_ind.append(event_counter[-1])
        event_ind.append(len(event_indicators))
        self.events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[
            ::2
        ]

        event_diff = self.population_count - self.events

        with np.errstate(divide="ignore"):
            # ignore division by zero warnings,
            # such warnings are expected when the last time point has an event so the event_diff is 0.
            # but we will set the last point to 0 anyway.
            if type == "Clayton":
                diff_ = (event_diff / self.n_samples) ** (-alpha) - (
                    self.population_count / self.n_samples
                ) ** (-alpha)
                diff_[-1] = 0
                self.survival_probabilities = (1.0 + np.cumsum(diff_)) ** (-1.0 / alpha)
            elif type == "Gumbel":
                diff_ = (-np.log(event_diff / self.n_samples)) ** (alpha + 1) - (
                    -np.log(self.population_count / self.n_samples)
                ) ** (alpha + 1)
                diff_[-1] = 0
                self.survival_probabilities = np.exp(
                    -np.cumsum(diff_) ** (1 / (1 + alpha))
                )
            elif type == "Frank":
                log_diff_ = np.log(
                    (np.exp(-alpha * event_diff / self.n_samples) - 1)
                    / (np.exp(-alpha * self.population_count / self.n_samples) - 1)
                )
                log_diff_[-1] = 0
                self.survival_probabilities = (
                    -1
                    / alpha
                    * np.log(1 + (np.exp(-alpha) - 1) * np.exp(np.cumsum(log_diff_)))
                )
            else:
                raise ValueError(
                    f"Unknown copula type: {type}. Supported types are 'Clayton', 'Gumbel', and 'Frank'."
                )

        # Add the pre-event baseline explicitly and keep all fitted arrays aligned.
        # An observed time zero is left untouched because it contains a real update.
        if self.survival_times[0] > 0:
            self.survival_times = np.insert(self.survival_times, 0, 0.0)
            self.population_count = np.insert(
                self.population_count, 0, len(event_indicators)
            )
            self.events = np.insert(self.events, 0, 0)
            self.survival_probabilities = np.insert(
                self.survival_probabilities, 0, 1.0
            )

        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

    def predict(self, prediction_times: int | float | np.ndarray) -> float | np.ndarray:
        """
        Predict the survival probabilities at the given prediction times.
        Parameters
        ----------
        prediction_times: int | float | np.ndarray
            Time(s) at which to predict the survival probabilities.
        Returns
        -------
        probabilities: float | np.ndarray
            Predicted survival probabilities at the given time(s).
        """
        prediction_times = np.asarray(prediction_times, dtype=float)
        original_shape = prediction_times.shape
        prediction_times = prediction_times.reshape(-1)

        # ensure the prediction times are all non-negative
        if np.any(prediction_times < 0):
            raise ValueError("Prediction times must be non-negative.")

        probs = infer_survival_probabilities(
            prediction_times, self.survival_times, self.survival_probabilities
        )

        probabilities = probs.reshape(original_shape)
        if probabilities.ndim == 0:
            return float(probabilities)

        return probabilities

    def predict_median(self) -> float:
        """
        Predict the median survival time based on the survival probabilities.
        It is calculated as the first time point where the survival probability is less than or equal to 0.5.
        No interpolation is performed, as we are strictly following the implementation of the CG estimator in R.
        Returns
        -------
        float
            The predicted median survival time.
        """
        median_index = np.where(self.survival_probabilities <= 0.5)[0]
        if median_index.size == 0:
            return np.inf
        return self.survival_times[median_index[0]]
