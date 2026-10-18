from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import numpy as np

from SurvivalEVAL.NonparametricEstimator.util import _compute_event_counts


@dataclass
class NelsonAalen:
    """
    Estimate the cumulative hazard and survival functions with Nelson-Aalen increments.

    The estimator is described by Aalen [1]_ and Nelson [2]_, [3]_.

    Parameters
    ----------
    event_times : ndarray, shape (n_samples,)
        Non-negative observed event or censoring times.
    event_indicators : ndarray, shape (n_samples,)
        Binary indicators: 1 for observed events and 0 for censoring.

    References
    ----------
    .. [1] Aalen, O. (1978). "Nonparametric Inference for a Family of Counting Processes."
       The Annals of Statistics, 701-726.
    .. [2] Nelson, W. (1969). "Hazard Plotting for Incomplete Failure Data."
       Journal of Quality Technology, 1(1), 27-52.
    .. [3] Nelson, W. (1972). "Theory and Applications of Hazard Plotting for Censored Failure Data."
       Technometrics, 14(4), 945-966.
    """

    event_times: InitVar[np.ndarray]
    event_indicators: InitVar[np.ndarray]
    survival_times: np.ndarray = field(init=False)
    population_count: np.ndarray = field(init=False)
    events: np.ndarray = field(init=False)
    hazard: np.ndarray = field(init=False)
    cumulative_hazard: np.ndarray = field(init=False)
    survival_probabilities: np.ndarray = field(init=False)

    def __post_init__(
        self,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
    ) -> None:
        """
        Fit the NelsonAalen estimator from the supplied observations.

        Parameters
        ----------
        event_times : ndarray
            Observed event or censoring times, one per sample.
        event_indicators : ndarray
            Binary event indicators: 1 denotes an observed event and 0 denotes censoring.
        """
        self.survival_times, self.population_count, self.events = _compute_event_counts(
            event_times, event_indicators
        )

        self.hazard = self.events / self.population_count
        self.cumulative_hazard = np.cumsum(self.hazard)
        self.survival_probabilities = np.exp(-self.cumulative_hazard)

        # Add the pre-event baseline explicitly and keep all fitted arrays aligned.
        # An observed time zero is left untouched because it contains a real update.
        if self.survival_times[0] > 0:
            self.survival_times = np.insert(self.survival_times, 0, 0.0)
            self.population_count = np.insert(
                self.population_count, 0, len(event_indicators)
            )
            self.events = np.insert(self.events, 0, 0)
            self.hazard = np.insert(self.hazard, 0, 0.0)
            self.cumulative_hazard = np.insert(self.cumulative_hazard, 0, 0.0)
            self.survival_probabilities = np.insert(self.survival_probabilities, 0, 1.0)

    def predict(self, prediction_times: float | np.ndarray) -> float | np.ndarray:
        """
        Predict the cumulative hazard based on the survival times.

        Parameters
        ----------
        prediction_times : float | ndarray
            Time(s) at which to predict the cumulative hazard.

        Returns
        -------
        cumulative_hazard : float | ndarray
            The predicted cumulative hazard at the given times.
        """
        prediction_times = np.asarray(prediction_times, dtype=float)
        original_shape = prediction_times.shape
        prediction_times = prediction_times.reshape(-1)
        if np.any(prediction_times < 0):
            raise ValueError("Prediction times must be non-negative.")

        # Nelson-Aalen is a right-continuous step function: use the latest
        # cumulative hazard whose fitted time is not greater than the query.
        indices = (
            np.searchsorted(self.survival_times, prediction_times, side="right") - 1
        )
        indices = np.clip(indices, 0, self.survival_times.size - 1)
        cumulative_hazard = self.cumulative_hazard[indices].astype(float, copy=True)

        cumulative_hazard = cumulative_hazard.reshape(original_shape)
        if cumulative_hazard.ndim == 0:
            return float(cumulative_hazard)

        return cumulative_hazard

    def predict_survival(
        self,
        prediction_times: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Predict the survival probabilities at the given prediction times.

        Parameters
        ----------
        prediction_times : float | ndarray
            Time(s) at which to predict the survival probabilities.

        Returns
        -------
        survival_probabilities : float | ndarray
            The predicted survival probabilities at the given times.
        """
        cum_hazards = self.predict(prediction_times)
        survival_probabilities = np.exp(-cum_hazards)
        return survival_probabilities
