from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Union

import numpy as np


@dataclass
class NelsonAalen:
    """
    Implementation of the Nelson-Aalen estimator for cumulative hazard function.
    """

    event_times: InitVar[np.ndarray]
    event_indicators: InitVar[np.ndarray]
    survival_times: np.ndarray = field(init=False)
    population_count: np.ndarray = field(init=False)
    events: np.ndarray = field(init=False)
    hazard: np.ndarray = field(init=False)
    cumulative_hazard: np.ndarray = field(init=False)
    survival_probabilities: np.ndarray = field(init=False)

    def __post_init__(self, event_times, event_indicators):
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

    def predict(
        self, prediction_times: Union[int, float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Predict the cumulative hazard based on the survival times.
        Parameters
        ----------
        prediction_times: int | float | np.ndarray
            Time(s) at which to predict the cumulative hazard.
        Returns
        -------
        cumulative_hazard: float | np.ndarray
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
        self, prediction_times: Union[int, float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Predict the survival probabilities at the given prediction times.
        Parameters
        ----------
        prediction_times: int | float | np.ndarray
            Time(s) at which to predict the survival probabilities.
        Returns
        -------
        survival_probabilities: float | np.ndarray
            The predicted survival probabilities at the given times.
        """
        cum_hazards = self.predict(prediction_times)
        survival_probabilities = np.exp(-cum_hazards)
        return survival_probabilities
