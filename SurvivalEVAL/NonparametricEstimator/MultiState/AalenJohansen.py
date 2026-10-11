"""Aalen–Johansen estimation from aggregated multi-state transition counts."""

import numpy as np

from SurvivalEVAL.NonparametricEstimator.util import _predict_step


class AalenJohansenMultiState:
    """Estimate transition matrices for states numbered 0 through n_states - 1.

    At each time, off-diagonal hazard increments are transition counts divided
    by their source state's risk set. Diagonal increments are minus the sum of
    outgoing hazards, so each jump matrix is a probability matrix. The estimate
    is the chronological product of these jump matrices, starting at identity.
    Tied transitions must be aggregated into one update per distinct time.

    Full transition probabilities have the usual Markov interpretation.
    For an initial state distribution p0, state occupation probabilities are
    ``p0 @ predict_P(t)``. Occupation probabilities need not be monotone when
    states allow onward transitions or return visits.

    Parameters
    ----------
    n_states : int
        Positive number of states, including absorbing states.

    Attributes
    ----------
    unique_times_ : ndarray, shape (J,)
        Sorted update times.
    P_ : ndarray, shape (J, n_states, n_states)
        Transition matrices from before the first update through each time.
    """

    def __init__(self, n_states: int):
        if (
            isinstance(n_states, (bool, np.bool_))
            or not isinstance(n_states, (int, np.integer))
            or n_states < 1
        ):
            raise ValueError("n_states must be a positive integer.")
        self.n_states = n_states
        self.unique_times_ = None
        self.P_ = None

    def fit(self, event_times, risk_sets, transitions):
        """Fit from aggregated risk sets and transition counts, returning self.

        Parameters
        ----------
        event_times : array-like, shape (J,)
            Finite, non-negative, strictly increasing update times.
        risk_sets : array-like, shape (J, n_states)
            Non-negative numbers at risk in each state immediately before
            each update. Censorings at that time remain in these risk sets.
        transitions : array-like, shape (J, n_states, n_states)
            Non-negative counts from source state r to destination state s.
            Diagonal entries must be zero; do not supply negative diagonal
            hazard increments. Total departures cannot exceed the risk set.

        Notes
        -----
        A zero risk set requires zero departures and gives an identity jump
        row. An empty update schedule is valid and predicts identity everywhere.
        Fit uses O(J n_states**3) time and O(J n_states**2) storage.
        """
        event_times = np.asarray(event_times, dtype=float)
        risk_sets = np.asarray(risk_sets, dtype=float)
        transitions = np.asarray(transitions, dtype=float)
        if (
            event_times.ndim != 1
            or not np.all(np.isfinite(event_times))
            or np.any(event_times < 0)
            or np.any(event_times[1:] <= event_times[:-1])
        ):
            raise ValueError(
                "event_times must be finite, non-negative and strictly increasing."
            )
        n_times = event_times.size
        if risk_sets.shape != (n_times, self.n_states):
            raise ValueError("risk_sets must have shape (J, n_states).")
        if transitions.shape != (n_times, self.n_states, self.n_states):
            raise ValueError("transitions must have shape (J, n_states, n_states).")
        if not np.all(np.isfinite(risk_sets)) or np.any(risk_sets < 0):
            raise ValueError("risk_sets must be finite and non-negative.")
        if not np.all(np.isfinite(transitions)) or np.any(transitions < 0):
            raise ValueError("transitions must be finite and non-negative.")
        if np.any(np.diagonal(transitions, axis1=1, axis2=2) != 0):
            raise ValueError("transitions must have zero diagonal entries.")
        if np.any(transitions.sum(axis=2) > risk_sets):
            raise ValueError(
                "Total departures must not exceed the source state's risk set."
            )

        matrices = np.empty((n_times, self.n_states, self.n_states), dtype=float)
        previous = np.eye(self.n_states)
        diagonal = np.diag_indices(self.n_states)
        for j in range(n_times):
            jump = np.divide(
                transitions[j],
                risk_sets[j, :, None],
                out=np.zeros((self.n_states, self.n_states)),
                where=risk_sets[j, :, None] > 0,
            )
            jump[diagonal] = 1 - jump.sum(axis=1)
            previous = previous @ jump
            matrices[j] = previous

        self.unique_times_ = event_times.copy()
        self.P_ = matrices
        return self

    def predict_P(self, t):
        """Return matrices with shape ``shape(t) + (n_states, n_states)``.

        Predictions are right-continuous, equal identity before the first
        update, and remain constant after the last update. Positive infinity
        returns the final estimate; negative times and NaN are invalid.
        """
        return _predict_step(t, self.unique_times_, self.P_, np.eye(self.n_states))
