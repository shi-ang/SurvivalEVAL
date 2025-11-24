import numpy as np
from SurvivalEVAL.NonparametricEstimator.CompetingRisks import AalenJohansenCompetingRisks

class AalenJohansenMultiState(AalenJohansenCompetingRisks):
    """
    General multi-state Aalen–Johansen estimator.

    This subclass reuses the AalenJohansenCompetingRisks and overrides `fit` 
    to perform the multi-state AJ product integral.

    Instead of raw individual-level data, we work with *aggregated* data:

    Parameters
    ----------
    n_states : int
        Total number of states in the multi-state process
        (states are indexed as 0, 1, ..., n_states - 1).

    Attributes
    ----------
    n_states : int
        Number of states.
    n_causes : int
        For compatibility with the base class, set to n_states - 1.
        The "CIF" stored in `cif_` is then the probability of being in
        each state k >= 1, given that we started in state 0.
    unique_times_ : ndarray, shape (J,)
        Event times t_(1) < ... < t_(J).
    P_ : ndarray, shape (J, n_states, n_states)
        Transition matrices P(t_j) at each event time.
    surv_ : ndarray, shape (J,)
        For compatibility: P_{00}(t_j) = probability of still being
        in state 0 at time t_j.
    cif_ : ndarray, shape (n_states - 1, J)
        For compatibility: cif_[k-1, j] = P_{0k}(t_j), i.e. probability
        of being in state k at time t_j given X(0)=0.
    """

    def __init__(self, n_states):
        if n_states < 1:
            raise ValueError("n_states must be >= 1.")
        # Base class expects n_causes; we don't really use the 'cause'
        # semantics here, but we set it so that predict_cif() works
        # as 'probability of being in state k>=1 given start in 0'.
        super().__init__(n_causes=n_states - 1)
        self.n_states = n_states

    def fit(self, event_times, Y, dN):
        """
        Fit the full multi-state Aalen–Johansen estimator from aggregated
        risk sets and transition counts.

        Parameters
        ----------
        event_times : array-like of shape (J,)
            Distinct, *sorted* event times t_(1) < ... < t_(J) where
            at least one transition occurs.
        Y : array-like of shape (J, n_states)
            Risk set sizes: Y[j, r] = number at risk in state r just
            before time event_times[j].
        dN : array-like of shape (J, n_states, n_states)
            Transition counts: dN[j, r, s] = number of transitions
            r -> s at time event_times[j].

        Returns
        -------
        self
        """
        event_times = np.asarray(event_times, dtype=float)
        Y = np.asarray(Y, dtype=float)
        dN = np.asarray(dN, dtype=float)

        if event_times.ndim != 1:
            raise ValueError("event_times must be a 1D array of shape (J,).")
        J = event_times.shape[0]

        if Y.shape != (J, self.n_states):
            raise ValueError(
                f"Y must have shape (J, n_states) = ({J}, {self.n_states}), "
                f"got {Y.shape}."
            )

        if dN.shape != (J, self.n_states, self.n_states):
            raise ValueError(
                f"dN must have shape (J, n_states, n_states) = "
                f"({J}, {self.n_states}, {self.n_states}), got {dN.shape}."
            )

        # Initialize
        P = np.zeros((J, self.n_states, self.n_states), dtype=float)
        P_prev = np.eye(self.n_states, dtype=float)

        for j in range(J):
            # Construct ΔA(t_j)
            dA = np.zeros((self.n_states, self.n_states), dtype=float)

            for r in range(self.n_states):
                Y_r = Y[j, r]
                if Y_r > 0:
                    dA[r, :] = dN[j, r, :] / Y_r
                else:
                    # No one at risk in state r -> no contribution
                    dA[r, :] = 0.0

            # Jump matrix J(t_j) = I + ΔA(t_j)
            J_jump = np.eye(self.n_states, dtype=float) + dA

            # Product integral: P(t_j) = P(t_{j-1}) @ J(t_j)
            P_j = P_prev @ J_jump
            P[j] = P_j
            P_prev = P_j

        self.unique_times_ = event_times
        self.P_ = P

        # For compatibility with the base-class API:
        #   surv_(t_j) = P_00(t_j)
        #   cif_(k-1, t_j) = P_0k(t_j)  (probability to be in state k at t_j)
        self.surv_ = P[:, 0, 0]
        if self.n_states > 1:
            # probabilities of being in states 1..n_states-1 when starting in 0
            self.cif_ = P[:, 0, 1:].T  # shape (n_states-1, J)
        else:
            self.cif_ = np.zeros((0, J), dtype=float)

        return self
