"""
Aalen–Johansen Estimator
==========================================================

This module implements the Aalen–Johansen (AJ) estimator using the standard
**aggregate-at-unique-times** approach. Theoretically, the AJ product-integral
formulation assumes a continuous-time process where no two distinct transitions
occur at exactly the same instant. 

However, real datasets often exhibit *tied event times*.

There are two common strategies for handling tied heterogeneous events:

1. **Jittering (e.g., in lifelines' implementation)**  
   Small random noise is added to tied event times to artificially impose a
   strict temporal ordering. 

2. **Aggregate-at-unique-times (e.g., in R's `mstate`, `cmprsk` implementations)**  
   All transitions sharing the same recorded time are grouped together, and a
   single Nelson–Aalen increment \Delta A(t) is computed using aggregated counts.
   The AJ estimator then applies one joint jump matrix at each distinct event
   time. 

This module follows the second approach. Aggregating tied transitions into a
single time point avoids unnecessary randomness.
"""

import numpy as np

class AalenJohansenCompetingRisks:
    """
    Aalen–Johansen estimator in a competing risks setting
    (no further multi-state transitions).

    States:
        0      : initial / no-event state
        1..K   : absorbing event states (one per cause)

    Data encoding:
        events = 0      -> censored
        events = 1..K   -> event of cause k

    Parameters
    ----------
    n_causes : int, optional
        Number of competing causes. If None, inferred from max(events).

    Attributes
    ----------
    n_causes : int
        Number of competing causes.
    n_states : int
        Number of states (K + 1, including the initial no-event state).
    unique_times_ : ndarray, shape (J,)
        Sorted distinct event times (any cause).
    surv_ : ndarray, shape (J,)
        Overall survival S(t_j) at event times.
    cif_ : ndarray, shape (n_causes, J)
        Cumulative incidence functions F_k(t_j) for each cause k at event times.
        cif_[k-1, j] corresponds to cause k at time unique_times_[j].
    P_ : ndarray, shape (J, n_states, n_states)
        Transition matrices P(t_j). Currently only the first row is non-zero:
            P_[j, 0, 0]   = S(t_j)
            P_[j, 0, k]   = F_k(t_j),  k = 1..K
            P_[j, r, :]   = 0 for r >= 1
    """

    def __init__(self, n_causes=None):
        self.n_causes = n_causes
        self.n_states = None
        self.unique_times_ = None
        self.surv_ = None
        self.cif_ = None
        self.P_ = None

    def fit(self, times, events, n_causes=None):
        """
        Fit the Aalen–Johansen estimator in competing risks.

        Parameters
        ----------
        times : array-like of shape (n,)
            Observed times (event or censoring).
        events : array-like of shape (n,)
            Event indicators: 0 = censored, 1..K = cause k.
        n_causes : int, optional
            Number of causes. If None, use self.n_causes or max(events).

        Returns
        -------
        self
        """
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)
        if times.shape != events.shape:
            raise ValueError("times and events must have the same shape")

        # Infer number of causes if needed
        if n_causes is None:
            if self.n_causes is not None:
                n_causes = self.n_causes
            else:
                if events.max() < 1:
                    raise ValueError("No events found (all events == 0).")
                n_causes = int(events.max())
        self.n_causes = n_causes
        self.n_states = n_causes + 1  # state 0 + K event states

        # Sort by time
        order = np.argsort(times)
        t_sorted = times[order]
        e_sorted = events[order]

        # Extract distinct times with at least one event
        mask_event = e_sorted > 0
        event_times = t_sorted[mask_event]
        if event_times.size == 0:
            raise ValueError("No events observed; cannot fit AJ estimator.")
        unique_times = np.unique(event_times)
        n_times = unique_times.shape[0]

        surv = np.empty(n_times, dtype=float)
        cif = np.zeros((n_causes, n_times), dtype=float)

        S_prev = 1.0
        cif_prev = np.zeros(n_causes, dtype=float)

        # Core AJ recursion (competing risks form)
        for j, t in enumerate(unique_times):
            # Risk set Y_j: all with T_i >= t
            at_risk = t_sorted >= t
            Y_j = at_risk.sum()

            if Y_j == 0:
                # No one at risk anymore: survival and CIF stay constant
                surv[j:] = S_prev
                cif[:, j:] = cif_prev[:, None]
                break

            # All events at time t
            at_t = t_sorted == t
            counts = np.bincount(
                e_sorted[at_t],
                minlength=self.n_causes + 1
            )  # index 0 = censored, 1..K = causes
            dNk = counts[1:]
            dN = dNk.sum()

            if dN == 0:
                surv[j] = S_prev
                cif[:, j] = cif_prev
                continue

            # All-cause hazard increment
            hazard = dN / Y_j

            # Survival update
            S_j = S_prev * (1.0 - hazard)

            # CIF increments
            dFk = S_prev * dNk / Y_j
            cif_j = cif_prev + dFk

            surv[j] = S_j
            cif[:, j] = cif_j

            S_prev = S_j
            cif_prev = cif_j

        self.unique_times_ = unique_times
        self.surv_ = surv
        self.cif_ = cif

        # Build P(t_j) matrices with only first row non-zero
        P = np.zeros((n_times, self.n_states, self.n_states), dtype=float)
        for j in range(n_times):
            P[j, 0, 0] = surv[j]            # P_00(t_j) = S(t_j)
            P[j, 0, 1:] = cif[:, j]         # P_0k(t_j) = F_k(t_j), k = 1..K

        self.P_ = P
        return self

    def predict_surv(self, t):
        """Evaluate S(t) as a right-continuous step function."""
        if self.unique_times_ is None:
            raise RuntimeError("Must call fit() before predict_surv().")

        t = np.asarray(t, dtype=float)
        S = np.ones_like(t, dtype=float)
        idx = np.searchsorted(self.unique_times_, t, side="right") - 1
        valid = idx >= 0
        S[valid] = self.surv_[idx[valid]]
        return S

    def predict_cif(self, t):
        """Evaluate CIFs F_k(t) as right-continuous step functions."""
        if self.unique_times_ is None:
            raise RuntimeError("Must call fit() before predict_cif().")

        t = np.asarray(t, dtype=float)
        m = t.shape[0]
        F = np.zeros((m, self.n_causes), dtype=float)

        idx = np.searchsorted(self.unique_times_, t, side="right") - 1
        valid = idx >= 0
        F[valid, :] = self.cif_[:, idx[valid]].T
        return F

    def predict_P(self, t):
        """
        Evaluate the transition matrices P(t) at given times.

        For this class (competing risks), only the first row is non-zero.
        """
        if self.P_ is None or self.unique_times_ is None:
            raise RuntimeError("Must call fit() before predict_P().")

        t = np.asarray(t, dtype=float)
        m = t.shape[0]
        P_t = np.zeros((m, self.n_states, self.n_states), dtype=float)

        idx = np.searchsorted(self.unique_times_, t, side="right") - 1
        valid = idx >= 0
        P_t[valid] = self.P_[idx[valid]]
        return P_t