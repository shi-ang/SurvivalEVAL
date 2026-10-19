"""Aalen–Johansen estimation for right-censored competing risks."""

import numpy as np
from numpy.typing import ArrayLike

from SurvivalEVAL.nonparametric_estimator.util import _predict_step


class AalenJohansenCompetingRisks:
    """
    Estimate survival and cause-specific cumulative incidence functions.

    The estimator follows Aalen and Johansen [1]_.

    State 0 is the initial state; states 1 through K are absorbing causes.
    Events and censorings at the same time share the risk set immediately
    before that time. All causes at a tied time are updated together, without
    jittering. Observations enter at time zero; delayed entry is not supported.

    Parameters
    ----------
    n_causes : int, optional
        Number of causes. If omitted, infer it from the largest event label
        on each fit. Specify it to include unobserved causes or fit data with
        only censorings.

    Attributes
    ----------
    n_causes_ : int
        Number of causes in the fitted model.
    unique_times_ : ndarray, shape (J,)
        Sorted distinct event times, excluding censoring-only times.
    surv_ : ndarray, shape (J,)
        Probability of remaining in state 0 after each event time.
    cif_ : ndarray, shape (K, J)
        Cumulative incidence for each cause; row k corresponds to cause k + 1.

    References
    ----------
    .. [1] Aalen, O. O.; Johansen, S. (1978). "An Empirical Transition Matrix for
       Non-Homogeneous Markov Chains Based on Censored Observations".
       Scandinavian Journal of Statistics, 5(3), 141-150.
       https://www.jstor.org/stable/4615704
    """

    def __init__(self, n_causes: int | None = None) -> None:
        """
        Initialize the AalenJohansenCompetingRisks estimator.

        Parameters
        ----------
        n_causes : int | None
            Positive number of event causes; None infers it from the fitted event labels.
        """
        if n_causes is not None and (
            isinstance(n_causes, (bool, np.bool_))
            or not isinstance(n_causes, (int, np.integer))
            or n_causes < 1
        ):
            raise ValueError("n_causes must be a positive integer.")
        self.n_causes = n_causes
        self.n_causes_ = None
        self.unique_times_ = None
        self.surv_ = None
        self.cif_ = None
        self._at_risk = None
        self._event_counts = None

    def fit(self, times: ArrayLike, events: ArrayLike) -> "AalenJohansenCompetingRisks":
        """
        Fit from observed times and event labels, returning self.

        Parameters
        ----------
        times : array-like, shape (n,)
            Nonempty, finite, non-negative event or censoring times.
        events : array-like, shape (n,)
            Integer labels: 0 for censoring, 1 through K for event causes.

        Returns
        -------
        AalenJohansenCompetingRisks
            This estimator after fitting.

        Notes
        -----
        At each distinct event time, survival is multiplied by
        ``1 - total_events / at_risk``. Each CIF increases by
        ``survival_before_time * cause_events / at_risk``.
        Grouped counts give O(n log n + K J) time and O(n + K J) memory.
        """
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=float)
        if times.ndim != 1 or times.size == 0 or events.shape != times.shape:
            raise ValueError(
                "times and events must be nonempty 1D arrays of equal length."
            )
        if not np.all(np.isfinite(times)) or np.any(times < 0):
            raise ValueError("times must be finite and non-negative.")
        if (
            not np.all(np.isfinite(events))
            or np.any(events < 0)
            or np.any(events != np.floor(events))
        ):
            raise ValueError("events must be non-negative integer labels.")

        n_causes = self.n_causes if self.n_causes is not None else int(events.max())
        if n_causes < 1:
            raise ValueError("Specify n_causes when all observations are censored.")
        if np.any(events > n_causes):
            raise ValueError("Event labels must not exceed n_causes.")

        observed = events > 0
        unique_times, groups = np.unique(times[observed], return_inverse=True)
        at_risk = times.size - np.searchsorted(np.sort(times), unique_times)
        counts = np.zeros((n_causes, unique_times.size), dtype=float)
        np.add.at(counts, (events[observed].astype(np.intp) - 1, groups), 1)

        survival = np.cumprod(1 - counts.sum(axis=0) / at_risk)
        survival_before = np.r_[1.0, survival][:-1]
        cifs = counts * (survival_before / at_risk)
        np.cumsum(cifs, axis=1, out=cifs)

        self.n_causes_ = n_causes
        self.unique_times_ = unique_times
        self.surv_ = survival
        self.cif_ = cifs
        self._at_risk = at_risk
        self._event_counts = counts
        return self

    def predict_surv(self, t: ArrayLike) -> float | np.ndarray:
        """
        Return survival at scalar or array times, preserving the input shape.

        Predictions include events at t (right continuity), equal 1 before
        the first event, and remain constant after the last event. Positive
        infinity returns the final estimate; negative times and NaN are invalid.

        Parameters
        ----------
        t : array_like
            Non-negative query times; positive infinity requests the final fitted estimate.

        Returns
        -------
        float | ndarray
            Survival probabilities with the query shape, or a float for a scalar query.
        """
        survival = _predict_step(t, self.unique_times_, self.surv_, 1.0)
        return survival.item() if survival.ndim == 0 else survival

    def predict_cif(self, t: ArrayLike) -> np.ndarray:
        """
        Return CIFs with shape ``shape(t) + (K,)``.

        Uses the same right-continuous step convention as ``predict_surv``;
        all CIFs are zero before the first event.

        Parameters
        ----------
        t : array_like
            Non-negative query times; positive infinity requests the final fitted estimate.

        Returns
        -------
        ndarray
            Cause-specific cumulative incidence probabilities with shape shape(t) + (n_causes,).
        """
        values = None if self.cif_ is None else self.cif_.T
        return _predict_step(t, self.unique_times_, values, 0.0)

    def predict_P(self, t: ArrayLike) -> np.ndarray:
        """
        Return transition matrices with shape ``shape(t) + (K + 1, K + 1)``.

        Row 0 contains survival and CIFs. Each absorbing state's row is its
        identity row. Before the first event the entire matrix is the identity.
        Matrices are constructed on demand.

        Parameters
        ----------
        t : array_like
            Non-negative query times; positive infinity requests the final fitted estimate.

        Returns
        -------
        ndarray
            Transition probability matrices with the query shape followed by the two state axes.
        """
        survival = self.predict_surv(t)
        n_states = self.n_causes_ + 1
        matrices = np.broadcast_to(
            np.eye(n_states), np.shape(survival) + (n_states, n_states)
        ).copy()
        matrices[..., 0, 0] = survival
        matrices[..., 0, 1:] = self.predict_cif(t)
        return matrices

    def predict_cif_variance(self, t: float | np.ndarray) -> np.ndarray:
        """
        Predict Greenwood-type variances of the fitted cause-specific CIFs.

        Parameters
        ----------
        t : float | ndarray
            Non-negative times, including positive infinity. Queries include
            events at the given time, with simultaneous updates for tied causes.

        Returns
        -------
        ndarray
            Variances with shape shape(t) + (n_causes,), matching predict_cif.
            Values are zero before the first event and stay constant after
            the final event. Calculates marginal variances on demand from the
            fitted counts and risk sets, without refitting. A single cause
            reduces to Kaplan–Meier Greenwood variance.

        Raises
        ------
        RuntimeError
            If fit has not been called.
        ValueError
            If times are negative or NaN.

        Notes
        -----
        The method is described in [1]_.

        References
        ----------
        .. [1] Allignol, Schumacher and Beyersmann (2010). "A Note on Variance Estimation
           of the Aalen-Johansen Estimator ...". Biometrical Journal, 52(1), 126-137.
           https://doi.org/10.1002/bimj.200900039
        """
        cifs = self.predict_cif(t)
        times = np.asarray(t, dtype=float)
        risk = self._at_risk.astype(float, copy=False)
        total = self._event_counts.sum(axis=0)
        survival_before = np.r_[1.0, self.surv_][:-1]
        cause_fraction = self._event_counts / total
        # Orthogonal cause-split and total-event terms keep contributions non-negative.
        split_variance = (
            (survival_before / risk) ** 2 * self._event_counts * (1 - cause_fraction)
        )
        event_weight = np.divide(
            total, risk * (risk - total), out=np.zeros(risk.shape), where=risk > total
        )
        event_term = survival_before * cause_fraction * (1 - total / risk)
        stops, inverse = np.unique(
            np.searchsorted(self.unique_times_, times.ravel(), side="right"),
            return_inverse=True,
        )
        result = np.zeros((stops.size, self.n_causes_), dtype=float)
        for i, stop in enumerate(stops):
            if stop == 0:
                continue
            remaining = self.cif_[:, stop - 1, None] - self.cif_[:, :stop]
            result[i] = np.sum(
                split_variance[:, :stop]
                + (event_term[:, :stop] - remaining) ** 2 * event_weight[:stop],
                axis=1,
            )
            if self.surv_[stop - 1] == 0:
                # A sole observed cause with exhausted survival has exactly zero variance.
                sole_cause = np.all(
                    self._event_counts[:, :stop] == total[:stop], axis=1
                )
                result[i, sole_cause] = 0.0
        return result[inverse].reshape(cifs.shape)
