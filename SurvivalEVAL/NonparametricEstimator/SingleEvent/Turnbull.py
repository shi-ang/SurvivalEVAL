from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Optional, Union

import numpy as np
from lifelines import KaplanMeierFitter

from SurvivalEVAL.NonparametricEstimator.SingleEvent.util import (
    infer_survival_probabilities,
)


def create_support_intervals(tau: np.ndarray, exact_times: np.ndarray) -> np.ndarray:
    """
    Create the support cells used by the Turnbull EM algorithm.

    The first cells represent the ranges between adjacent tau values. Exact
    observations add singleton cells {t}; those cells own their endpoints so
    they remain distinct from the adjacent ranges.
    """
    adjacent_intervals = np.column_stack((tau[:-1], tau[1:]))
    exact_intervals = np.column_stack((exact_times, exact_times))
    return np.concatenate((adjacent_intervals, exact_intervals), axis=0)


def initialise_p(support_intervals: np.ndarray) -> np.ndarray:
    """Initialize masses uniformly over the support cells."""
    num_intervals = len(support_intervals)
    if num_intervals == 0:
        raise ValueError("At least one support interval is required.")
    return np.full(num_intervals, 1.0 / num_intervals, dtype=float)


def build_alphas(
    left: np.ndarray, right: np.ndarray, support_intervals: np.ndarray
) -> np.ndarray:
    """
    Build the incidence matrix between observations and support cells.

    Non-exact observations use (left_i, right_i]. Exact observations use the
    singleton {right_i}, represented by left_i == right_i.
    """
    left = left[:, None]  # shape (n, 1)
    right = right[:, None]  # shape (n, 1)

    interval_left = support_intervals[:, 0][None, :]
    interval_right = support_intervals[:, 1][None, :]
    observation_is_exact = left == right
    interval_is_exact = interval_left == interval_right

    exact_contains = (
        observation_is_exact & interval_is_exact & (interval_right == right)
    )
    lower_is_contained = np.where(
        interval_is_exact,
        interval_left > left,
        interval_left >= left,
    )
    interval_contains = (
        (~observation_is_exact) & lower_is_contained & (interval_right <= right)
    )
    alphas = (exact_contains | interval_contains).astype(float)

    if np.any(np.all(alphas == 0.0, axis=1)):
        raise ValueError("Each observation must contain at least one support interval.")
    return alphas


@dataclass
class TurnbullEstimator:
    """
    Turnbull non-parametric estimator for interval-censored survival data.

    Non-exact observations use left-open, right-closed intervals (left, right].
    Observations with left == right are treated as exact event times.

    https://www.ms.uky.edu/~mai/splus/icensem.pdf
    """

    eps: float = 1e-8
    iter_max: int = 1000
    verbose: bool = False

    # learned / derived attributes
    tau_: Optional[np.ndarray] = field(init=False, default=None)
    probability_dens_: Optional[np.ndarray] = field(
        init=False, default=None
    )  # masses for adjacent tau intervals, followed by exact-time atoms
    survival_times_: Optional[np.ndarray] = field(
        init=False, default=None
    )  # plotting x (tau possibly truncated)
    survival_probabilities_: Optional[np.ndarray] = field(
        init=False, default=None
    )  # step survival, length len(time_)
    n_iter_: int = field(init=False, default=0)
    max_diff_: float = field(init=False, default=np.nan)

    def fit(
        self,
        left: np.ndarray,
        right: np.ndarray,
        tau: Optional[np.ndarray] = None,
        p_init: Optional[np.ndarray] = None,
    ) -> "TurnbullEstimator":
        """
        Fit the Turnbull estimator to interval-censored data.
        Parameters
        ----------
        left : np.ndarray
            Left limits of intervals. For exact events, use the event time in
            both left and right.
        right : np.ndarray
            Right limits of intervals. Can contain np.inf for right-censored.
        tau : np.ndarray, optional
            Unique time points for the survival function. If None, it is
            constructed as the unique sorted union of left and right,
            including np.inf when present.
        p_init : np.ndarray, optional
            Initial masses for adjacent tau intervals, followed by one mass
            for each unique exact event time. If None, masses are initialized
            uniformly.
        """
        exact_times = np.unique(left[left == right])
        if tau is None:
            tau_vals = np.concatenate([left, right])
            tau = np.unique(np.sort(tau_vals))
        else:
            tau = np.asarray(tau, dtype=float).copy()
            if exact_times.size:
                tau = np.unique(np.sort(np.concatenate([tau, exact_times])))

        support_intervals = create_support_intervals(tau, exact_times)
        alphas = build_alphas(left, right, support_intervals)
        n, num_intervals = alphas.shape

        # Initialize the density p
        if p_init is None:
            p = initialise_p(support_intervals)
        else:
            p = np.asarray(p_init, dtype=float).copy()
            if p.shape[0] != num_intervals:
                raise ValueError("p_init must have one mass per support interval.")
            if (p <= 0).any():
                raise ValueError("p_init must be strictly positive.")
            p = p / p.sum()

        # EM iterations
        Q = np.ones_like(p)
        iter_count = 0
        maxdiff = np.inf

        while iter_count < self.iter_max:
            iter_count += 1
            diff = Q - p
            maxdiff = float(np.max(np.abs(diff)))
            if self.verbose:
                print(f"Iter {iter_count:3d} | maxdiff={maxdiff:.6g}")
            if maxdiff < self.eps:
                break

            Q = p.copy()
            C = alphas @ p  # shape (n,)
            if np.any(C <= 0):
                # Safeguard; should not happen if A rows are valid and p>0
                raise RuntimeError("Zero or negative A @ p encountered.")

            # EM update: p <- p * (t(A) %*% (1/C)) / n
            p = p * ((alphas.T @ (1.0 / C)) / n)
            # No explicit normalization needed; the update preserves sum to 1 in theory.
            # Numerically, we can renormalize slightly:
            s = p.sum()
            if s <= 0:
                raise RuntimeError("p collapsed to zero measure.")
            p /= s

        # Place each support cell's mass at its right endpoint. This preserves
        # the original step-function convention while allowing exact atoms.
        time_out = tau[np.isfinite(tau)]
        interval_ends = support_intervals[:, 1]
        cumulative_dens = (interval_ends[None, :] <= time_out[:, None]) @ p
        surv_out = np.clip(1.0 - cumulative_dens, 0.0, 1.0)

        # Add a point at time 0 if not present
        if time_out[0] > 0:
            time_out = np.insert(time_out, 0, 0.0)
            surv_out = np.insert(surv_out, 0, 1.0)

        self.tau_ = tau
        self.probability_dens_ = p
        self.survival_times_ = time_out
        self.survival_probabilities_ = surv_out
        self.n_iter_ = iter_count
        self.max_diff_ = maxdiff

        if self.verbose:
            print(f"Iterations = {self.n_iter_}")
            print(f"Max difference = {self.max_diff_}")
            print(f"Convergence criteria: Max difference < {self.eps}")

        return self

    def predict(
        self, prediction_times: Union[int, float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Predict survival probabilities at given times using the fitted Turnbull estimator.
        Parameters
        ----------
        prediction_times: int | float | np.ndarray
            Time(s) at which to predict survival probabilities.
        Returns
        -------
        probabilities: float | np.ndarray
            Predicted survival probabilities at the given times.
        """
        if self.survival_times_ is None or self.survival_probabilities_ is None:
            raise RuntimeError("The estimator must be fitted before prediction.")

        prediction_times = np.asarray(prediction_times, dtype=float)
        original_shape = prediction_times.shape
        prediction_times = prediction_times.reshape(-1)

        # ensure the prediction times are all non-negative
        if np.any(prediction_times < 0):
            raise ValueError("Prediction times must be non-negative.")

        probs = infer_survival_probabilities(
            prediction_times, self.survival_times_, self.survival_probabilities_
        )

        probabilities = probs.reshape(original_shape)
        if probabilities.ndim == 0:
            return float(probabilities)

        return probabilities


@dataclass
class TurnbullEstimatorLifelines:
    left: InitVar[np.ndarray]
    right: InitVar[np.ndarray]
    alpha: InitVar[float] = 0.05
    tol: InitVar[float] = 1e-5
    label: InitVar[str] = "Turnbull"

    # learned / derived attributes
    probability_dens: Optional[np.ndarray] = field(init=False, default=None)
    cumulative_dens: Optional[np.ndarray] = field(init=False, default=None)
    survival_times: Optional[np.ndarray] = field(init=False, default=None)
    survival_probabilities: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self, left, right, alpha, tol, label):
        kmf = KaplanMeierFitter(alpha=alpha)
        kmf.fit_interval_censoring(left, right, label=label, tol=tol)

        self.survival_times = kmf.survival_function_.index.values
        # We use the '_upper' column, as it has the same behavior as the Turnbull estimator in icensem package in R.
        self.survival_probabilities = kmf.survival_function_[f"{label}_upper"].values

        # If the last survival times is inf, we need to remove it
        if np.isinf(self.survival_times[-1]):
            self.survival_times = self.survival_times[:-1]
            self.survival_probabilities = self.survival_probabilities[:-1]

        # If the first survival time is not 0, we need to add it
        if self.survival_times[0] != 0:
            self.survival_times = np.insert(self.survival_times, 0, 0)
            self.survival_probabilities = np.insert(self.survival_probabilities, 0, 1)

        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

    def predict(
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
