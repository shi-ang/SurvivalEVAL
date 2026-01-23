from dataclasses import dataclass, field, InitVar
from typing import Optional

import numpy as np
from lifelines import KaplanMeierFitter

from SurvivalEVAL.NonparametricEstimator.SingleEvent.util import infer_survival_probabilities


def initialise_p(tau: np.ndarray) -> np.ndarray:
    """
    Initialize the interval masses p uniformly for every interval (tau[j], tau[j+1]].
    """
    m = len(tau)
    if m < 2:
        raise ValueError("tau must contain at least two unique points.")
    return np.full(m - 1, 1.0 / (m - 1), dtype=float)


def build_alphas(left: np.ndarray, right: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    For i-th sample, and j-th unique time (tau):
    alpha[i, j] = 1 if (tau[j], tau[j+1]] lies within [left_i, right_i].
    Rows with all zeros are removed.
    """
    left = left[:, None]  # shape (n, 1)
    right = right[:, None]  # shape (n, 1)

    tau_lo = tau[:-1][None, :]  # shape (1, m-1)
    tau_hi = tau[1:][None, :]  # shape (1, m-1)

    A = ((tau_lo >= left) & (tau_hi <= right)).astype(float)  # (n, m-1)

    # Drop rows that are all zeros
    keep = ~np.all(A == 0.0, axis=1)
    A = A[keep, :]
    return A


@dataclass
class TurnbullEstimator:
    """
    Turnbull non-parametric estimator for interval-censored survival data.

    https://www.ms.uky.edu/~mai/splus/icensem.pdf
    """

    eps: float = 1e-8
    iter_max: int = 1000
    verbose: bool = False

    # learned / derived attributes
    tau_: Optional[np.ndarray] = field(init=False, default=None)
    probability_dens_: Optional[np.ndarray] = field(
        init=False, default=None
    )  # interval masses, length m-1
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
            Left limits of intervals.
        right : np.ndarray
            Right limits of intervals. Can contain np.inf for right-censored.
        tau : np.ndarray, optional
            Unique time points for the survival function.
            If None, it is constructed as the unique sorted union of left and right (excluding np.inf).
        p_init : np.ndarray, optional
            Initial interval masses for the intervals defined by tau.
            If None, it is initialized uniformly across intervals.
        """
        if tau is None:
            tau_vals = np.concatenate([left, right])
            tau = np.unique(np.sort(tau_vals))
        else:
            tau = np.asarray(tau, dtype=float).copy()

        alphas = build_alphas(left, right, tau)
        n, m_minus_1 = alphas.shape

        # Initialize the density p
        if p_init is None:
            p = initialise_p(tau)
        else:
            p = np.asarray(p_init, dtype=float).copy()
            if p.shape[0] != m_minus_1:
                raise ValueError("p_init must have length len(tau)-1.")
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

        # Compute survival step function: surv = [1] + 1 - cumsum(p)
        surv_full = np.concatenate([[1.0], 1.0 - np.cumsum(p)])
        # Possibly truncate at the max finite time
        finite_mask = np.isfinite(tau)
        if finite_mask.sum() < len(tau):
            time_out = tau[finite_mask]
            surv_out = surv_full[finite_mask]
        else:
            time_out = tau
            surv_out = surv_full

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

    def predict(self, prediction_times: int | float | np.ndarray) -> float | np.ndarray:
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
