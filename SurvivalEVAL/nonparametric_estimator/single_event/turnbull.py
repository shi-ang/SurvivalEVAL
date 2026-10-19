from __future__ import annotations

from dataclasses import InitVar, dataclass, field

import numpy as np
from lifelines import KaplanMeierFitter

from SurvivalEVAL.nonparametric_estimator.util import (
    infer_survival_probabilities,
)


def create_support_intervals(tau: np.ndarray, exact_times: np.ndarray) -> np.ndarray:
    """
    Create the support cells used by the Turnbull EM algorithm.

    The first cells represent the ranges between adjacent tau values. Exact
    observations add singleton cells {t}; those cells own their endpoints so
    they remain distinct from the adjacent ranges.

    Parameters
    ----------
    tau : ndarray
        Sorted support-grid endpoints, including positive infinity when required.
    exact_times : ndarray
        Distinct exact event times to represent as singleton support cells.

    Returns
    -------
    ndarray
        Adjacent grid intervals followed by singleton exact-event cells, with shape (n_cells, 2).
    """
    adjacent_intervals = np.column_stack((tau[:-1], tau[1:]))
    exact_intervals = np.column_stack((exact_times, exact_times))
    return np.concatenate((adjacent_intervals, exact_intervals), axis=0)


def initialise_p(support_intervals: np.ndarray) -> np.ndarray:
    """
    Initialize masses uniformly over the support cells.

    Parameters
    ----------
    support_intervals : ndarray
        Support-cell bounds with shape (n_cells, 2), including singleton cells for exact events.

    Returns
    -------
    ndarray
        Uniform positive masses over all support cells, summing to one.
    """
    num_intervals = len(support_intervals)
    if num_intervals == 0:
        raise ValueError("At least one support interval is required.")
    return np.full(num_intervals, 1.0 / num_intervals, dtype=float)


def build_alphas(
    left: np.ndarray,
    right: np.ndarray,
    support_intervals: np.ndarray,
) -> np.ndarray:
    """
    Build the incidence matrix between observations and support cells.

    Non-exact observations use (left_i, right_i]. Exact observations use the
    singleton {right_i}, represented by left_i == right_i.

    Parameters
    ----------
    left : ndarray
        Left observation bounds; equality with right denotes an exact event.
    right : ndarray
        Right observation bounds; positive infinity denotes right censoring.
    support_intervals : ndarray
        Support-cell bounds with shape (n_cells, 2), including singleton cells for exact events.

    Returns
    -------
    ndarray
        Observation-by-support incidence matrix with shape (n_samples, n_cells).
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
    Estimate survival nonparametrically from interval-censored observations.

    Non-exact observations use left-open, right-closed intervals (left, right].
    Observations with left == right are exact event times. The EM algorithm follows
    Turnbull [1]_; implementation notes are available in [2]_.

    Parameters
    ----------
    eps : float, default=1e-8
        Convergence threshold for the maximum change in support-cell masses.
    iter_max : int, default=1000
        Maximum number of expectation-maximization iterations.
    verbose : bool, default=False
        Whether to print iteration progress.

    References
    ----------
    .. [1] Turnbull, B. W. (1976). "The Empirical Distribution Function with Arbitrarily Grouped,
       Censored and Truncated Data". Journal of the Royal Statistical Society: Series B
       (Methodological), 38(3), 290-295. https://doi.org/10.1111/j.2517-6161.1976.tb01597.x
       https://academic.oup.com/jrsssb/article/38/3/290/7027379
    .. [2] Interval-censoring EM algorithm notes.
       https://www.ms.uky.edu/~mai/splus/icensem.pdf
    """

    eps: float = 1e-8
    iter_max: int = 1000
    verbose: bool = False

    # learned / derived attributes
    tau_: np.ndarray | None = field(init=False, default=None)
    probability_dens_: np.ndarray | None = field(
        init=False, default=None
    )  # masses for adjacent tau intervals, followed by exact-time atoms
    survival_times_: np.ndarray | None = field(
        init=False, default=None
    )  # plotting x (tau possibly truncated)
    survival_probabilities_: np.ndarray | None = field(
        init=False, default=None
    )  # step survival, length len(time_)
    n_iter_: int = field(init=False, default=0)
    max_diff_: float = field(init=False, default=np.nan)

    def fit(
        self,
        left: np.ndarray,
        right: np.ndarray,
        tau: np.ndarray | None = None,
        p_init: np.ndarray | None = None,
    ) -> TurnbullEstimator:
        """
        Fit the Turnbull estimator to interval-censored data.

        Parameters
        ----------
        left : ndarray
            Left limits of intervals. For exact events, use the event time in
            both left and right.
        right : ndarray
            Right limits of intervals. Can contain np.inf for right-censored.
        tau : ndarray, optional
            Unique time points for the survival function. If None, it is
            constructed as the unique sorted union of left and right,
            including np.inf when present.
        p_init : ndarray, optional
            Initial masses for adjacent tau intervals, followed by one mass
            for each unique exact event time. If None, masses are initialized
            uniformly.

        Returns
        -------
        TurnbullEstimator
            This estimator after fitting.
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

    def predict(self, prediction_times: float | np.ndarray) -> float | np.ndarray:
        """
        Predict survival probabilities at given times using the fitted Turnbull estimator.

        Parameters
        ----------
        prediction_times : float | ndarray
            Time(s) at which to predict survival probabilities.

        Returns
        -------
        probabilities : float | ndarray
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
    """
    Fit a Turnbull estimate using the upper survival-bound convention.

    The general fit delegates to ``lifelines.KaplanMeierFitter.fit_interval_censoring`` [1]_.

    If all observation intervals intersect, a point mass at their common
    upper endpoint maximizes the likelihood. Construct that estimate directly
    rather than fitting a degenerate model. An infinite common upper endpoint
    leaves survival equal to 1 at every finite time.

    Parameters
    ----------
    left : ndarray, shape (n_samples,)
        Left observation bounds.
    right : ndarray, shape (n_samples,)
        Right observation bounds; infinity denotes right censoring.
    alpha : float, default=0.05
        Significance level passed to the lifelines estimator.
    tol : float, default=1e-5
        Convergence tolerance for the fitted interval-censoring model.
    label : str, default="Turnbull"
        Label used for the fitted survival function.

    References
    ----------
    .. [1] lifelines documentation. "KaplanMeierFitter.fit_interval_censoring".
       https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html
    """

    left: InitVar[np.ndarray]
    right: InitVar[np.ndarray]
    alpha: InitVar[float] = 0.05
    tol: InitVar[float] = 1e-5
    label: InitVar[str] = "Turnbull"

    # learned / derived attributes
    probability_dens: np.ndarray | None = field(init=False, default=None)
    cumulative_dens: np.ndarray | None = field(init=False, default=None)
    survival_times: np.ndarray | None = field(init=False, default=None)
    survival_probabilities: np.ndarray | None = field(init=False, default=None)

    def __post_init__(
        self,
        left: np.ndarray,
        right: np.ndarray,
        alpha: float,
        tol: float,
        label: str,
    ) -> None:
        """
        Fit the TurnbullEstimatorLifelines estimator from the supplied observations.

        Parameters
        ----------
        left : ndarray
            Left observation bounds; equality with right denotes an exact event.
        right : ndarray
            Right observation bounds; positive infinity denotes right censoring.
        alpha : float
            Confidence-interval significance level.
        tol : float
            Convergence tolerance passed to the interval-censoring estimator.
        label : str
            Label assigned to the fitted survival function.
        """
        common_left, common_right = np.max(left), np.min(right)
        if common_left <= common_right:
            self.survival_times = np.unique([common_left, common_right]).astype(float)
            self.survival_probabilities = (self.survival_times < common_right).astype(
                float
            )
        else:
            kmf = KaplanMeierFitter(alpha=alpha)
            kmf.fit_interval_censoring(left, right, label=label, tol=tol)
            self.survival_times = kmf.survival_function_.index.values
            self.survival_probabilities = kmf.survival_function_[
                f"{label}_upper"
            ].values

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

    def predict(self, prediction_times: float | np.ndarray) -> float | np.ndarray:
        """
        Predict the survival probabilities at the given prediction times.

        Parameters
        ----------
        prediction_times : float | ndarray
            Time(s) at which to predict the survival probabilities.

        Returns
        -------
        probabilities : float | ndarray
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
