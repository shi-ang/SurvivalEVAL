import warnings
from dataclasses import dataclass, InitVar, field
from scipy.integrate import trapezoid
import numpy as np
from typing import Optional

from SurvivalEVAL.Evaluations.util import get_prob_at_zero


def km_mean(
        times: np.ndarray,
        survival_probabilities: np.ndarray
) -> float:
    """
    Calculate the mean of the Kaplan-Meier curve.

    Parameters
    ----------
    times: np.ndarray, shape = (n_samples, )
        Survival times for KM curve of the testing samples
    survival_probabilities: np.ndarray, shape = (n_samples, )
        Survival probabilities for KM curve of the testing samples

    Returns
    -------
    The mean of the Kaplan-Meier curve.
    """
    # calculate the area under the curve for each interval
    area_probabilities = np.append(1, survival_probabilities)
    area_times = np.append(0, times)
    km_linear_zero = -1 / ((area_probabilities[-1] - 1) / area_times[-1])
    if survival_probabilities[-1] != 0:
        area_times = np.append(area_times, km_linear_zero)
        area_probabilities = np.append(area_probabilities, 0)
    area_diff = np.diff(area_times, 1)
    # we are using trap rule
    average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
    area = np.flip(np.flip(area_diff * average_probabilities).cumsum())
    area = np.append(area, 0)
    # or the step function rule (deprecated for now)
    # area_subs = area_diff * area_probabilities[0:-1]
    # area_subs[-1] = area_subs[-1] / 2
    # area = np.flip(np.flip(area_subs).cumsum())

    # calculate the mean
    surv_prob = get_prob_at_zero(times, survival_probabilities)
    return area[0] / surv_prob


@dataclass
class KaplanMeier:
    """
    This class is borrowed from survival_evaluation package.
    """
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]

    # learned / derived attributes
    survival_times: np.array = field(init=False)
    population_count: np.array = field(init=False)
    events: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)
    cumulative_dens: np.array = field(init=False)
    probability_dens: np.array = field(init=False)

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
        self.events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        event_ratios = 1 - self.events / self.population_count
        self.survival_probabilities = np.cumprod(event_ratios)
        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

    def predict(self, prediction_times: np.array) -> np.array:
        """
        Predict the survival probabilities at the given prediction times.
        Parameters
        ----------
        prediction_times: np.array
            The times at which to predict the survival probabilities.
        Returns
        -------
        np.array
            The predicted survival probabilities at the given times.
        """
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

        return probabilities


@dataclass
class KaplanMeierArea(KaplanMeier):
    area_times: np.array = field(init=False)
    area_probabilities: np.array = field(init=False)
    area: np.array = field(init=False)
    km_linear_zero: float = field(init=False)

    def __post_init__(self, event_times, event_indicators):
        super().__post_init__(event_times, event_indicators)
        area_probabilities = np.append(1, self.survival_probabilities)
        area_times = np.append(0, self.survival_times)
        self.km_linear_zero = area_times[-1] / (1 - area_probabilities[-1])
        if self.survival_probabilities[-1] != 0:
            area_times = np.append(area_times, self.km_linear_zero)
            area_probabilities = np.append(area_probabilities, 0)

        # we are facing the choice of using the trapzoidal rule or directly using the area under the step function
        # we choose to use trapz because it is more accurate
        area_diff = np.diff(area_times, 1)
        average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
        area = np.flip(np.flip(area_diff * average_probabilities).cumsum())
        # area = np.flip(np.flip(area_diff * area_probabilities[0:-1]).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)

    @property
    def mean(self):
        return self.best_guess(np.array([0])).item()

    def best_guess(self, censor_times: np.array):
        # calculate the slope using the [0, 1] - [max_time, S(t|x)]
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))
        # if after the last time point, then the best guess is the linear function
        before_last_idx = censor_times <= max(self.survival_times)
        after_last_idx = censor_times > max(self.survival_times)
        surv_prob = np.empty_like(censor_times).astype(float)
        surv_prob[after_last_idx] = 1 + censor_times[after_last_idx] * slope
        surv_prob[before_last_idx] = self.predict(censor_times[before_last_idx])
        # do not use np.clip(a_min=0) here because we will use surv_prob as the denominator,
        # if surv_prob is below 0 (or 1e-10 after clip), the nominator will be 0 anyway.
        surv_prob = np.clip(surv_prob, a_min=1e-10, a_max=None)

        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )

        # for those beyond the end point, censor_area = 0
        beyond_idx = censor_indexes > len(self.area_times) - 2
        censor_area = np.zeros_like(censor_times).astype(float)
        # trapzoidal rule:  (x1 - x0) * (f(x0) + f(x1)) * 0.5
        censor_area[~beyond_idx] = ((self.area_times[censor_indexes[~beyond_idx]] - censor_times[~beyond_idx]) *
                                    (self.area_probabilities[censor_indexes[~beyond_idx]] + surv_prob[~beyond_idx])
                                    * 0.5)
        censor_area[~beyond_idx] += self.area[censor_indexes[~beyond_idx]]
        return censor_times + censor_area / surv_prob

    def _km_linear_predict(self, times):
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))

        predict_prob = np.empty_like(times)
        before_last_time_idx = times <= max(self.survival_times)
        after_last_time_idx = times > max(self.survival_times)
        predict_prob[before_last_time_idx] = self.predict(times[before_last_time_idx])
        predict_prob[after_last_time_idx] = np.clip(1 + times[after_last_time_idx] * slope, a_min=0, a_max=None)
        # if time <= max(self.survival_times):
        #     predict_prob = self.predict(time)
        # else:
        #     predict_prob = max(1 + time * slope, 0)
        return predict_prob

    def _compute_best_guess(self, time: float, restricted: bool = False):
        """
        Given a censor time, compute the decensor event time based on the residual mean survival time on KM curves.
        :param time:
        :return:
        """
        # Using integrate.quad from Scipy should be more accurate, but also making the program unbearably slow.
        # The compromised method uses numpy.trapz to approximate the integral using composite trapezoidal rule.
        warnings.warn("This method is deprecated. Use best_guess instead.", DeprecationWarning)
        if restricted:
            last_time = max(self.survival_times)
        else:
            last_time = self.km_linear_zero
        time_range = np.linspace(time, last_time, 2000)
        if self.predict(time) == 0:
            best_guess = time
        else:
            best_guess = time + trapezoid(self._km_linear_predict(time_range), time_range) / self.predict(time)

        return best_guess

    def best_guess_revise(self, censor_times: np.array, restricted: bool = False):
        warnings.warn("This method is deprecated. Use best_guess instead.", DeprecationWarning)
        bg_times = np.zeros_like(censor_times)
        for i in range(len(censor_times)):
            bg_times[i] = self._compute_best_guess(censor_times[i], restricted=restricted)
        return bg_times


@dataclass
class NelsonAalen:
    """
    Implementation of the Nelson-Aalen estimator for cumulative hazard function.
    """
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    survival_times: np.array = field(init=False)
    population_count: np.array = field(init=False)
    events: np.array = field(init=False)
    hazard: np.array = field(init=False)
    cumulative_hazard: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)

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
        self.events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        self.hazard = self.events / self.population_count
        self.cumulative_hazard = np.cumsum(self.hazard)
        self.survival_probabilities = np.exp(-self.cumulative_hazard)

    def predict(self, prediction_times: np.array) -> np.array:
        """
        Predict the cumulative hazard based on the survival times.
        Parameters
        ----------
        prediction_times: np.array
            The times at which to predict the cumulative hazard.
        Returns
        -------
        np.array
            The predicted cumulative hazard at the given times.
        """
        hazard_index = np.digitize(prediction_times, self.survival_times)
        hazard_index = np.where(
            hazard_index == self.survival_times.size + 1,
            hazard_index - 1,
            hazard_index,
        )
        hazards = np.append(0, self.cumulative_hazard)[hazard_index]
        return hazards

    def predict_survival(self, prediction_times: np.array) -> np.array:
        """
        Predict the survival probabilities at the given prediction times.
        Parameters
        ----------
        prediction_times: np.array
            The times at which to predict the survival probabilities.
        Returns
        -------
        np.array
            The predicted survival probabilities at the given times.
        """
        cum_hazards = self.predict(prediction_times)
        survival_probabilities = np.exp(-cum_hazards)
        return survival_probabilities

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
    event_times: InitVar[np.array]
    event_indicators: InitVar[np.array]
    alpha: InitVar[float]
    type: InitVar[str] = "Clayton"
    n_samples: int = field(init=False)
    survival_times: np.array = field(init=False)
    population_count: np.array = field(init=False)
    events: np.array = field(init=False)
    survival_probabilities: np.array = field(init=False)

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
        self.events = np.add.reduceat(np.append(event_indicators[index], 0), event_ind)[::2]

        event_diff = self.population_count - self.events

        with np.errstate(divide="ignore"):
            # ignore division by zero warnings,
            # such warnings are expected when the last time point has an event so the event_diff is 0.
            # but we will set the last point to 0 anyway.
            if type == "Clayton":
                diff_ = (event_diff / self.n_samples) ** (- alpha) - (self.population_count / self.n_samples) ** (- alpha)
                diff_[-1] = 0
                self.survival_probabilities = (1.0 + np.cumsum(diff_)) ** ( - 1.0 / alpha)
            elif type == "Gumbel":
                diff_ = ((- np.log(event_diff / self.n_samples)) ** (alpha + 1) -
                         (-np.log(self.population_count / self.n_samples)) ** (alpha + 1))
                diff_[-1] = 0
                self.survival_probabilities = np.exp(  -np.cumsum(diff_) ** (1 / (1 + alpha))  )
            elif type == "Frank":
                log_diff_ = np.log(  (np.exp(-alpha * event_diff / self.n_samples) - 1) / (np.exp(-alpha * self.population_count / self.n_samples) - 1)  )
                log_diff_[-1] = 0
                self.survival_probabilities = -1 / alpha * np.log(  1 + (np.exp(-alpha) - 1) * np.exp(np.cumsum(log_diff_))  )
            else:
                raise ValueError(f"Unknown copula type: {type}. Supported types are 'Clayton', 'Gumbel', and 'Frank'.")

        self.cumulative_dens = 1 - self.survival_probabilities
        self.probability_dens = np.diff(np.append(self.cumulative_dens, 1))

    def predict(self, prediction_times: np.array) -> np.array:
        """
        Predict the survival probabilities at the given prediction times.
        Parameters
        ----------
        prediction_times: np.array
            The times at which to predict the survival probabilities.
        Returns
        -------
        np.array
            The predicted survival probabilities at the given times.
        """
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]

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

# TODO: Turnbull estimator

if __name__ == "__main__":
    ### test the Nelson-Aalen estimator and compare it with the lifelines implementation
    from lifelines import NelsonAalenFitter
    import numpy as np

    # generate some synthetic data
    np.random.seed(42)
    n_samples = 1000
    event_times = np.random.exponential(scale=10, size=n_samples)
    event_indicators = np.random.binomial(n=1, p=0.7, size=n_samples)
    event_indicators = (event_indicators >= 0.5).astype(int)

    # create the Nelson-Aalen estimator
    na_estimator = NelsonAalen(event_times, event_indicators)
    # create the lifelines Nelson-Aalen fitter
    na_fitter = NelsonAalenFitter()
    na_fitter.fit(event_times, event_indicators)

    # compare the cumulative hazard functions
    times = np.linspace(0, 30, 100)
    na_cumulative_hazard = na_estimator.predict(times)
    lifelines_cumulative_hazard = na_fitter.cumulative_hazard_at_times(times).values

    # make some predictions at random times
    random_times = np.random.uniform(0, 100, size=10)
    na_predictions = na_estimator.predict(random_times)
    lifelines_predictions = na_fitter.predict(random_times)

    mse = np.mean((na_cumulative_hazard - lifelines_cumulative_hazard) ** 2)

    ### Test the Copula Graphical estimator
    times = np.array([1, 3, 5, 4, 4, 7, 8, 10, 13, 15])
    events = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    clayton_estimator = CopulaGraphic(event_times, event_indicators, alpha=18, type="Clayton")

    gumbel_estimator = CopulaGraphic(times, events, alpha=18, type="Gumbel")

    frank_estimator = CopulaGraphic(times, events, alpha=18, type="Frank")


def initialise_p(
        tau: np.ndarray
) -> np.ndarray:
    """
    Initialize the interval masses p uniformly for every interval (tau[j], tau[j+1]].
    """
    m = len(tau)
    if m < 2:
        raise ValueError("tau must contain at least two unique points.")
    return np.full(m - 1, 1.0 / (m - 1), dtype=float)


def build_alphas(
        left: np.ndarray,
        right: np.ndarray,
        tau: np.ndarray
) -> np.ndarray:
    """
    For i-th sample, and j-th unique time (tau):
    alpha[i, j] = 1 if (tau[j], tau[j+1]] lies within [left_i, right_i].
    Rows with all zeros are removed.
    """
    left = left[:, None]  # shape (n, 1)
    right = right[:, None]  # shape (n, 1)

    tau_lo = tau[:-1][None, :]  # shape (1, m-1)
    tau_hi = tau[1:][None, :]   # shape (1, m-1)

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
    probability_dens_: Optional[np.ndarray] = field(init=False, default=None)        # interval masses, length m-1
    survival_times_: Optional[np.ndarray] = field(init=False, default=None)     # plotting x (tau possibly truncated)
    survival_probabilities_: Optional[np.ndarray] = field(init=False, default=None)     # step survival, length len(time_)
    n_iter_: int = field(init=False, default=0)
    max_diff_: float = field(init=False, default=np.nan)

    def fit(
        self,
        left: np.ndarray,
        right: np.ndarray,
        tau: Optional[np.ndarray] = None,
        p_init: Optional[np.ndarray] = None
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
            tau_vals = np.concatenate([left, right[np.isfinite(right)]])
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
        # Possibly truncate at the max finite right time
        if np.any(~np.isfinite(right)):
            t_max_finite = np.max(right[np.isfinite(right)])
            mask = tau < t_max_finite
            time_out = tau[mask]
            surv_out = surv_full[mask]
        else:
            time_out = tau
            surv_out = surv_full

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


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import pandas as pd

    os.chdir("../..")
    data = pd.read_csv("data/breast.csv")
    data.right = data.right.fillna(np.inf)

    # group1
    data1 = data.loc[data["ther"] == 1].copy()
    tb1 = TurnbullEstimator().fit(data1.left.values, data1.right.values)

    # group2
    data2 = data.loc[data["ther"] == 0].copy()
    tb2 = TurnbullEstimator().fit(data2.left.values, data2.right.values)

    # plotting
    plt.figure(figsize=(7, 5))
    plt.step(tb1.survival_times_, tb1.survival_probabilities_, where="post", linestyle="-", label="Radiotherapy (intervals)")
    plt.step(tb2.survival_times_, tb2.survival_probabilities_, where="post", linestyle="-", label="Radio + Chemo (intervals)")
    plt.xlabel("Time")
    plt.ylabel("S(t)")
    plt.legend()
    plt.title("Turnbull Interval-Censored Survival")
    plt.tight_layout()
    plt.show()

    # compare midpoint-based KM with Turnbull
    # Midpoints:
    p_mid = data["left"].to_numpy(float) + (data["right"].to_numpy(float) - data["left"].to_numpy(float)) / 2.0
    finite_mid = np.isfinite(p_mid)
    pm = np.where(finite_mid, p_mid, data["left"].to_numpy(float))
    cens = finite_mid.astype(int)  # 1 == event, 0 == right-censored

    # KM by group
    km1 = KaplanMeier(pm[data["ther"] == 1], cens[data["ther"] == 1])
    km0 = KaplanMeier(pm[data["ther"] == 0], cens[data["ther"] == 0])
    times1, surv1 = km1.survival_times, km1.survival_probabilities
    times0, surv0 = km0.survival_times, km0.survival_probabilities

    plt.figure(figsize=(7, 5))
    # Interval-censored (solid)
    plt.step(tb1.survival_times_, tb1.survival_probabilities_, where="post", linestyle="-", label="Radiotherapy (intervals)")
    plt.step(tb2.survival_times_, tb2.survival_probabilities_, where="post", linestyle="-", label="Radio + Chemo (intervals)")
    # Midpoint-based KM (dashed)
    if times1.size:
        plt.step(times1, surv1, where="post", linestyle="--", label="Radiotherapy (midpoints)")
    if times0.size:
        plt.step(times0, surv0, where="post", linestyle="--", label="Radio + Chemo (midpoints)")

    plt.xlabel("Time")
    plt.ylabel("S(t)")
    plt.legend()
    plt.title("Interval-Censored (Turnbull) vs Midpoint KM")
    plt.tight_layout()
    plt.show()
