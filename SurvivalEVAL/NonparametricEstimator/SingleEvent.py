import warnings
from dataclasses import dataclass, InitVar, field

import numpy as np

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
            best_guess = time + np.trapezoid(self._km_linear_predict(time_range), time_range) / self.predict(time)

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
