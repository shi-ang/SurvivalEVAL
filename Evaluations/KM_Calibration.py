import numpy as np

from Evaluations.util import check_and_convert, KaplanMeier

EPS = 1e-8


def km_calibration(
        predicted_survival_curves: np.ndarray,
        event_time: np.ndarray,
        event_indicator: np.ndarray,
) -> float:
    average_survival_curve = np.mean(predicted_survival_curves, axis=0)
    average_cdf = 1 - average_survival_curve
    average_pdf = np.diff(np.append(average_cdf, 1))

    km_model = KaplanMeier(event_time, event_indicator)
    observed_pdf = km_model.probability_dens

    # calculate the KL divergence between observed and average pdf curve
    kl_divergence = np.sum(observed_pdf * np.log(observed_pdf / (average_pdf + EPS)))

    return kl_divergence


if __name__ == '__main__':
    # test the KM calibration

    # generate random survival curves
    num_curves = 1000
    num_time_points = 100
    num_bins = 10
    num_censored = 0
    num_uncensored = 0
    num_observed = 0
    num_unobserved = 0

    predicted_survival_curves = np.random.rand(num_curves, num_time_points)