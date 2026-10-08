from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from lifelines import CoxPHFitter
from patsy import dmatrix
from scipy.stats import chi2

from SurvivalEVAL.Evaluations.custom_types import Numeric, NumericArrayLike
from SurvivalEVAL.Evaluations.util import (
    check_and_convert,
    check_and_convert_event_data,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    KaplanMeier,
    TurnbullEstimatorLifelines,
)


def _probability_bins(
    preds: np.ndarray, num_bins: int, strategy: str
) -> list[np.ndarray]:
    """Group prediction indices into equal-sized or equal-width bins.

    Parameters
    ----------
    preds : np.ndarray, shape (n_samples,)
        Event probabilities in [0, 1].
    num_bins : int
        Positive number of bins to construct.
    strategy : {"c", "h"}
        "c" sorts probabilities in descending order and splits the indices
        into approximately equal-sized groups. "h" uses equal-width bins
        on [0, 1], including probability 1 in the final bin.

    Returns
    -------
    list[np.ndarray]
        One array of sample indices per bin, including empty bins.

    Raises
    ------
    TypeError
        If the strategy is neither "c" nor "h".
    """
    if strategy == "c":
        return np.array_split(np.argsort(-preds), num_bins)
    if strategy == "h":
        edges = np.linspace(0, 1, num_bins + 1)
        return [
            np.flatnonzero(
                (preds >= edges[i])
                & (preds <= edges[i + 1] if i == num_bins - 1 else preds < edges[i + 1])
            )
            for i in range(num_bins)
        ]
    raise TypeError("Please enter one of 'C','H' for binning_strategy.")


def _calibration_result(observed, expected, bin_sizes, *, uncensored=False):
    """Compute the one-calibration statistic and its chi-square p-value.

    Parameters
    ----------
    observed, expected : array-like, shape (n_bins,)
        Observed and mean predicted event probabilities for retained bins.
    bin_sizes : array-like, shape (n_bins,)
        Number of retained observations in each bin.
    uncensored : bool, default=False
        Use n_bins - 2 degrees of freedom for the uncensored method.
        Otherwise use n_bins - 1, except when n_bins > 15, which uses
        n_bins - 2.

    Returns
    -------
    tuple[float, float, array-like, array-like]
        P-value, statistic, and the original observed and expected inputs.
        A predicted probability of 0 or 1 contributes zero when it matches
        the observed probability, and infinity otherwise. The p-value is
        NaN when the degrees of freedom are nonpositive.
    """
    obs, exp = np.asarray(observed), np.asarray(expected)
    numerator = np.asarray(bin_sizes) * (obs - exp) ** 2
    variance = exp * (1 - exp)
    contributions = np.divide(
        numerator,
        variance,
        out=np.where(numerator == 0, 0.0, np.inf),
        where=variance != 0,
    )
    statistic = float(contributions.sum())
    num_bins = len(observed)
    dof = num_bins - (2 if uncensored or num_bins > 15 else 1)
    p_value = float(chi2.sf(statistic, dof)) if dof > 0 else np.nan
    return p_value, statistic, observed, expected


def _cloglog(probabilities: np.ndarray) -> np.ndarray:
    """Apply the complementary log-log transform with finite boundaries.

    Parameters
    ----------
    probabilities : np.ndarray
        Event probabilities in [0, 1].

    Returns
    -------
    np.ndarray
        Float64 values of log(-log(1 - p)), with the input shape preserved.
        Probabilities are clipped to [1e-10, 1 - 1e-10] after conversion to
        float64 so both boundaries remain finite for float32 inputs too.
        The input array is not modified.
    """
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-10, 1 - 1e-10)
    return np.log(-np.log1p(-probabilities))


def _maximum_local_deviation(observed, expected) -> float:
    """Measure the largest adjacent calibration slope or its reciprocal.

    Parameters
    ----------
    observed, expected : array-like, shape (n_bins,)
        Observed and expected probabilities in matching bin order.

    Returns
    -------
    float
        Maximum of each slope, diff(observed) / diff(expected), and its
        reciprocal. Returns infinity for a zero slope, or NaN if fewer
        than two bins remain or adjacent expected probabilities are tied.
    """
    increments = np.diff(expected)
    if increments.size == 0 or np.any(increments == 0):
        return np.nan
    slopes = np.diff(observed) / increments
    inverse_slopes = np.divide(
        1.0, slopes, out=np.full(slopes.shape, np.inf), where=slopes != 0
    )
    return float(np.max(np.maximum(slopes, inverse_slopes)))


def one_calibration(
    preds: np.ndarray,
    event_time: np.ndarray,
    event_indicator: np.ndarray,
    target_time: Numeric,
    num_bins: int = 10,
    binning_strategy: str = "C",
    method: str = "DN",
) -> tuple[float, float, list, list]:
    """
    Compute the one calibration score for a given set of predictions and true event times.

    Parameters
    ----------
    preds: np.ndarray
        The predicted probabilities of experiencing the event at the time of interest.
    event_time: np.ndarray
        The true event times.
    event_indicator: np.ndarray
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    target_time: Numeric
        The time of interest.
    num_bins: int
        The number of bins to divide the predictions into.
    binning_strategy: str
        The strategy to bin the predictions. The options are: "C" (default), and "H".
        C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
        H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
    method: str
        The method to handle censored patients. The options are: "DN" (default), and "Uncensored".
        "Uncensored" removes observations censored before the target time, whose event status is unknown,
        and uses the standard Hosmer-Lemeshow test on the remaining observations.
        "DN" method uses the D'Agostino-Nam method, which uses the Kaplan-Meier estimate of the survival function
        to compute the average observed probabilities in each bin.

    Returns
    -------
    p_value: float
        The one calibration p-value, or NaN if too few nonempty bins remain.
    statistics: float
        The Hosmer-Lemeshow statistics.
        Boundary probabilities contribute zero when observed and expected
        probabilities agree, and infinity otherwise.
    observed_probabilities: list
        The observed probabilities in each bin.
    expected_probabilities: list
        The expected probabilities in each bin.
    """
    method = method.lower()
    if method not in {"uncensored", "dn"}:
        raise TypeError("Please enter one of 'Uncensored','DN' for method.")

    observed, expected, bin_sizes = [], [], []
    for indices in _probability_bins(preds, num_bins, binning_strategy.lower()):
        if method == "uncensored":
            indices = indices[
                ~((event_time[indices] < target_time) & (event_indicator[indices] == 0))
            ]
        if indices.size == 0:
            continue

        if method == "uncensored":
            event_probability = np.mean(event_time[indices] < target_time)
        else:
            km = KaplanMeier(event_time[indices], event_indicator[indices])
            event_probability = 1 - km.predict(target_time)
        observed.append(event_probability)
        expected.append(np.mean(preds[indices], dtype=float))
        bin_sizes.append(indices.size)

    return _calibration_result(
        observed, expected, bin_sizes, uncensored=method == "uncensored"
    )


def one_cal_ic(
    preds: np.ndarray,
    left_limits: np.ndarray,
    right_limits: np.ndarray,
    target_time: Numeric,
    num_bins: int = 10,
    binning_strategy: str = "C",
    method: str = "Turnbull",
) -> tuple[float, float, list, list]:
    """
    Compute the one calibration score for a given set of predictions and true event times.
    Parameters
    ----------
    preds: np.ndarray
        The predicted probabilities of experiencing the event at the time of interest.
    left_limits: np.ndarray
        The left limits of the interval-censored event times.
    right_limits: np.ndarray
        The right limits of the interval-censored event times.
    target_time: Numeric
        The time of interest.
    num_bins: int
        The number of bins to divide the predictions into.
    binning_strategy: str
        The strategy to bin the predictions. The options are: "C" (default), and "H".
        C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
        H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
    method: str
        The method to handle censored patients. The options are: "Turnbull" (default), and "MidPoint".
        "MidPoint" method simply treats the midpoint of the interval as the event time, and
        uses the DN's method (Kaplan-Meier estimate of the survival function).
        "Turnbull" method uses the Turnbull estimator for the survival function
        to compute the average observed probabilities in each bin.
    Returns
    -------
    p_value: float
        The one-calibration p-value, or NaN if too few nonempty bins remain.
    statistics: float
        The Hosmer-Lemeshow statistic.
        Boundary probabilities contribute zero when observed and expected
        probabilities agree, and infinity otherwise.
    observed_probabilities: list
        The observed probabilities in each bin.
    expected_probabilities: list
        The expected probabilities in each bin.
    """
    method = method.lower()
    if method not in {"midpoint", "turnbull"}:
        raise TypeError("Please enter one of 'MidPoint','Turnbull' for method.")

    observed, expected, bin_sizes = [], [], []
    for indices in _probability_bins(preds, num_bins, binning_strategy.lower()):
        if indices.size == 0:
            continue
        left, right = left_limits[indices], right_limits[indices]
        if method == "midpoint":
            mid = left + (right - left) / 2.0
            finite = np.isfinite(mid)
            estimator = KaplanMeier(np.where(finite, mid, left), finite)
        else:
            estimator = TurnbullEstimatorLifelines(left, right)
        observed.append(1 - estimator.predict(target_time))
        expected.append(np.mean(preds[indices], dtype=float))
        bin_sizes.append(indices.size)

    return _calibration_result(observed, expected, bin_sizes)


def integrated_calibration_index(
    preds: NumericArrayLike,
    event_time: NumericArrayLike,
    event_indicator: NumericArrayLike,
    target_time: Numeric,
    knots: int = 3,
    draw_figure: bool = False,
    figure_range: tuple | None = None,
) -> dict | tuple[dict, tuple[plt.Figure, plt.Axes]]:
    """
    Compute the Integrated Calibration Index (ICI) for a given set of predictions and true event times.
    The method is presented in [1]. The implementation is based on the R code available in Appendix A of [1].

    Constant predictions use the Kaplan-Meier event probability at the target
    time. Other predictions use a spline and Cox model, clipping probabilities
    only for the log transform while retaining the original values for errors.

    Parameters
    ----------
    preds: NumericArrayLike
        The predicted probabilities of experiencing the event at the time of interest.
    event_time: NumericArrayLike
        The true event times.
    event_indicator: NumericArrayLike
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    target_time: Numeric
        The time of interest for calibration.
    knots: int
        The number of knots to use for the spline basis. Default is 3.
        Austin et al. (2020) [1] compared 3-5 knots and found that 3 knots is best.
    draw_figure: bool
        Whether to plot the graphical calibration curve and return the plot. Default is False.
    figure_range: tuple
        The range of the x-axis and y-axis for the plot.
        It should be a tuple of the form (x_min, x_max, y_min, y_max).
        If None, it will be set to the range of predicted event probabilities.
        Default is None.
    Returns
    -------
    summary: dict
        A dictionary containing the integrated calibration index (ICI), E50, E90, E_max, and the information about the calibration curve.
    fig: tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes objects for the calibration curve plot. Returned only if draw_figure is True.

    References
    ----------
    [1] Austin et al., Graphical calibration curves and the integrated calibration index (ICI) for survival models.
    Stat Med. 2020
    """
    preds = check_and_convert(preds)
    event_time, event_indicator = check_and_convert_event_data(
        event_time, event_indicator
    )
    if preds.shape != event_time.shape:
        raise ValueError(
            "preds, event_time, and event_indicator must have the same shape."
        )
    if np.any((preds < 0) | (preds > 1)):
        raise ValueError("Event probabilities must be between 0 and 1.")

    grid = np.linspace(np.quantile(preds, 0.01), np.quantile(preds, 0.99), 100)
    pred_clls = _cloglog(preds)
    distinct_predictions = np.unique(pred_clls).size
    if distinct_predictions == 1:
        observed = 1 - KaplanMeier(event_time, event_indicator).predict(target_time)
        calibrated = np.full(preds.size + grid.size, observed)
    else:
        spline_df = min(knots, distinct_predictions - 1)
        spline = dmatrix(
            f"bs(x, df={spline_df}, degree={min(3, spline_df)}, include_intercept=False) - 1",
            {"x": pred_clls},
            return_type="dataframe",
        )
        data = spline.copy()
        data["time"] = event_time
        data["event"] = event_indicator
        fitter = CoxPHFitter().fit(data, duration_col="time", event_col="event")
        evaluation_spline = dmatrix(
            spline.design_info,
            {"x": _cloglog(np.concatenate([preds, grid]))},
            return_type="dataframe",
        )
        calibrated = (
            1
            - fitter.predict_survival_function(
                evaluation_spline, times=[target_time]
            ).values.flatten()
        )

    abs_err = np.abs(preds - calibrated[: preds.size])
    cal_pred = calibrated[preds.size :]
    summary = {
        "ICI": np.mean(abs_err, dtype=float),
        "E50": np.median(abs_err),
        "E90": np.quantile(abs_err, 0.9),
        "E_max": np.max(abs_err),
        "curve": {"grid": grid, "cal_pred": cal_pred},
    }

    if draw_figure:
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(grid, cal_pred, label="Calibration Curve", color="blue")
        ax.plot(grid, grid, label="Perfect Calibration", linestyle="--", color="grey")
        ax.set_xlabel("Predicted Event Probability")
        ax.set_ylabel("Observed Event Probability")
        ax.set_title("Graphical Calibration Curve")
        ax.legend()
        if figure_range is not None:
            ax.set_xlim(figure_range[0], figure_range[1])
            ax.set_ylim(figure_range[2], figure_range[3])
        return summary, (fig, ax)

    return summary
