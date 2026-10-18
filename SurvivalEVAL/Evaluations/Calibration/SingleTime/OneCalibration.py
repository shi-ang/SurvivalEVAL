"""Grouped fixed-time calibration for survival and competing-risk outcomes."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from SurvivalEVAL.Evaluations.custom_types import Numeric
from SurvivalEVAL.Evaluations.util import (
    check_and_convert,
    check_and_convert_event_data,
    validate_time_point,
)
from SurvivalEVAL.NonparametricEstimator.CompetingRisks import (
    AalenJohansenCompetingRisks,
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


def one_calibration_cr(
    preds: np.ndarray,
    event_time: np.ndarray,
    event_indicator: np.ndarray,
    target_time: Numeric,
    event_of_interest: int = 1,
    num_bins: int = 10,
    binning_strategy: str = "C",
) -> tuple[list[float], list[float]]:
    """Calculate a cause-specific calibration curve at one time.

    Compare each group's mean predicted CIF with its Aalen–Johansen estimate
    for the selected cause. All observed causes contribute to the risk sets;
    competing events are not treated as censorings. Assumes independent
    censoring within each group. This curve does not provide a calibration test.

    Args:
        preds: One finite CIF probability in [0, 1] per subject at target_time.
        event_time: Finite, non-negative observed event or censoring times.
        event_indicator: Matching non-negative integer labels, with 0 for censoring.
        target_time: Finite, non-negative evaluation time. Events at this time
            contribute to observed incidence (right continuity).
        event_of_interest: Positive integer cause label, including an absent cause.
        num_bins: Positive integer number of groups. Empty groups are omitted.
        binning_strategy: "C" (default) splits descending risks into approximately
            equal-sized groups; "H" uses equal-width groups on [0, 1].
            Case-insensitive, with the same grouping as survival one_calibration.

    Returns:
        Observed and mean predicted probabilities for each nonempty group.
        C groups follow descending risk; H groups follow ascending risk.
        All-censored groups and groups without the selected cause have observed
        probability zero. AJ estimates stay constant after the final event.

    Raises:
        ValueError: If predictions, outcomes, time, cause, or bin count are invalid.
        TypeError: If the binning strategy is unsupported.
    """
    event_time, event_indicator = check_and_convert_event_data(
        event_time, event_indicator, competing_risks=True
    )
    preds = check_and_convert(preds)
    if preds.shape != event_time.shape:
        raise ValueError("preds must have one probability per observed subject.")
    if not np.all(np.isfinite(preds)) or np.any((preds < 0) | (preds > 1)):
        raise ValueError("preds must contain finite probabilities in [0, 1].")
    target_time = validate_time_point(target_time)
    if not np.isfinite(target_time):
        raise ValueError("target_time must be finite.")
    if (
        isinstance(event_of_interest, (bool, np.bool_))
        or not isinstance(event_of_interest, (int, np.integer))
        or event_of_interest < 1
    ):
        raise ValueError("event_of_interest must be a positive integer cause label.")
    if (
        isinstance(num_bins, (bool, np.bool_))
        or not isinstance(num_bins, (int, np.integer))
        or num_bins < 1
    ):
        raise ValueError("num_bins must be a positive integer.")

    # Pool competing causes: only the selected cause's CIF is needed.
    # Our way is an equivalent computational simplification of the original labels,
    # it avoids calculating separate CIFs that this function never uses. 
    events = np.where(event_indicator > 0, 2, 0)
    events[event_indicator == event_of_interest] = 1
    observed, expected = [], []
    for indices in _probability_bins(preds, num_bins, binning_strategy.lower()):
        if indices.size == 0:
            continue
        aj = AalenJohansenCompetingRisks(n_causes=2).fit(
            event_time[indices], events[indices]
        )
        observed.append(float(aj.predict_cif(target_time)[0]))
        expected.append(float(np.mean(preds[indices], dtype=float)))
    return observed, expected


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
