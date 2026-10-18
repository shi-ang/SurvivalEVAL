"""Grouped fixed-time calibration for survival and competing-risk outcomes."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from SurvivalEVAL.Evaluations.custom_types import Numeric
from SurvivalEVAL.Evaluations.util import (
    check_and_convert_event_data,
    check_and_convert_probabilities,
    validate_time_point,
)
from SurvivalEVAL.NonparametricEstimator.CompetingRisks import (
    AalenJohansenCompetingRisks,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    KaplanMeier,
    TurnbullEstimatorLifelines,
)
from SurvivalEVAL.NonparametricEstimator.util import _predict_step


def _probability_bins(
    preds: np.ndarray,
    num_bins: int,
    strategy: str,
) -> list[np.ndarray]:
    """
    Group prediction indices into equal-sized or equal-width bins.

    Parameters
    ----------
    preds : ndarray, shape (n_samples,)
        Event probabilities in [0, 1].
    num_bins : int
        Positive number of bins to construct.
    strategy : {"c", "h"}
        "c" sorts probabilities in descending order and splits the indices
        into approximately equal-sized groups. "h" uses equal-width bins
        on [0, 1], including probability 1 in the final bin.

    Returns
    -------
    list[ndarray]
        One array of sample indices per bin, including empty bins.

    Raises
    ------
    TypeError
        If the strategy is neither "c" nor "h".
    """
    if (
        isinstance(num_bins, (bool, np.bool_))
        or not isinstance(num_bins, (int, np.integer))
        or num_bins < 1
    ):
        raise ValueError("num_bins must be a positive integer.")
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


def _calibration_result(
    observed: list,
    expected: list,
    bin_sizes: list,
    *,
    uncensored: bool = False,
    variances: list | None = None,
    return_details: bool = False,
) -> tuple[float, float, list, list] | tuple[float, dict]:
    """
    Compute the one-calibration statistic and its chi-square p-value.

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
    variances : list, optional
        Estimated variances of group incidence probabilities. When supplied,
        use G degrees of freedom and return an undefined p-value if any
        variance is zero. Otherwise use expected binomial variances.
    return_details : bool, default=False
        Return (p_value, details) instead of the four-element result.

    Returns
    -------
    tuple[float, float, list, list] or tuple[float, dict]
        P-value, statistic, and the original observed and expected inputs. When return_details
        is True, return (p_value, details), including group sizes, variances, and degrees of freedom.
        A predicted probability of 0 or 1 contributes zero when it matches
        the observed probability, and infinity otherwise. The p-value is
        NaN when the degrees of freedom are nonpositive.
    """
    obs, exp = np.asarray(observed), np.asarray(expected)
    numerator = (obs - exp) ** 2
    adjusted = variances is not None
    variance = (
        np.asarray(variances) if adjusted else exp * (1 - exp) / np.asarray(bin_sizes)
    )
    contributions = np.divide(
        numerator,
        variance,
        out=np.where(numerator == 0, 0.0, np.inf),
        where=variance != 0,
    )
    statistic = float(contributions.sum())
    num_bins = len(observed)
    dof = num_bins if adjusted else num_bins - (2 if uncensored or num_bins > 15 else 1)
    defined = dof > 0 and (not adjusted or np.all(variance > 0))
    p_value = float(chi2.sf(statistic, dof)) if defined else np.nan
    if return_details:
        return p_value, {
            "p_value": p_value,
            "statistics": statistic,
            "observed_probabilities": observed,
            "expected_probabilities": expected,
            "bin_sizes": bin_sizes,
            "variances": variance.tolist(),
            "degrees_of_freedom": dof,
            "adjust_variance": adjusted,
        }
    return p_value, statistic, observed, expected


def one_calibration(
    preds: np.ndarray,
    event_time: np.ndarray,
    event_indicator: np.ndarray,
    target_time: Numeric,
    num_bins: int = 10,
    binning_strategy: str = "C",
    method: str = "DN",
    adjust_variance: bool = True,
    return_details: bool = False,
) -> tuple[float, float, list, list] | tuple[float, dict]:
    """
    Test fixed-time calibration in groups of predicted event probabilities.

    Parameters
    ----------
    preds : ndarray
        One finite event probability in [0, 1] per subject at target_time.
    event_time : ndarray
        Matching finite, non-negative observed event/censoring times.
    event_indicator : ndarray
        Matching binary indicators, with 0 for censoring.
    target_time : Numeric
        Finite, non-negative horizon. Events at this time are included.
    num_bins : int
        Positive integer number of groups. Empty groups are omitted.
    binning_strategy : str
        "C" for approximately equal-sized groups ordered by
        descending risk, or "H" for equal-width probability groups.
    method : str
        "DN" (default) uses within-group Kaplan–Meier failure estimates [1]_.
        "Uncensored" excludes subjects censored before target_time and uses
        the remaining observed binary outcomes. Names are case-insensitive.
    adjust_variance : bool
        Use Greenwood variance for DN, or observed binomial
        variance for Uncensored. Defaults to True. False uses the original
        expected binomial variance, mean(preds) * (1 - mean(preds)) / n.
    return_details : bool
        Return (p_value, details), including statistics
        observed_probabilities, expected_probabilities, bin_sizes,
        variances, degrees_of_freedom, and adjust_variance.

    Returns
    -------
    tuple[float, float, list, list] | tuple[float, dict]
        (p_value, statistic, observed, expected), or (p_value, details).
        The adjusted statistic is the sum of squared group discrepancies divided
        by estimated variances, using G degrees of freedom for G nonempty
        groups with predictions fixed independently of the evaluation outcomes.
        The p-value is an asymptotic approximation,
        not a correction for evaluating a fitted model on its development data.
        It is NaN if no groups remain or any adjusted variance is zero.
        Without adjustment, the existing HL/DN degrees-of-freedom convention applies.
        KM estimates are right-continuous and stay constant after the final event.

    Notes
    -----
    Censoring must be independent within risk groups. Sparse groups make
    chi-square approximations unreliable; inspect the curve and group sizes.
    The original DN test is described in [1]_; its Greenwood variance
    modification is discussed in [2]_.

    References
    ----------
    .. [1] D'Agostino, R. B.; Nam, B.-H. (2003). "Evaluation of the performance
       of survival analysis models: discrimination and calibration measures".
       Handbook of Statistics, 23, 1-25.
       https://doi.org/10.1016/S0169-7161(03)23001-7
    .. [2] Demler, Paynter and Cook (2015). "Tests of Calibration and Goodness of Fit
       in the Survival Setting". Statistics in Medicine, 34(10), 1659-1680.
       https://doi.org/10.1002/sim.6428
    """
    method = method.lower()
    if method not in {"uncensored", "dn"}:
        raise TypeError("Please enter one of 'Uncensored','DN' for method.")
    event_time, event_indicator = check_and_convert_event_data(
        event_time, event_indicator
    )
    return _one_calibration(
        preds,
        event_time,
        event_indicator,
        target_time,
        num_bins,
        binning_strategy,
        adjust_variance,
        return_details,
        method=method,
    )


def one_calibration_cr(
    preds: np.ndarray,
    event_time: np.ndarray,
    event_indicator: np.ndarray,
    target_time: Numeric,
    event_of_interest: int = 1,
    num_bins: int = 10,
    binning_strategy: str = "C",
    adjust_variance: bool = True,
    return_details: bool = False,
) -> tuple[float, float, list, list] | tuple[float, dict]:
    """
    Test cause-specific calibration at one time using grouped AJ incidence.

    Parameters
    ----------
    preds : ndarray
        One finite CIF probability in [0, 1] per subject at target_time.
    event_time : ndarray
        Matching finite, non-negative observed event/censoring times.
    event_indicator : ndarray
        Matching non-negative integer cause labels; 0 is censoring.
    target_time : Numeric
        Finite, non-negative horizon. Events at this time are included.
    event_of_interest : int
        Positive integer cause label, including an absent cause.
    num_bins : int
        Positive integer number of groups. Empty groups are omitted.
    binning_strategy : str
        "C" for approximately equal-sized groups ordered by
        descending risk, or "H" for equal-width probability groups.
    adjust_variance : bool
        Use the Greenwood-type variance of the Aalen–Johansen
        incidence estimate, accounting for simultaneous tied events.
        The method is described in [1]_.
        Defaults to True. False uses expected binomial variance instead.
    return_details : bool
        Return (p_value, details), with the same fields as
        single-event one_calibration, including group variances and sizes.

    Returns
    -------
    tuple[float, float, list, list] | tuple[float, dict]
        (p_value, statistic, observed, expected), or (p_value, details).
        Observed probabilities are within-group AJ estimates for the selected
        cause. Other causes remain observed events, pooled for this calculation.
        The adjusted test is a grouped Wald extension using AJ variance and G
        degrees of freedom, not the original single-event GND test. P-values are
        asymptotic approximations assuming independent censoring within groups
        and an independently trained prediction model. Sparse groups can be
        unreliable. The adjusted p-value is NaN if any group variance is zero
        or no groups remain. Estimates stay constant after the final event;
        unobserved causes and all-censored groups have incidence zero.

    References
    ----------
    .. [1] Allignol, Schumacher and Beyersmann (2010). "A Note on Variance Estimation
       of the Aalen-Johansen Estimator ...". Biometrical Journal, 52(1), 126-137.
       https://doi.org/10.1002/bimj.200900039
    """
    event_time, event_indicator = check_and_convert_event_data(
        event_time, event_indicator, competing_risks=True
    )
    if (
        isinstance(event_of_interest, (bool, np.bool_))
        or not isinstance(event_of_interest, (int, np.integer))
        or event_of_interest < 1
    ):
        raise ValueError("event_of_interest must be a positive integer cause label.")

    # Pool competing causes: the selected CIF and its variance are unchanged.
    events = np.where(event_indicator > 0, 2, 0)
    events[event_indicator == event_of_interest] = 1
    return _one_calibration(
        preds,
        event_time,
        events,
        target_time,
        num_bins,
        binning_strategy,
        adjust_variance,
        return_details,
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
    preds : ndarray
        The predicted probabilities of experiencing the event at the time of interest.
    left_limits : ndarray
        The left limits of the interval-censored event times.
    right_limits : ndarray
        The right limits of the interval-censored event times.
    target_time : Numeric
        The time of interest.
    num_bins : int
        The number of bins to divide the predictions into.
    binning_strategy : str
        The strategy to bin the predictions. The options are: "C" (default), and "H".
        C-statistics means the predictions are divided into equal-sized bins based on the predicted probabilities.
        H-statistics means the predictions are divided into equal-increment bins from 0 to 1.
    method : str
        The method to handle censored patients. The options are: "Turnbull" (default), and "MidPoint".
        "MidPoint" method simply treats the midpoint of the interval as the event time, and
        uses the DN's method (Kaplan-Meier estimate of the survival function).
        "Turnbull" method uses the Turnbull estimator for the survival function
        to compute the average observed probabilities in each bin.

    Returns
    -------
    p_value : float
        The one-calibration p-value, or NaN if too few nonempty bins remain.
    statistics : float
        The Hosmer-Lemeshow statistic.
        Boundary probabilities contribute zero when observed and expected
        probabilities agree, and infinity otherwise.
    observed_probabilities : list
        The observed probabilities in each bin.
    expected_probabilities : list
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


def _one_calibration(
    preds: np.ndarray,
    event_time: np.ndarray,
    events: np.ndarray,
    target_time: float,
    num_bins: int,
    binning_strategy: str,
    adjust_variance: bool,
    return_details: bool,
    *,
    method: str = "aj",
) -> tuple[float, float, list, list] | tuple[float, dict]:
    """
    Evaluate pooled labels (0 censoring, 1 selected, 2 competing) in risk bins.

    Parameters
    ----------
    preds : ndarray
        Predicted cumulative event probabilities, one per sample.
    event_time : ndarray
        Observed event or censoring times, one per sample.
    events : ndarray
        Pooled event labels: 0 for censoring, 1 for the selected cause, and 2 for competing events.
    target_time : float
        Non-negative evaluation horizon; None selects the evaluator's default target time.
    num_bins : int
        Positive number of probability bins.
    binning_strategy : str
        "C" forms approximately equal-sized risk groups; "H" forms equal-width probability groups.
    adjust_variance : bool
        Whether to use estimated group variances instead of expected binomial variances.
    return_details : bool
        Whether to return diagnostic details along with the score.
    method : str
        Group estimator: "aj" for Aalen-Johansen, "dn" for Kaplan-Meier, or "uncensored" for empirical risks.

    Returns
    -------
    tuple[float, float, list, list] | tuple[float, dict]
        The p-value, statistic, observed probabilities, and expected probabilities, or (p_value, details).
    """
    preds = check_and_convert_probabilities(preds, "preds")
    if preds.shape != event_time.shape:
        raise ValueError("preds must have one probability per observed subject.")
    target_time = validate_time_point(target_time)
    if not np.isfinite(target_time):
        raise ValueError("target_time must be finite.")
    observed, expected, bin_sizes = [], [], []
    variances = [] if adjust_variance else None
    uncensored = method == "uncensored"
    for indices in _probability_bins(preds, num_bins, binning_strategy.lower()):
        if uncensored:
            indices = indices[
                ~((event_time[indices] < target_time) & (events[indices] == 0))
            ]
        if indices.size == 0:
            continue
        times, indicators = event_time[indices], events[indices]
        if uncensored:
            probability = float(np.mean((times <= target_time) & (indicators == 1)))
            if adjust_variance:
                variances.append(probability * (1 - probability) / indices.size)
        elif method == "dn":
            km = KaplanMeier(times, indicators)
            probability = float(
                _predict_step(target_time, km.survival_times, km.cumulative_dens, 0.0)
            )
            if adjust_variance:
                variances.append(km.predict_variance(target_time))
        else:
            aj = AalenJohansenCompetingRisks(n_causes=2).fit(times, indicators)
            probability = float(aj.predict_cif(target_time)[0])
            if adjust_variance:
                variances.append(float(aj.predict_cif_variance(target_time)[0]))
        observed.append(probability)
        expected.append(float(np.mean(preds[indices], dtype=float)))
        bin_sizes.append(indices.size)
    return _calibration_result(
        observed,
        expected,
        bin_sizes,
        uncensored=uncensored,
        variances=variances,
        return_details=return_details,
    )
