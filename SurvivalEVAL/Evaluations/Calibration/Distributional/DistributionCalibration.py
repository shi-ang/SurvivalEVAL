"""D-calibration and KS variants for censored survival and competing risks."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import trapezoid
from scipy.stats import chisquare

from SurvivalEVAL.Evaluations.util import check_and_convert
from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    KaplanMeier,
    TurnbullEstimatorLifelines,
)

from .util import discrepancy_to_uniform, ks_pvalue


def d_calibration(
    pred_probs: np.ndarray, event_indicators: np.ndarray, num_bins: int = 10
) -> tuple[float, float, np.ndarray]:
    """
    Calculate the D-Calibration score.
    Parameters
    ----------
    pred_probs: np.ndarray
        The predicted survival probabilities at individual's event/censor time.
    event_indicators: np.ndarray
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    num_bins: int
        The number of bins to use for the D-Calibration score.

    Returns
    -------
    statistic: float
        The test statistic of the D-Calibration test.
    pvalue: float
        The p-value of the D-Calibration test.
    combine_hist: np.ndarray
        The binning histogram of the D-Calibration test.
    """
    quantile = np.linspace(1, 0, num_bins + 1)
    event_indicators = event_indicators.astype(bool, copy=False)
    censor_indicators = ~event_indicators

    event_probs = pred_probs[event_indicators]
    event_position = np.digitize(event_probs, quantile)
    event_position[event_position == 0] = 1  # class probability==1 to the first bin

    event_hist = np.bincount(event_position - 1, minlength=num_bins)

    censored_probs = pred_probs[censor_indicators]

    # Match create_censor_hist, which contributes zero outside [0, 1].
    censored_probs = censored_probs[
        (censored_probs >= 0) & (censored_probs <= 1)
    ].astype(float, copy=False)
    positions = np.maximum(np.digitize(censored_probs, quantile) - 1, 0)
    partial_weights = np.divide(
        censored_probs - quantile[positions + 1],
        censored_probs,
        out=np.ones(censored_probs.shape),
        where=censored_probs != 0,
    )
    partial_weights[censored_probs == 1] = 1 / num_bins
    censor_hist = np.bincount(positions, weights=partial_weights, minlength=num_bins)

    # A censored sample adds the same weight to every subsequent bin. Group
    # those weights by their starting bin, then accumulate once across bins.
    # Samples already in the last bin need no tail weight (including p == 0).
    has_tail = positions < num_bins - 1
    tail_weights = np.bincount(
        positions[has_tail],
        weights=1 / (num_bins * censored_probs[has_tail]),
        minlength=num_bins,
    )
    censor_hist[1:] += np.cumsum(tail_weights[:-1])

    combine_hist = event_hist + censor_hist
    statistic, pvalue = chisquare(combine_hist)
    return statistic, pvalue, combine_hist


def create_censor_hist(prob: float, num_bins: int) -> np.ndarray:
    """Create the binning histogram for a right censored instance.

    The bins are defined as follows:
    [b_0, b_1), [b_1, b_2), ..., [b_{num_bins-1}, b_num_bins] (note that the last bin is closed on both side).

    For censoring instance,
    b2 will be the infimum probability of the bin that contains S(c),
    for the bin of [b2, b3) which contains S(c), probability = (S(c) - b2) / S(c)
    for the rest of the bins, [b0, b1), [b1, b2), etc., probability = 1 / (B * S(c)), where B is the number of bins.
    :param prob: float

        The predicted probability at the censored time of a censoring instance.
    :param num_bins: int
        The number of bins to use for the D-Calibration score.
    :return:
    final_binning: np.ndarray
        The "split" histogram of this censored subject.
    """
    quantile = np.linspace(1, 0, num_bins + 1)
    censor_binning = np.zeros(num_bins)
    for i in range(num_bins):
        if prob == 1:
            censor_binning += 1 / num_bins
            break
        elif quantile[i] > prob >= quantile[i + 1]:
            first_bin = (prob - quantile[i + 1]) / prob if prob != 0 else 1
            rest_bins = 1 / (num_bins * prob) if prob != 0 else 0
            censor_binning[i] += first_bin
            censor_binning[i + 1 :] += rest_bins
            break
    # assert len(censor_binning) == num_bins, f"censor binning should have size of {num_bins}"
    return censor_binning


def d_cal_ic(
    pred_probs_left: np.ndarray, pred_probs_right: np.ndarray, num_bins: int = 10
) -> tuple[float, float, np.ndarray]:
    """
    Calculate the D-Calibration score for interval censored data.
    Parameters
    ----------
    pred_probs_left: np.ndarray
        The predicted survival probabilities at individual's left censor time.
    pred_probs_right: np.ndarray
        The predicted survival probabilities at individual's right censor time.
        For right-censored instances, the right event/censor time is infinity, so the predicted probability is 0.
    num_bins: int
        The number of bins to use for the D-Calibration score.
    Returns
    -------
    statistic: float
        The test statistic of the D-Calibration test.
    pvalue: float
        The p-value of the D-Calibration test.
    binning: np.ndarray
        The binning histogram of the D-Calibration test.
    """
    assert len(pred_probs_left) == len(
        pred_probs_right
    ), "The length of pred_probs_left and pred_probs_right should have same length."

    assert np.all(
        pred_probs_left >= pred_probs_right
    ), "The left survival probabilities should be greater than or equal to the right survival probabilities."

    assert (
        np.all(pred_probs_left >= 0)
        and np.all(pred_probs_right >= 0)
        and np.all(pred_probs_left <= 1)
        and np.all(pred_probs_right <= 1)
    ), "The predicted probabilities should be in the range [0, 1]."

    n = len(pred_probs_left)

    binning = np.zeros([num_bins])
    for i in range(n):
        partial_binning = create_interval_c_hist(
            pred_probs_left[i], pred_probs_right[i], num_bins
        )
        binning += partial_binning

    statistic, pvalue = chisquare(binning)
    return statistic, pvalue, binning


def create_interval_c_hist(
    prob_left: float, prob_right: float, num_bins: int
) -> np.ndarray:
    """Distribute one observation's mass over descending probability bins.

    A nonzero interval contributes its fractional overlap with each bin.
    Equal endpoints contribute all mass to their containing bin, including
    the boundary probabilities 0 and 1. Reversed endpoints are reordered.
    """
    prob_left, prob_right = max(prob_left, prob_right), min(prob_left, prob_right)
    edges = np.linspace(1.0, 0.0, num_bins + 1)
    if prob_left == prob_right:
        hist = np.zeros(num_bins)
        index = np.clip(np.digitize(prob_left, edges) - 1, 0, num_bins - 1)
        hist[index] = 1.0
        return hist

    overlap = np.maximum(
        0.0, np.minimum(edges[:-1], prob_left) - np.maximum(edges[1:], prob_right)
    )
    return overlap / (prob_left - prob_right)


def _check_cr_d_inputs(
    pred_cifs: np.ndarray,
    event_indicators: np.ndarray,
    pred_event_probs: np.ndarray,
    pred_survs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert and validate inputs shared by the CR-D-calibration metrics.

    Parameters
    ----------
    pred_cifs : np.ndarray, shape (n_samples,) or (n_samples, n_causes)
        One cause's CIFs or all causes' CIFs at each subject's observed time.
        For multiple causes, column k - 1 represents cause k.
    event_indicators : np.ndarray, shape (n_samples,)
        Non-negative integer-valued labels: 0 for censoring and k for cause k.
    pred_event_probs : np.ndarray, same shape as pred_cifs
        Eventual cause probabilities, no smaller than the observed-time CIFs.
    pred_survs : np.ndarray, shape (n_samples,)
        Overall survival at each observed time, positive for censored subjects.

    Returns
    -------
    cifs, events, event_probs, survs : tuple of np.ndarray
        Validated numeric arrays in this order. Compatible arrays may share
        input storage; this function does not modify them.

    Raises
    ------
    ValueError
        If shapes, labels, probability bounds, or probability conservation are
        inconsistent. For joint predictions, CIFs plus survival and eventual
        probabilities must each sum to one per subject (tolerance 1e-6).

    See Also
    --------
    check_and_convert : Numeric input conversion and basic array validation.
    """
    cifs, event_probs = check_and_convert(pred_cifs, pred_event_probs)
    survs, events = check_and_convert(pred_survs, event_indicators)
    if cifs.ndim not in (1, 2) or survs.ndim != 1 or cifs.shape[0] != survs.size:
        raise ValueError(
            "CIFs must have shape (n_samples,) or (n_samples, n_causes); "
            "survival probabilities and event_indicators must have shape (n_samples,)."
        )
    if (
        not np.all(np.isfinite(events))
        or np.any(events < 0)
        or np.any(events != np.floor(events))
    ):
        raise ValueError("event_indicators must be non-negative integer cause labels.")
    for name, values in (
        ("pred_cifs", cifs),
        ("pred_event_probs", event_probs),
        ("pred_survs", survs),
    ):
        if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} must contain finite probabilities in [0, 1].")
    if np.any(cifs > event_probs):
        raise ValueError(
            "Observed-time CIFs must not exceed terminal event probabilities."
        )

    # The unobserved remaining probability of any cause cannot exceed survival.
    survival = survs if cifs.ndim == 1 else survs[:, None]
    if np.any(cifs + survival > 1 + 1e-6) or np.any(
        event_probs - cifs > survival + 1e-6
    ):
        raise ValueError(
            "CIFs and terminal event probabilities are inconsistent with survival."
        )
    if np.any(survs[events == 0] == 0):
        raise ValueError("Censored observations must have positive predicted survival.")
    if cifs.ndim == 2:
        if np.any(events > cifs.shape[1]):
            raise ValueError(
                "Event labels must not exceed the number of predicted causes."
            )
        if not np.allclose(cifs.sum(axis=1) + survs, 1, rtol=0, atol=1e-6):
            raise ValueError("CIFs across causes plus survival must sum to one.")
        if not np.allclose(event_probs.sum(axis=1), 1, rtol=0, atol=1e-6):
            raise ValueError(
                "Terminal event probabilities across causes must sum to one."
            )
    return cifs, events, event_probs, survs


def _cr_d_curve(
    cifs: np.ndarray,
    events: np.ndarray,
    event_probs: np.ndarray,
    survs: np.ndarray,
    cause: int,
    rho: np.ndarray,
) -> np.ndarray:
    """Compute one cause's empirical CR-D-calibration curve on a uniform grid.

    Parameters
    ----------
    cifs : np.ndarray, shape (n_samples,)
        Selected cause's CIFs at each subject's observed time.
    events : np.ndarray, shape (n_samples,)
        Cause labels, with 0 denoting censoring.
    event_probs : np.ndarray, shape (n_samples,)
        Selected cause's eventual probabilities F_k(infinity).
    survs : np.ndarray, shape (n_samples,)
        Overall survival at each observed time.
    cause : int
        Positive cause label to evaluate.
    rho : np.ndarray, shape (num_bins + 1,)
        Uniformly spaced thresholds from 0 to 1, including both endpoints.
        Must contain at least two points.

    Returns
    -------
    curve : np.ndarray, same shape as rho
        Cumulative observed and imputed event mass divided by the sum of
        eventual cause probabilities. The endpoint is not forced to one.

    Raises
    ------
    ValueError
        If total eventual cause mass is zero or an observed event of the
        selected cause has zero eventual probability.

    Notes
    -----
    Inputs must already satisfy ``_check_cr_d_inputs``. Observed events of this
    cause contribute when F_k(T) / F_k(infinity) <= rho. Censored subjects
    contribute ``max(rho * F_k(infinity) - F_k(T), 0) / S(T)``; other observed
    causes contribute zero. Grouping partial and subsequent bucket weights
    avoids allocating a sample-by-bin matrix.
    """
    observed = events == cause
    if np.any(event_probs[observed] == 0):
        raise ValueError(
            "Observed events must have positive terminal probability for their cause."
        )
    total_mass = event_probs.sum(dtype=float)
    if total_mass == 0:
        raise ValueError(
            "The selected cause must have positive total terminal probability."
        )

    ratios = np.divide(
        cifs, event_probs, out=np.zeros(cifs.shape), where=event_probs > 0
    )
    positions = np.searchsorted(rho, ratios, side="left")
    mass = np.bincount(positions[observed], minlength=rho.size).astype(float)

    censored = events == 0
    positions = positions[censored]
    terminal, current, survival = event_probs[censored], cifs[censored], survs[censored]
    partial = np.maximum(rho[positions] * terminal - current, 0) / survival
    mass += np.bincount(positions, weights=partial, minlength=rho.size)

    # Only samples with later bins need tail weights. This also avoids dividing
    # by tiny survival when the contribution lies entirely in the final bin.
    has_tail = positions < rho.size - 1
    tail = np.bincount(
        positions[has_tail],
        weights=(terminal[has_tail] / (rho.size - 1)) / survival[has_tail],
        minlength=rho.size,
    )
    mass[1:] += np.cumsum(tail[:-1])
    return np.cumsum(mass) / total_mass


def cr_d_calibration_per_event(
    pred_cifs: np.ndarray,
    event_indicators: np.ndarray,
    pred_event_probs: np.ndarray,
    pred_survs: np.ndarray,
    event_of_interest: int,
    num_bins: int = 100,
    alpha: float = 2,
    method: str = "integral",
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """Measure CR-D-calibration for one competing cause (lower is better).

    Parameters
    ----------
    pred_cifs : array-like, shape (n_samples,)
        CIF for the selected cause at each subject's observed event/censor time.
    event_indicators : array-like, shape (n_samples,)
        Cause labels: 0 for censoring, positive integers for observed events.
        Other competing events are observed outcomes, not censorings.
    pred_event_probs : array-like, shape (n_samples,)
        Eventual probability F_k(infinity) of the selected cause. A terminal
        CIF value is suitable only when the prediction horizon is long enough
        to approximate this probability. Observed events require positive mass.
    pred_survs : array-like, shape (n_samples,)
        Overall survival at each observed time, including all competing causes.
        Must be positive for censored subjects.
    event_of_interest : int
        Positive cause label to evaluate.
    num_bins : int, default=100
        Number of equal-width intervals in the risk grid over [0, 1].
    alpha : float, default=2
        Finite exponent at least 1 for the integral norm. `alpha=1`: area under
        the absolute-error curve; errors contribute linearly.
        `alpha=2` (default): root integrated squared error;
        Ignored for KS.
    method : {"integral", "ks"}, default="integral"
        Integral returns the L_alpha norm of b_k(rho) - rho over [0, 1],
        approximated with the trapezoidal rule. KS returns the maximum absolute
        deviation on the same grid.
    return_details : bool, default=False
        Also return a dictionary with ``rho`` and ``calibration_curve``. For KS,
        include ``approx_p_value`` and ``event_count``. This p-value uses the
        asymptotic Kolmogorov distribution and the observed cause count. With
        fractional censoring weights, estimated terminal masses, and a finite
        risk grid, it is a heuristic, not an exact or proven level-correct test.
        It is NaN when the selected cause has no observed events.

    Returns
    -------
    score : float
        Zero denotes agreement with the identity curve on the risk grid.
    details : dict, optional
        Calibration curve and, for KS, approximate test information.

    Notes
    -----
    The curve is normalized by the sum of predicted terminal event probabilities,
    not its observed endpoint. Its endpoint can differ from one, revealing
    marginal miscalibration. Censored subjects contribute
    ``max(rho * F_k(infinity) - F_k(T), 0) / S(T)``; other observed causes
    contribute zero. Conditional non-informative censoring is assumed.

    Uses the paper's L_alpha norm. The referenced hazardous implementation instead
    divides the integrated powered deviation by alpha and omits [0, 1/num_bins]
    from its numerical integral; its scalar scores therefore differ.

    References
    ----------
    Alberge, Haugomat, Varoquaux and Abecassis (2026), "On the Calibration of
    Survival Models with Competing Risks", https://proceedings.mlr.press/v300/alberge26a.html
    https://github.com/soda-inria/hazardous/blob/dcal/hazardous/metrics/_d_calibration.py
    """
    cifs, events, event_probs, survs = _check_cr_d_inputs(
        pred_cifs, event_indicators, pred_event_probs, pred_survs
    )
    if cifs.ndim != 1:
        raise ValueError("Per-event CIFs and terminal probabilities must be 1D arrays.")
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
    method = method.lower()
    if method not in ("integral", "ks"):
        raise ValueError("method must be 'integral' or 'ks'.")
    if method == "integral" and (
        not np.isscalar(alpha) or not np.isfinite(alpha) or alpha < 1
    ):
        raise ValueError("alpha must be finite and at least 1.")

    rho = np.linspace(0, 1, num_bins + 1)
    curve = _cr_d_curve(cifs, events, event_probs, survs, event_of_interest, rho)
    deviations = np.abs(curve - rho)
    if method == "integral":
        # Normalize before exponentiation: a tiny terminal cause probability
        # can produce a large, still representable calibration deviation.
        scale = deviations.max()
        score = (
            float(scale * trapezoid((deviations / scale) ** alpha, rho) ** (1 / alpha))
            if scale
            else 0.0
        )
    else:
        score = float(deviations.max())
    if not return_details:
        return score
    details = {"rho": rho, "calibration_curve": curve}
    if method == "ks":
        count = int(np.count_nonzero(events == event_of_interest))
        details.update(
            event_count=count,
            approx_p_value=ks_pvalue(score, count) if count else np.nan,
        )
    return score, details


def cr_d_calibration(
    pred_cifs: np.ndarray,
    event_indicators: np.ndarray,
    pred_event_probs: np.ndarray,
    pred_survs: np.ndarray,
    num_bins: int = 100,
    alpha: float = 2,
    method: str = "integral",
    reduction: str = "sum",
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """Aggregate CR-D-calibration over all predicted competing causes.

    Parameters
    ----------
    pred_cifs : array-like, shape (n_samples, n_causes)
        CIFs at each subject's observed time; column k - 1 represents cause k.
    event_indicators : array-like, shape (n_samples,)
        Labels 0 through n_causes, with 0 denoting censoring.
    pred_event_probs : array-like, shape (n_samples, n_causes)
        Eventual cause probabilities, summing to one per subject. Every cause
        must have positive total mass. All predicted causes are evaluated,
        including those absent from the observed event labels.
    pred_survs : array-like, shape (n_samples,)
        Overall survival at each observed time; this and CIFs must sum to one.
    num_bins, alpha, method
        See ``cr_d_calibration_per_event``. The default sums the paper's L_alpha
        scores; ``method="ks"`` instead aggregates maximum grid deviations.
    alpha : float, default=2
        Finite exponent at least 1 for the integral norm. `alpha=1`: area under
        the absolute-error curve; errors contribute linearly.
        `alpha=2` (default): root integrated squared error;
        Ignored for KS.
    reduction : {"sum", "mean", "max"}, default="sum"
        Reduction of per-cause scores. Sum follows the paper's definition.
    return_details : bool, default=False
        Also return ``per_event_scores`` and ``per_event_details`` dictionaries,
        keyed by cause label. For KS, include ``bonferroni_approx_p_values``:
        per-cause approximate p-values multiplied by the number of predicted
        causes and capped at one. This adjustment does not make the underlying
        censor-weighted KS approximation an exact test.

    Returns
    -------
    score : float
        Aggregated calibration score. Lower is better.
    details : dict, optional
        Per-event scores, curves, and optional approximate KS test information.

    See Also
    --------
    cr_d_calibration_per_event : Definition, assumptions and references.
    """
    cifs, events, event_probs, survs = _check_cr_d_inputs(
        pred_cifs, event_indicators, pred_event_probs, pred_survs
    )
    if cifs.ndim != 2:
        raise ValueError(
            "CIFs and terminal probabilities must have shape (n_samples, n_causes)."
        )
    reduction = reduction.lower()
    if reduction not in ("sum", "mean", "max"):
        raise ValueError("reduction must be 'sum', 'mean' or 'max'.")

    scores, per_event = {}, {}
    for k in range(cifs.shape[1]):
        result = cr_d_calibration_per_event(
            cifs[:, k],
            events,
            event_probs[:, k],
            survs,
            k + 1,
            num_bins=num_bins,
            alpha=alpha,
            method=method,
            return_details=return_details,
        )
        if return_details:
            scores[k + 1], per_event[k + 1] = result
        else:
            scores[k + 1] = result
    score = float(
        {"sum": np.sum, "mean": np.mean, "max": np.max}[reduction](
            list(scores.values())
        )
    )
    if not return_details:
        return score
    details = {"per_event_scores": scores, "per_event_details": per_event}
    if method.lower() == "ks":
        details["bonferroni_approx_p_values"] = {
            cause: float(np.minimum(1, cifs.shape[1] * item["approx_p_value"]))
            for cause, item in per_event.items()
        }
    return score, details


def ksd_calibration(
    pred_probs: np.ndarray,
    event_indicators: np.ndarray,
    return_details: bool = False,
) -> tuple[float, float] | tuple[float, dict]:
    """
    Calculate the K-S D-Calibration score.

    Parameters
    ----------
    pred_probs: np.ndarray
        The predicted survival probabilities at individual's event/censor time.
    event_indicators: np.ndarray
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    return_details: bool
        Whether to return the detailed information including the empirical distribution and the figure.

    Returns
    -------
    pvalue: float
        The p-value of the K-S D-Calibration test.
    statistic: float
        The test statistic of the K-S D-Calibration test.
    details: dict
        The detailed information including the empirical distribution and the figure.
        - statistics: float
        - p_value: float
        - empirical_distribution: tuple (x_support, cdf_values)
        - figure: tuple (fig, ax)
    """
    assert len(pred_probs) == len(
        event_indicators
    ), "The length of pred_probs and event_indicators should have same length."

    n = len(pred_probs)
    km = KaplanMeier(pred_probs, event_indicators)
    x_support = km.survival_times
    cdf_values = km.cumulative_dens

    D_n = discrepancy_to_uniform(x_support, cdf_values)

    p_value = ks_pvalue(D_n, n)

    if return_details:
        fig, ax = plt.subplots()
        # plot the empirical CDF
        ax.step(
            x_support,
            cdf_values,
            where="post",
            label="Prediction",
            color="blue",
            linewidth=2,
        )
        # plot the ideal CDF
        ax.plot(
            [0, 1], [0, 1], label="Ideal", linestyle="--", color="grey", linewidth=2
        )
        ax.legend()
        ax.set_xlabel(r"Survival Probability $S(t_i \mid x)$")
        ax.set_ylabel("Cumulative Distribution Function")
        fig.tight_layout()

        details = {
            "statistics": D_n,
            "p_value": p_value,
            "empirical_distribution": (x_support, cdf_values),
            "figure": (fig, ax),
        }
        return p_value, details
    return p_value, D_n


def ksd_cal_ic(
    pred_probs_left: np.ndarray,
    pred_probs_right: np.ndarray,
    return_details: bool = False,
):
    """
    Calculate the K-S D-Calibration score for interval censored data.

    Parameters
    ----------
    pred_probs_left: np.ndarray
        The predicted survival probabilities at individual's left censor time.
    pred_probs_right: np.ndarray
        The predicted survival probabilities at individual's right censor time.
    return_details: bool
        Whether to return the detailed information including the empirical distribution and the figure.

    Returns
    -------
    pvalue: float
        The p-value of the K-S D-Calibration test.
    statistic: float
        The test statistic of the K-S D-Calibration test.
    details: dict
        The detailed information including the empirical distribution and the figure.
        - statistics: float
        - p_value: float
        - empirical_distribution: tuple (x_support, cdf_values)
        - figure: tuple (fig, ax)
    """
    assert len(pred_probs_left) == len(
        pred_probs_right
    ), "The length of pred_probs_left and pred_probs_right should have same length."

    assert np.all(
        pred_probs_left >= pred_probs_right
    ), "The left survival probabilities should be greater than or equal to the right survival probabilities."

    # Fit a Turnbull estimator on the predicted probabilities
    n = len(pred_probs_left)
    tb = TurnbullEstimatorLifelines(pred_probs_right, pred_probs_left)
    x_support = tb.survival_times
    cdf_values = tb.cumulative_dens

    D_n = discrepancy_to_uniform(x_support, cdf_values)

    p_value = ks_pvalue(D_n, n)

    if return_details:
        fig, ax = plt.subplots()
        # plot the empirical CDF
        ax.step(
            x_support,
            cdf_values,
            where="post",
            label="Prediction",
            color="blue",
            linewidth=2,
        )
        # plot the ideal CDF
        ax.plot(
            [0, 1], [0, 1], label="Ideal", linestyle="--", color="grey", linewidth=2
        )
        ax.legend()
        ax.set_xlabel(r"Survival Probability $S(t_i \mid x)$")
        ax.set_ylabel("Cumulative Distribution Function")
        fig.tight_layout()

        details = {
            "statistics": D_n,
            "p_value": p_value,
            "empirical_distribution": (x_support, cdf_values),
            "figure": (fig, ax),
        }
        return p_value, details

    return p_value, D_n
