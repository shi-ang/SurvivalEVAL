from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import trapezoid
from scipy.stats import chisquare, kstwobign

from SurvivalEVAL.Evaluations.util import check_and_convert, interpolated_curve
from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    KaplanMeier,
    NelsonAalen,
    TurnbullEstimatorLifelines,
)


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


_residual_names = {
    "coxsnell": "Cox-Snell Residuals",
    "modified coxsnell-v1": "Cox-Snell Residuals",
    "modified coxsnell-v2": "Cox-Snell Residuals",
    "martingale": "Martingale Residuals",
    "deviance": "Deviance Residuals",
}


def residuals(
    pred_probs: np.ndarray,
    event_indicators: np.ndarray,
    method: str = "CoxSnell",
    draw_figure: bool = False,
) -> np.ndarray:
    """
    Calculate the residuals based on the predicted probabilities and event times.
    For details of the residuals, please refer to the Chapter 4 of the book:
    "Modelling survival data in medical research" By David Collett

    Parameters
    ----------
    pred_probs: np.ndarray
        The predicted survival probabilities at individual's event/censor time.
    event_indicators: np.ndarray
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    method: str
        The method to calculate residuals. Options are "CoxSnell", "Modified CoxSnell-v1", "Modified CoxSnell-v2",
        "Martingale", "Deviance".
    draw_figure: bool
        Whether to visualize the Cox-Snell residuals and draw the cumulative hazard plot.
    Returns
    -------
    np.ndarray
        The calculated residuals.
    """
    cox_residuals = -np.log(pred_probs)
    event_indicators = event_indicators.astype(bool, copy=False)
    method = method.lower()

    if method == "coxsnell":
        residuals = cox_residuals
    elif method in ["modified coxsnell-v1", "modified coxsnell-v2"]:
        # Compare with standard CoxSnell residuals, this method adds an 'excess residual' for censored instances.
        # The excess residual should also follow a unit exponential distribution, based on the lack of memory property.
        # There are two choices of excess residuals:
        # (1) use the mean of the unit exponential distribution, which is 1,
        # or (2) use the median of the unit exponential distribution, which is ln(2).
        excess_residual = 1 if method == "modified coxsnell-v1" else np.log(2)
        residuals = cox_residuals + excess_residual * (~event_indicators)
    elif method == "martingale":
        residuals = event_indicators - cox_residuals
    elif method == "deviance":

        def safe_log(x):
            return np.log(x + 1e-8)

        martingale_res = event_indicators - cox_residuals
        # get the sign of the martingale residuals
        sign = np.sign(martingale_res)
        residuals = sign * np.sqrt(
            -2 * (martingale_res + event_indicators * safe_log(cox_residuals))
        )
    else:
        raise ValueError(f"Unknown method {method}")

    if draw_figure:
        cum_haz_empirical = NelsonAalen(cox_residuals, event_indicators)
        max_res = np.max(cox_residuals)
        _fig, ax = plt.subplots(nrows=1, ncols=2, tight_layout=True, dpi=400)
        ax[0].plot(
            cum_haz_empirical.survival_times,
            cum_haz_empirical.cumulative_hazard,
            label="Empirical",
        )
        ax[0].plot(
            [0, max_res], [0, max_res], label="Ideal", linestyle="--", color="red"
        )
        ax[0].legend()
        ax[0].set_xlabel("Cox-Snell Residuals")
        ax[0].set_ylabel("Cumlative hazard of residuals")

        # use solid scatter points for uncensored instances and hollow scatter points for censored instances
        idx = np.arange(len(residuals))
        ax[1].scatter(
            idx[event_indicators],
            residuals[event_indicators],
            alpha=0.5,
            color="red",
            label="Event",
        )
        ax[1].scatter(
            idx[~event_indicators],
            residuals[~event_indicators],
            facecolors="none",
            edgecolors="red",
            alpha=0.5,
            label="Censored",
        )
        ax[1].set_xlabel("Index")
        ax[1].set_ylabel(_residual_names[method])
        ax[1].legend()
        plt.show()
    return residuals


def km_calibration(
    average_survival_curve: np.ndarray,
    time_coordinates: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    interpolation_method: str = "Linear",
    draw_figure: bool = False,
) -> float | tuple[float, tuple[plt.Figure, plt.Axes]]:
    """
    Calculate the KM calibration score between the average prediction curve and KM curve.
    The first version of KM calibration [1] is by visual inspection of the KM curve and the average curve.
    The second version of KM calibration [2] is by calculating the KL divergence between the KM curve and the average.
    This function actually calculates the (normalized) integrated mean squared error
    between the KM curve and the average prediction curve.

    This version has three benefits over [1] and [2]:
    1. It is a single number, so it is easier to compare.
    2. This calculation is symmetric (note that KL-divergence is not).
    3. The score is between 0 and 1, where 0 means perfect calibration and 1 means worst calibration.
        And the random prediction curve will have a score of 0.25.

    Parameters
    ----------
    average_survival_curve: np.ndarray
        The average survival curves.
    time_coordinates: np.ndarray
        The time coordinates of the average survival curves.
    event_times: np.ndarray
        The event time of the test data.
    event_indicators: np.ndarray
        Binary event indicators: 1 denotes an observed event and 0 denotes a
        censored observation.
    interpolation_method: str, default "Linear"
        Interpolation method for evaluating the average survival curve.
        Options are "Linear" and "Pchip".
    draw_figure: bool
        Whether to visualize the comparison of the KM curve and average curve.

    Returns
    -------
    mse: float
        The (normalized) integrated mean squared error between the KM curve and the average prediction curve.
    fig: tuple(plt.Figure, plt.Axes)
        The matplotlib figure and axes objects for the calibration curve plot. Returned only if draw_figure
        is True.

    References
    ----------
    [1] Chapfuwa et al., Calibration and Uncertainty in Neural Time-to-Event Modeling， TNNLS， 2020
    [2] Yanagisawa, Proper Scoring Rules for Survival Analysis, ICML, 2023
    """
    unique_event_times = np.unique(event_times[event_indicators == 1])

    km_model = KaplanMeier(event_times, event_indicators)
    km_curve = km_model.predict(unique_event_times)

    # add 0 to the beginning of the time coordinates, and 1 to the beginning of the average_survival_curves
    unique_event_times = np.concatenate([[0], unique_event_times])
    km_curve = np.concatenate([[1], km_curve])
    if time_coordinates[0] != 0:
        time_coordinates = np.concatenate([[0], time_coordinates])
        average_survival_curve = np.concatenate([[1], average_survival_curve])

    # interpolate the average curve, so that it will have the same time coordinates as km_curve
    spline = interpolated_curve(
        time_coordinates, average_survival_curve, interpolation_method
    )
    average_survival_curve = spline(unique_event_times)
    average_survival_curve = np.clip(average_survival_curve, 0, 1)

    # integrated over the joint time coordinates
    mse = trapezoid((average_survival_curve - km_curve) ** 2, unique_event_times)
    # normalize by the maximum time coordinate
    mse /= np.max(unique_event_times)

    if draw_figure:
        fig, ax = plt.subplots(dpi=400)
        ax.plot(
            unique_event_times, average_survival_curve, label="Average Prediction Curve"
        )
        ax.plot(unique_event_times, km_curve, label="KM Curve")
        ax.fill_between(unique_event_times, average_survival_curve, km_curve, alpha=0.2)
        score_text = rf"KM-Calibration$= {mse:.3f}$"
        ax.plot([], [], " ", label=score_text)
        ax.legend()
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        plt.show()
        return mse, (fig, ax)

    return mse


def coverage_ic(
    pred_l: np.ndarray,
    pred_r: np.ndarray,
    obs_l: np.ndarray,
    obs_r: np.ndarray,
    obs_l_train: np.ndarray | None = None,
    obs_r_train: np.ndarray | None = None,
    cov_level: float = 0.95,
    method: str = "Turnbull",
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """
    Compute the Interval-Censor Coverage (IC) metric.
    The coverage is the proportion of instances where the predicted interval [t_li, t_ui]
    contains the true event time.
    Since the true event time is interval-censored between [L_i, U_i], we consider the event time to be contained
    - 100% if the predicted interval fully covers the censoring interval [L_i, U_i]
    - 0% if there is no overlap between the predicted interval and the censoring interval [L_i, U_i]
    - partial coverage if there is a partial overlap between the predicted interval and the censoring interval [L_i, U_i]

    The partial coverage is estimated using the empirical distribution of censoring intervals from the training data.
    Like what we did for the partial weights for concordance_ic().
    That means we estimate the empirical CDF of censoring intervals [L_j, U_j] from the training data using Turnbull estimator,
    and use it to compute S(L_i), S(U_i), S(t_li), S(t_ui) for each test instance.
    The partial coverage is then calculated as:
        partial_coverage = (S(max(L_i, t_li)) - S(min(U_i, t_ui))) / (S(L_i) - S(U_i))

    Parameters
    ----------
    pred_l : np.ndarray, shape (n_samples,)
        The lower bounds of the predicted intervals.
    pred_r : np.ndarray, shape (n_samples,)
        The upper bounds of the predicted intervals.
    obs_l : np.ndarray, shape (n_samples,)
        The lower bounds of the observed censoring intervals.
    obs_r : np.ndarray, shape (n_samples,)
        The upper bounds of the observed censoring intervals.
    obs_l_train : np.ndarray, shape (n_train_samples,)
        The lower bounds of the observed censoring intervals in the training data.
    obs_r_train : np.ndarray, shape (n_train_samples,)
        The upper bounds of the observed censoring intervals in the training data.
    cov_level : float, default 0.95
        Target coverage level for calibration reference.
    method : str, default "Turnbull"
        Method to compute the coverage.
        - "Turnbull": use the empirical distribution (Turnbull estimator) of censoring intervals from the training data.
        - "linear": use the overlap fraction for finite observed intervals.
          For unbounded observed intervals, finite overlap contributes 0 and
          unbounded overlap contributes 1, the limit as the upper bound grows.
    eps : float, default 1e-12
        Numerical tolerance for treating the conditional coverage denominator as zero.

    Returns
    -------
    observed_cov : float
        Average (partial) coverage across samples.
    cov_gap : float
        Difference between observed coverage and the target level (observed_cov - cov_level).
    avg_length : float
        Average length of the predicted intervals. Infinite lower bounds
        indicate unbounded predicted quantiles and give infinite width.
        Predictions [inf, inf] cover no finite event time.
    """
    if pred_l.ndim != 1 or pred_r.ndim != 1:
        raise ValueError("pred_l and pred_r must be 1-dimensional arrays.")
    if obs_l.ndim != 1 or obs_r.ndim != 1:
        raise ValueError("obs_l and obs_r must be 1-dimensional arrays.")
    if not (pred_l.shape == pred_r.shape == obs_l.shape == obs_r.shape):
        raise ValueError(
            "pred_l, pred_r, obs_l, and obs_r must contain the same number of samples."
        )

    overlap_left = np.maximum(obs_l, pred_l)
    overlap_right = np.minimum(obs_r, pred_r)
    intersects = (overlap_left <= overlap_right) & np.isfinite(overlap_left)

    method = method.lower()
    if method == "linear":
        denom = obs_r - obs_l
        numer = np.zeros_like(denom, dtype=float)
        np.subtract(overlap_right, overlap_left, out=numer, where=intersects)
    elif method == "turnbull":
        if obs_l_train is None or obs_r_train is None:
            raise ValueError(
                "obs_l_train and obs_r_train must be provided for Turnbull method."
            )
        if obs_l_train.ndim != 1 or obs_r_train.ndim != 1:
            raise ValueError(
                "obs_l_train and obs_r_train must be 1-dimensional arrays."
            )
        if np.any(obs_l_train > obs_r_train):
            raise ValueError("Found training intervals with left > right.")

        tb = TurnbullEstimatorLifelines(obs_l_train, obs_r_train)
        denom = tb.predict(obs_l) - tb.predict(obs_r)
        numer = tb.predict(overlap_left) - tb.predict(overlap_right)
    else:
        raise ValueError(f"Unknown method: {method}")

    coverage = intersects.astype(float)
    np.divide(numer, denom, out=coverage, where=(denom > eps) & np.isfinite(denom))
    if method == "linear":
        unbounded = np.isposinf(denom)
        coverage[unbounded] = intersects[unbounded] & np.isposinf(
            overlap_right[unbounded]
        )
    coverage = np.clip(coverage, 0.0, 1.0)

    widths = np.full(pred_l.shape, np.inf)
    np.subtract(pred_r, pred_l, out=widths, where=np.isfinite(pred_l))
    observed_cov = float(np.mean(coverage))
    return observed_cov, observed_cov - float(cov_level), float(np.mean(widths))


def discrepancy_to_uniform(
    x: np.ndarray, cdf: np.ndarray, x_support: tuple[float, float] | None = None
) -> float:
    """
    Compute the Kolmogorov-Smirnov (KS) statistic for one-sample test against uniform distribution.
    The KS statistic is defined as:
        D_n = max|F_n(x) - F(x)|,
    where F_n(x) is the empirical CDF and F(x) is the CDF of the uniform distribution.
    Parameters
    ----------
    x: np.ndarray
        The support points of the empirical CDF.
    cdf: np.ndarray
        The values of the empirical CDF at the support points.
    x_support: tuple[float, float] | None
        The support points of the uniform distribution. If None, it is assumed to be [0, 1].
    Returns
    -------
    D_n: float
        The KS statistic.
    """
    # sanity check
    if x.ndim != 1 or cdf.ndim != 1 or x.size != cdf.size:
        raise ValueError("x and cdf must be 1D arrays of the same length.")

    assert np.all(cdf >= 0) and np.all(
        cdf <= 1
    ), "The cdf values must be in the range [0, 1]."

    if not (np.all(np.diff(x) >= 0) and np.all(np.diff(cdf) >= 0)):
        raise ValueError("x and cdf must be nondecreasing.")

    # empirical CDF values at the support points
    F_n = cdf
    # CDF of the uniform distribution at the support points
    if x_support is None:
        F = x
    else:
        F = (x - x_support[0]) / (x_support[1] - x_support[0])

    D_plus = np.max(F_n - F)
    D_minus = np.max(F - np.concatenate(([0], F_n[:-1])))
    D_n = float(max(D_plus, D_minus))
    return D_n


def ks_pvalue(D_n: float, n: int) -> float:
    """
    Compute asymptotic KS one-sample p-value using SciPy's Kolmogorov distribution.
    This function has the same behavior as the last half part of
    scipy.stats.kstest(method='asymp') for one-sample KS test.

    But this function directly takes D_n and n as input, which is more convenient for our use case.

    The reason we do not use 'exact' method from scipy.stats.kstest is that it assumes continuous distribution,
    while in our case, the distribution estimated by empirical estimator is discrete.
    """
    lambda_n = np.sqrt(n) * D_n
    # Survival function (1 - CDF) gives the p-value directly
    return float(kstwobign.sf(lambda_n))


if __name__ == "__main__":
    # Test the KM calibration.

    # # first we define the time coordinates
    # times = np.linspace(0, 100, 11)
    # # then we define the survival probabilities at each time coordinate
    # survival_probabilities = np.exp(-times / 100)
    #
    #
    # # randomly generate the event time and event indicator for 20 samples
    # num_samples = 20
    # true_t = np.random.randint(0, 100, num_samples)
    # true_e = np.random.randint(0, 2, num_samples)
    #
    # # make some random predictions between 0 and 1
    # pred_at_observed_times = np.random.rand(num_samples)
    #
    # # # calculate the KM calibration score
    # # km_calibration_score = km_calibration(survival_probabilities, times, true_t, true_e, draw_figure=True)
    # # print(km_calibration_score)
    # # calculate the Cox-Snell residuals
    # coxsnell_residuals = residuals(pred_at_observed_times, true_e, method="Martingale", draw_figure=True)

    # test interval censored D-calibration
    pred_prob_left = np.array([0.84, 0.65, 0.42, 0.75, 1, 1])
    pred_prob_right = np.array([0.72, 0.63, 0.03, 0, 0.24, 0])
    statistics, pvalue, binning = d_cal_ic(pred_prob_left, pred_prob_right, num_bins=10)
