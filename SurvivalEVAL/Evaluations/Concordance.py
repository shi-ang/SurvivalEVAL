from typing import Optional, Tuple

import numpy as np

from SurvivalEVAL.NonparametricEstimator.SingleEvent import (
    KaplanMeierArea,
    TurnbullEstimatorLifelines,
)


def concordance(
    predicted_times: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: Optional[np.ndarray] = None,
    train_event_indicators: Optional[np.ndarray] = None,
    method: str = "Harrell",
    ties: str = "Risk",
) -> tuple[float, float, float]:
    """
    Calculate the concordance index between the predicted survival times and the true survival times.

    Parameters
    ----------
    predicted_times: np.ndarray, shape = (n_samples,)
        The predicted survival times.
    event_times: np.ndarray, shape = (n_samples,)
        The true survival times.
    event_indicators: np.ndarray, shape = (n_samples,)
        The event indicators of the true survival times.
    train_event_times: np.ndarray, shape = (n_train_samples,)
        The true survival times of the training set.
    train_event_indicators: np.ndarray, shape = (n_train_samples,)
        The event indicators of the true survival times of the training set.
    method: str, optional (default="Harrell")
        A string indicating the method for constructing the pairs of samples.
        "Harrell": the pairs are constructed by comparing the predicted survival time of each sample with the
        event time of all other samples. The pairs are only constructed between samples with comparable
        event times. For example, if i-th sample has a censor time of 10, then the pairs are constructed by
        comparing the predicted survival time of sample i with the event time of all samples with event
        time of 10 or less.
        "Margin": the pairs are constructed between all samples. A best-guess time for the censored samples
        will be calculated and used to construct the pairs.
    ties: str, optional (default="Risk")
        A string indicating the way ties should be handled.
        Options: "None" (default), "Time", "Risk", or "All"
        "None" will throw out all ties in true survival time and all ties in predict survival times (risk scores).
        "Time" includes ties in true survival time but removes ties in predict survival times (risk scores).
        "Risk" includes ties in predict survival times (risk scores) but not in true survival time.
        "All" includes all ties.
        Note the concordance calculation is given by
        (Concordant Pairs + (Number of Ties/2))/(Concordant Pairs + Discordant Pairs + Number of Ties).

    Returns
    -------
    c_index: float
        The concordance index.
    concordant_pairs: float
        The number of concordant pairs.
    total_pairs: float
        The total number of comparable pairs.
    """
    # the scikit-survival concordance function only takes risk scores to calculate.
    # So at first we should transfer the predicted time -> risk score.
    # The risk score should be higher for subjects that live shorter (i.e. lower average survival time).

    event_indicators = event_indicators.astype(bool)

    assert (
        len(predicted_times) == len(event_times) == len(event_indicators)
    ), "The lengths of the predicted times and labels must be the same."

    if method == "Harrell":
        risks = -1 * predicted_times
        partial_weights = None
        bg_event_times = None
    elif method == "Margin":
        if train_event_times is None or train_event_indicators is None:
            error = "If 'Margin' is chosen, training set information must be provided."
            raise ValueError(error)

        train_event_indicators = train_event_indicators.astype(bool)

        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = -1 / (
            (1 - min(km_model.survival_probabilities))
            / (0 - max(km_model.survival_times))
        )
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)
        risks = -1 * predicted_times

        censor_times = event_times[~event_indicators]
        partial_weights = np.ones_like(event_indicators, dtype=float)
        partial_weights[~event_indicators] = 1 - km_model.predict(censor_times)

        best_guesses = km_model.best_guess(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[
            censor_times > km_linear_zero
        ]

        bg_event_times = np.copy(event_times)
        bg_event_times[~event_indicators] = best_guesses
    else:
        raise TypeError("Method for calculating concordance is unrecognized.")
    # risk_ties means predicted times are the same while true times are different.
    # time_ties means true times are the same while predicted times are different.
    # c_index, concordant_pairs, discordant_pairs, risk_ties, time_ties = metrics.concordance_index_censored(
    #     event_indicators, event_times, estimate=risk)
    c_index, concordant_pairs, discordant_pairs, risk_ties, time_ties = (
        _estimate_concordance_index(
            event_indicators,
            event_times,
            estimate=risks,
            bg_event_time=bg_event_times,
            partial_weights=partial_weights,
        )
    )
    if ties == "None":
        total_pairs = concordant_pairs + discordant_pairs
        c_index = concordant_pairs / total_pairs
    elif ties == "Time":
        total_pairs = concordant_pairs + discordant_pairs + time_ties
        concordant_pairs = concordant_pairs + 0.5 * time_ties
        c_index = concordant_pairs / total_pairs
    elif ties == "Risk":
        # This should be the same as original outputted c_index from above
        total_pairs = concordant_pairs + discordant_pairs + risk_ties
        concordant_pairs = concordant_pairs + 0.5 * risk_ties
        c_index = concordant_pairs / total_pairs
    elif ties == "All":
        total_pairs = concordant_pairs + discordant_pairs + risk_ties + time_ties
        concordant_pairs = concordant_pairs + 0.5 * (risk_ties + time_ties)
        c_index = concordant_pairs / total_pairs
    else:
        error = "Please enter one of 'None', 'Time', 'Risk', or 'All' for handling ties for concordance."
        raise TypeError(error)

    return c_index, concordant_pairs, total_pairs


def _estimate_concordance_index(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    estimate: np.ndarray,
    bg_event_time: np.ndarray = None,
    partial_weights: np.ndarray = None,
    tied_tol: float = 1e-8,
) -> tuple[float, float, float, float, float]:
    """
    Estimate the concordance index.
    This backbone of this function is borrowed from scikit-survival:
    https://github.com/sebp/scikit-survival/blob/4e664d8e4fe5e5b55006e3913f2bbabcf2455496/sksurv/metrics.py#L85-L118
    In here, we make modifications to the original function to allow for partial weights and best-guess times (margin times).

    All functions in scikit-survival are licensed under the GPLv3 License:
    https://github.com/sebp/scikit-survival/blob/master/COPYING

    Parameters
    ----------
    event_indicator: np.ndarray, shape = (n_samples,)
        The event indicators of the true survival times.
    event_time: np.ndarray, shape = (n_samples,)
        The true survival times.
    estimate: np.ndarray, shape = (n_samples,)
        The estimated risk scores. A higher score should correspond to a higher risk.
    bg_event_time: np.ndarray, shape = (n_samples,), optional (default=None)
        The best-guess event times. For uncensored samples, this should be the same as the true event times.
        For censored samples, this should be the best-guess time (margin time) for the censored samples.
    partial_weights: np.ndarray, shape = (n_samples,), optional (default=None)
        The partial weights for the censored samples.
    tied_tol: float, optional (default=1e-8)
        The tolerance for considering two times as tied.

    Returns
    -------
    c_index: float
        The concordance index.
    concordant: float
        The number of concordant pairs.
    discordant: float
        The number of discordant pairs.
    tied_risk: float
        The number of tied risk scores.
    tied_time: float
        The number of tied times.
    -------
    """
    order = np.argsort(event_time, kind="stable")

    comparable, tied_time, weight = _get_comparable(event_indicator, event_time, order)

    if partial_weights is not None:
        event_indicator = np.ones_like(event_indicator)
        comparable_2, tied_time, weight = _get_comparable(
            event_indicator, bg_event_time, order, partial_weights
        )
        for ind, mask in comparable.items():
            weight[ind][mask] = 1
        comparable = comparable_2

    if len(comparable) == 0:
        raise ValueError(
            "Data has no comparable pairs, cannot estimate concordance index."
        )

    concordant = 0
    discordant = 0
    tied_risk = 0
    numerator = 0.0
    denominator = 0.0
    for ind, mask in comparable.items():
        est_i = estimate[order[ind]]
        event_i = event_indicator[order[ind]]
        # w_i = partial_weights[order[ind]] # change this
        w_i = weight[ind]
        weight_i = w_i[order[mask]]

        est = estimate[order[mask]]

        assert event_i, (
            "got censored sample at index %d, but expected uncensored" % order[ind]
        )

        ties = np.absolute(est - est_i) <= tied_tol
        # n_ties = ties.sum()
        n_ties = np.dot(weight_i, ties.T)
        # an event should have a higher score
        con = est < est_i
        # n_con = con[~ties].sum()
        con[ties] = False
        n_con = np.dot(weight_i, con.T)

        # numerator += w_i * n_con + 0.5 * w_i * n_ties
        # denominator += w_i * mask.sum()
        numerator += n_con + 0.5 * n_ties
        denominator += np.dot(w_i, mask.T)

        tied_risk += n_ties
        concordant += n_con
        # discordant += est.size - n_con - n_ties
        discordant += np.dot(w_i, mask.T) - n_con - n_ties

    c_index = numerator / denominator
    return c_index, concordant, discordant, tied_risk, tied_time


def _get_comparable(
    event_indicator: np.ndarray,
    event_time: np.ndarray,
    order: np.ndarray,
    partial_weights: np.ndarray = None,
) -> tuple[dict, int, dict]:
    """
    Given the labels of the survival outcomes, get the comparable pairs.

    This backbone of this function is borrowed from scikit-survival:
    https://github.com/sebp/scikit-survival/blob/4e664d8e4fe5e5b55006e3913f2bbabcf2455496/sksurv/metrics.py#L57-L81
    In here, we make modifications to the original function to calculates the weights for each pair.

    All functions in scikit-survival are licensed under the GPLv3 License:
    https://github.com/sebp/scikit-survival/blob/master/COPYING

    Parameters
    ----------
    event_indicator: np.ndarray, shape = (n_samples,)
        The event indicators of the true survival times.
    event_time: np.ndarray, shape = (n_samples,)
        The true survival times.
    order: np.ndarray, shape = (n_samples,)
        The indices that would sort the event times.
    partial_weights: np.ndarray, shape = (n_samples,), optional (default=None)
        The partial weights for the censored samples.

    Returns
    -------
    comparable: dict
        A dictionary where the keys are the indices of the samples with events, and the values are boolean masks
        indicating which samples are comparable to the key sample.
    tied_time: int
        The number of tied times.
    weight: dict
        A dictionary where the keys are the indices of the samples with events, and the values are the weights
        for each comparable sample.
    """

    if partial_weights is None:
        partial_weights = np.ones_like(event_indicator, dtype=float)
    n_samples = len(event_time)
    tied_time = 0
    comparable = {}
    weight = {}

    i = 0
    while i < n_samples - 1:
        time_i = event_time[order[i]]
        end = i + 1
        while end < n_samples and event_time[order[end]] == time_i:
            end += 1

        # check for tied event times
        event_at_same_time = event_indicator[order[i:end]]
        censored_at_same_time = ~event_at_same_time

        for j in range(i, end):
            if event_indicator[order[j]]:
                mask = np.zeros(n_samples, dtype=bool)
                mask[end:] = True
                # an event is comparable to censored samples at same time point
                mask[i:end] = censored_at_same_time
                comparable[j] = mask
                tied_time += censored_at_same_time.sum()
                weight[j] = partial_weights[order] * partial_weights[order[j]]
        i = end

    return comparable, tied_time, weight


def pairwise_w(S_Li, S_Ri, eps=1e-12):
    """
    Compute pair weights w_{i<j} for interval-censored pairs.

    Parameters
    ----------
    S_Li, S_Ri : (n,) arrays
        Survival probabilities at left and right endpoints of intervals.
    eps : float
        Numerical tolerance for zero denominators.

    Returns
    -------
    w : (n, n) array
        Pairwise weights w_{i<j}.
    """
    # column/row views for broadcasting
    S_Li = S_Li[:, None]  # (n,1)
    S_Ri = S_Ri[:, None]  # (n,1)

    S_Lj = S_Li.T  # (1,n)
    S_Rj = S_Ri.T  # (1,n)

    # Monotonicity shortcuts:
    # S(max(Lj, Ri)) = min(S(Lj), S(Ri)),  S(l_max) = min(S(Lj), S(Li)),  S(r_min) = max(S(Rj), S(Ri))
    S_maxLj_ri = np.minimum(S_Lj, S_Ri)  # (n,n)
    S_lmax = np.minimum(S_Lj, S_Li)  # (n,n)
    S_rmin = np.maximum(S_Rj, S_Ri)  # (n,n)

    pos = lambda x: np.clip(x, 0.0, None)

    # Base denominator and J terms
    denom = (S_Li - S_Ri) * (S_Lj - S_Rj)  # (n,n)
    J1 = (S_Li - 0.5 * S_lmax - 0.5 * S_rmin) * pos(S_lmax - S_rmin)
    J2 = (S_Li - S_Ri) * pos(S_maxLj_ri - S_Rj)
    J = J1 + J2

    # Masks for the three edge cases:
    # A: (S_Li - S_Ri) == 0  (row-wise condition broadcast over columns)
    # B: (S_Lj - S_Rj) == 0  (column-wise condition broadcast over rows)
    A = np.isclose(S_Li - S_Ri, 0.0, atol=eps)  # (n,1)
    B = np.isclose(S_Lj - S_Rj, 0.0, atol=eps)  # (1,n)

    only_i = A & (~B)  # denominator zero due to i
    only_j = (~A) & B  # denominator zero due to j
    both = A & B  # both zero
    neither = (~A) & (~B)

    w = np.zeros_like(denom, dtype=float)

    # Case 0: neither zero -> use main formula
    if np.any(neither):
        with np.errstate(divide="ignore", invalid="ignore"):
            w[neither] = (J / np.where(np.abs(denom) > eps, denom, np.inf))[neither]

    # Case 1: (S_Li - S_Ri) == 0, (S_Lj - S_Rj) != 0
    # w = ((S(max(l_j,l_i)) - S(r_j))_+) / (S(l_j) - S(r_j))
    if np.any(only_i):
        num1 = pos(S_lmax - S_Rj)  # (n,n)
        den1 = S_Lj - S_Rj  # (n,n)
        w[only_i] = (num1 / np.clip(den1, eps, None))[only_i]

    # Case 2: (S_Lj - S_Rj) == 0, (S_Li - S_Ri) != 0
    # w = ((S(l_i) - S(min(r_i,r_j)))_+) / (S(l_i) - S(r_i))
    if np.any(only_j):
        num2 = pos(S_Li - S_rmin)  # (n,n)
        den2 = S_Li - S_Ri  # (n,1) broadcast
        w[only_j] = (num2 / np.clip(den2, eps, None))[only_j]

    # Case 3: both zero -> compare S_Li vs S_Lj
    # if S_Li > S_Lj => w=1 else 0
    if np.any(both):
        w[both] = (S_Li > S_Lj)[both].astype(float)

    # Clean up: zero diagonal, clamp tiny negatives, and (optionally) cap at 1
    np.fill_diagonal(w, 0.0)
    w = np.clip(w, 0.0, 1.0)

    return w


def concordance_ic(
    eta: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    left_train: np.ndarray,
    right_train: np.ndarray,
    method: str = "probability",
    ties: str = "skip",
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Concordance index for interval-censored outcomes using closed-form pair weights.

    Parameters
    ----------
    eta : np.ndarray of shape (n_sample,)
        Predicted risk scores  (higher = riskier).
    left : np.ndarray of shape (n_sample,)
        Left endpoints l_i (can be -inf).
    right : np.ndarray of shape (n_sample,)
        Right endpoints r_i (can be +inf to represent right censoring).
    left_train : np.ndarray of shape (n_train_sample,)
        Left endpoints of training data for Turnbull estimator.
    right_train : np.ndarray of shape (n_train_sample,)
        Right endpoints of training data for Turnbull estimator.
    method : {"probability", "midpoint"}, default="probability"
        Method for forming pair weights:
          - "probability": use closed-form pair weights based on Turnbull estimator.
          - "midpoint": use standard right-censored C-index on midpoint imputed times.
    ties : {"skip", "half"}, default="skip"
        How to handle ties in eta:
          - "skip": pairs with eta_i == eta_j contribute 0 to the numerator.
          - "half":  ties contribute 0.5 * w_{i<j} to the numerator.
    eps : float, default=1e-12
        Numerical guard to avoid division by ~0.

    Returns
    -------
    c_index : float
        The interval-censored concordance index as in Eq. (cindex_ic).
        Returns np.nan if the total weight denominator is 0.
    num_matrix : np.ndarray of shape (n_sample, n_sample)
        per-pair contributions to numerator,
    den_matrix : np.ndarray of shape (n_sample, n_sample)
        per-pair weights (same as weights) in denominator.
    """
    eta = np.asarray(eta, dtype=float)
    l = np.asarray(left, dtype=float)
    r = np.asarray(right, dtype=float)
    l_train = np.asarray(left_train, dtype=float)
    r_train = np.asarray(right_train, dtype=float)
    n = eta.shape[0]
    if l.shape != (n,) or r.shape != (n,):
        raise ValueError("eta, left, right must all be 1-D arrays of same length.")

    # Basic sanity
    if np.any(l > r):
        raise ValueError("Found an interval with left > right in testing data.")
    if np.any(l_train > r_train):
        raise ValueError("Found an interval with left > right in training data.")

    # train Turnbull estimator on training data
    tb = TurnbullEstimatorLifelines(l_train, r_train)

    S_l = tb.predict(l)
    S_r = tb.predict(r)
    if method == "midpoint":
        pass
    elif method == "probability":
        w = pairwise_w(S_l, S_r, eps=eps)
    else:
        raise ValueError("method must be 'probability' or 'midpoint'.")

    # Concordant matrix based on eta
    gt = (eta[:, None] > eta[None, :]).astype(float)
    if ties == "half":
        gt += 0.5 * (eta[:, None] == eta[None, :]).astype(float)
        # still zero on diagonal because w_ii is 0
    elif ties == "skip":
        pass
    else:
        raise ValueError("'ties' must be 'skip' or 'half'.")

    # Numerator and denominator (sum over ordered pairs i != j)
    num = np.sum(gt * w)
    den = np.sum(w)

    c_idx = num / den if den > 0 else float("nan")

    return c_idx, gt * w, w


def impute_times_midpoint(
    left: np.ndarray,
    right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    according to (left, right] intervals, construct (t, delta) via "endpoint/midpoint imputation".

    rules:
      - Interval censoring: both L and R finite => t = (L+R)/2, delta=1
      - Right censor: L finite, R non-finite => t = L, delta= 0
      - Left censor: L non-finite, R finite => t = R, delta= 1
      - Other (both non-finite) => discard

    Returns
    -------
    t : observed/imputed time
    delta : event indicator (1=event, 0=censor)
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    is_L_finite = np.isfinite(left)
    is_R_finite = np.isfinite(right)

    n = left.shape[0]
    t = np.empty(n, dtype=float)
    delta = np.empty(n, dtype=int)

    # interval censoring：L limited, R limited -> use midpoint as event time
    mask_interval = is_L_finite & is_R_finite
    t[mask_interval] = 0.5 * (left[mask_interval] + right[mask_interval])
    delta[mask_interval] = 1

    # left censor: L non-finite, R finite -> use R as event time
    mask_left_cens = (~is_L_finite) & is_R_finite
    t[mask_left_cens] = right[mask_left_cens]
    delta[mask_left_cens] = 1

    # right censor: L finite, R non-finite -> use L, mark as censor
    mask_right_cens = is_L_finite & (~is_R_finite)
    t[mask_right_cens] = left[mask_right_cens]
    delta[mask_right_cens] = 0

    # other: both non-finite -> discard
    mask_bad = (~is_L_finite) & (~is_R_finite)
    if mask_bad.any():
        t[mask_bad] = np.nan
        delta[mask_bad] = -1  # mark as invalid

    # remove invalid entries
    valid = ~np.isnan(t) & (delta >= 0)
    return t[valid], delta[valid]
