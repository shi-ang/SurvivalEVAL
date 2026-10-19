"""Concordance and midpoint imputation for interval-censored outcomes."""

from __future__ import annotations

import numpy as np

from SurvivalEVAL.nonparametric_estimator.single_event import TurnbullEstimatorLifelines


def _get_comparable_ic(
    left: np.ndarray,
    right: np.ndarray,
    tol: float = 0.0,
) -> np.ndarray:
    """
    Return the directed comparability relation for interval-censored outcomes.

    ``comparable[i, j]`` is True when interval ``i`` lies entirely before
    interval ``j``, so the event for sample ``i`` is known to precede the event
    for sample ``j``. For a disjoint pair, exactly one of ``[i, j]`` and
    ``[j, i]`` is True; this matrix is intentionally not symmetric.

    By default, intervals are left-open, right-closed -- aka (l, r].
    Exact-time events are represented by left == right -- [t, t].

    Parameters
    ----------
    left, right : (n,) arrays
        Left and right endpoints of intervals.
    tol : float
        Tolerance for treating left==right / boundary equality for float times.

    Returns
    -------
    comparable : (n, n) bool array
        Directed precedence matrix with a False diagonal.
    """
    if left.size == 0:
        return np.zeros((0, 0), dtype=bool)

    # Identify exact times: treat as point [t, t]
    if tol > 0:
        exact = np.isclose(left, right, rtol=0.0, atol=tol)
    else:
        exact = left == right

    # Boundary inclusivity:
    # - Default (l, r]: left_inclusive=False, right_inclusive=True
    # - Exact [t, t]: left_inclusive=True, right_inclusive=True
    left_incl = exact  # only exact points include the left endpoint
    right_incl = np.ones(
        left.size, dtype=bool
    )  # (l, r] always includes right; exact includes too

    # Disjointness test for two 1D intervals with endpoint inclusivity:
    # i is strictly before j if:
    #   R_i < L_j  OR  (R_i == L_j AND NOT (R_i_inclusive AND L_j_inclusive))
    Ri = right[:, None]
    Lj = left[None, :]
    R_incl_i = right_incl[:, None]
    L_incl_j = left_incl[None, :]

    # check, for every pair (i,j), whether R_i == L_j
    if tol > 0:
        eq = np.isclose(Ri, Lj, rtol=0.0, atol=tol)
    else:
        eq = Ri == Lj

    # if equal, check whether both endpoints are inclusive
    is_not_equal_inclusive = eq & ~(R_incl_i & L_incl_j)

    # last step: i before j
    before_ij = (Ri < Lj) | is_not_equal_inclusive

    np.fill_diagonal(before_ij, False)
    return before_ij


def _pairwise_w(
    S_Li: np.ndarray,
    S_Ri: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
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

    def pos(x: np.ndarray) -> np.ndarray:
        """
        Return the elementwise positive part of an array.

        Parameters
        ----------
        x : ndarray
            Array whose positive part is required.

        Returns
        -------
        ndarray
            Elementwise positive part, with negative entries replaced by zero.
        """
        return np.clip(x, 0.0, None)

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
    left_train: np.ndarray | None = None,
    right_train: np.ndarray | None = None,
    method: str = "comparable",
    ties: str = "skip",
    eps: float = 1e-12,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Concordance index for interval-censored outcomes using closed-form pair weights.

    For ``method="probability"``, pair-ordering probabilities follow the approach
    of Tsouprou [1]_, with survival probabilities estimated by the Turnbull estimator.

    Parameters
    ----------
    eta : ndarray of shape (n_sample,)
        Predicted risk scores  (higher = riskier).
    left : ndarray of shape (n_sample,)
        Left endpoints l_i (can be -inf).
    right : ndarray of shape (n_sample,)
        Right endpoints r_i (can be +inf to represent right censoring).
    left_train : ndarray | None = None, shape (n_train_sample,)
        Left endpoints of training data for Turnbull estimator.
    right_train : ndarray | None = None, shape (n_train_sample,)
        Right endpoints of training data for Turnbull estimator.
    method : {"comparable", "probability"}, default="comparable"
        Method for forming pair weights:
          - "comparable": use only comparable pairs.
          - "probability": use closed-form pair weights based on Turnbull estimator.
    ties : {"skip", "half"}, default="skip"
        How to handle ties in eta:
          - "skip": pairs with eta_i == eta_j contribute 0 to the numerator.
          - "half":  ties contribute 0.5 * w_{i<j} to the numerator.
    eps : float, default=1e-12
        Numerical guard.

    Returns
    -------
    c_index : float
        The interval-censored concordance index as in Eq. (cindex_ic).
        Returns np.nan if the total weight denominator is 0.
    num_matrix : ndarray of shape (n_sample, n_sample)
        per-pair contributions to numerator,
    den_matrix : ndarray of shape (n_sample, n_sample)
        per-pair weights (same as weights) in denominator.

    References
    ----------
    .. [1] Tsouprou, Sofia (2015). "Measures of discrimination and predictive
       accuracy for interval censored survival data". Master's thesis,
       Leiden University. https://math.leidenuniv.nl/scripties/MasterTsouprou.pdf
    """
    eta = np.asarray(eta, dtype=float)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    n = eta.shape[0]
    if left.shape != (n,) or right.shape != (n,):
        raise ValueError("eta, left, right must all be 1-D arrays of same length.")

    # Basic sanity
    if np.any(left > right):
        raise ValueError("Found an interval with left > right in testing data.")

    method = method.lower()
    ties = ties.lower()

    if method not in {"comparable", "probability"}:
        raise ValueError("method must be 'comparable' or 'probability'.")
    if ties not in {"skip", "half"}:
        raise ValueError("'ties' must be 'skip' or 'half'.")

    if method == "probability":
        if left_train is None or right_train is None:
            raise ValueError(
                "Training data must be provided when method='probability'."
            )
        l_train = np.asarray(left_train, dtype=float)
        r_train = np.asarray(right_train, dtype=float)
        if np.any(l_train > r_train):
            raise ValueError("Found an interval with left > right in training data.")

        # train Turnbull estimator on training data
        tb = TurnbullEstimatorLifelines(l_train, r_train)

        S_l = tb.predict(left)
        S_r = tb.predict(right)

        w = _pairwise_w(S_l, S_r, eps=eps)
    else:
        comparable = _get_comparable_ic(left, right, tol=eps)
        w = comparable.astype(float)
        del comparable

    # Build the numerator contribution matrix in place.
    concordant_weights = (eta[:, None] > eta[None, :]).astype(float)
    if ties == "half":
        tied_pairs = eta[:, None] == eta[None, :]
        concordant_weights[tied_pairs] = 0.5
        del tied_pairs
        # The diagonal is removed when multiplying by the zero-diagonal weights.

    # Numerator and denominator (sum over ordered pairs i != j)
    concordant_weights *= w
    num = np.sum(concordant_weights)
    den = np.sum(w)

    c_idx = num / den if den > 0 else float("nan")

    return c_idx, concordant_weights, w


def impute_times_midpoint(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Impute observed times and event indicators from observation intervals.

    Parameters
    ----------
    left, right : ndarray
        Left and right interval endpoints. Left endpoints must be finite and
        non-negative. Use a left endpoint of 0 for left-censored observations
        and a right endpoint of ``np.inf`` for right-censored observations.

    Returns
    -------
    t : ndarray, shape (n_samples,)
        Midpoint of each finite interval, or its left endpoint for right-censored observations.
    delta : ndarray, shape (n_samples,)
        Event indicators: 1 for finite right bounds and 0 for right censoring.
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)

    if left.shape != right.shape:
        raise ValueError("left and right must have the same shape.")
    if np.any(~np.isfinite(left)) or np.any(left < 0):
        raise ValueError("Left endpoints must be finite and non-negative.")
    if np.any(left > right):
        raise ValueError(
            "Left endpoints must be less than or equal to right endpoints."
        )

    is_R_finite = np.isfinite(right)

    n = left.shape[0]
    t = np.empty(n, dtype=float)
    delta = np.empty(n, dtype=int)

    # Interval censoring: use midpoint as event time.
    mask_interval = is_R_finite
    t[mask_interval] = 0.5 * (left[mask_interval] + right[mask_interval])
    delta[mask_interval] = 1

    # Right censoring: use the finite left endpoint and mark as censored.
    mask_right_cens = ~is_R_finite
    t[mask_right_cens] = left[mask_right_cens]
    delta[mask_right_cens] = 0

    return t, delta
