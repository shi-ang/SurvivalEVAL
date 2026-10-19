from __future__ import annotations

import numpy as np
from tqdm import trange

from SurvivalEVAL.evaluations.util import (
    _inverse_censoring_weights,
    check_and_convert_event_data,
)
from SurvivalEVAL.nonparametric_estimator.single_event import (
    KaplanMeier,
    KaplanMeierArea,
)
from SurvivalEVAL.nonparametric_estimator.util import km_mean


def mean_error(
    predicted_times: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    train_event_times: np.ndarray | None = None,
    train_event_indicators: np.ndarray | None = None,
    error_type: str = "absolute",
    method: str = "Hinge",
    weighted: bool = True,
    log_scale: bool = False,
    verbose: bool = False,
    truncation_time: float | None = None,
) -> float:
    """
    Calculate the mean absolute/squared error score for the predicted survival times.

    Parameters
    ----------
    predicted_times : ndarray, shape = (n_samples, )
        Predicted survival times for the testing samples
    event_times : ndarray, shape = (n_samples, )
        Actual event/censor time for the testing samples.
    event_indicators : ndarray, shape = (n_samples, )
        Binary event indicators for the testing samples: 1 denotes an observed
        event and 0 denotes a censored observation.
    train_event_times : ndarray, shape = (n_train_samples, )
        Actual event/censor time for the training samples.
    train_event_indicators : ndarray, shape = (n_train_samples, )
        Binary event indicators for the training samples: 1 denotes an observed
        event and 0 denotes a censored observation.
    error_type : str, default="absolute"
        Type of mean error to use. Options are "absolute" and "squared".
    method : str, default="Hinge"
        Method of handling censorship.
        Options are "Uncensored", "Hinge", "Margin", "IPCW-T", "IPCW-D", and "Pseudo_obs".
        The ``Hinge`` method uses the one-sided censored-target loss in [1]_.
        The MAE formulations of "IPCW-D", "IPCW-T", and "Pseudo_obs" are described
        in Sections 3.4, 3.5, and 3.6 of [2]_, respectively.
        IPCW-D weights observed-event errors by 1/G(event_time-) from reverse KM
        fitted on training outcomes, with linear extrapolation beyond follow-up.
        Nonpositive or nonfinite contributing probabilities raise ValueError.
    weighted : bool, default=True
        Whether to use weighting scheme for MAE.
        If true, each censored sample has confidence weight
        1 - S_KM(censoring time).
    log_scale : bool, default=False
        Whether to use log scale for the loss function.
    verbose : bool, default=False
        Whether to show the progress bar.
    truncation_time : float | None
        Optional upper limit for event-time errors in Margin, IPCW-T, IPCW-D, and Pseudo_obs calculations.
        For the Margin method, see the L1-margin-truncated metric in [3]_.

    Returns
    -------
    float
        Mean absolute or squared prediction error under the selected censoring method.

    References
    ----------
    .. [1] Shivaswamy, Chu and Jansche (2007). "A Support Vector Approach to Censored
       Targets". ICDM, 655-660.
       https://ieeexplore.ieee.org/document/4470306/
    .. [2] Qi, S.-A.; Kumar, N.; Farrokh, M.; Sun, W.; Kuan, L.-H.;
       Ranganath, R.; Henao, R.; Greiner, R. (2023).
       "An Effective Meaningful Way to Evaluate Survival Models".
       Proceedings of the 40th International Conference on Machine Learning,
       Proceedings of Machine Learning Research, 202, 28244-28276.
       https://proceedings.mlr.press/v202/qi23b.html
    .. [3] Sun, W.; Kalmady, S. V.; Qi, S.-A.; Sepehrvand, N.; Hindle, A.;
       Greiner, R.; Kaul, P. (2024). "Predicting Individual Survival Distributions
       Using ECG: A Deep Learning Approach Utilizing Features Extracted by a
       Learned Diagnostic Model". Proceedings of the AAAI Symposium Series,
       2(1), 475-481. https://doi.org/10.1609/aaaiss.v2i1.27716
    """
    error_type = error_type.lower()
    method = method.lower()
    if method == "ipcw-d":
        event_times, event_indicators = check_and_convert_event_data(
            event_times, event_indicators
        )
    else:
        event_indicators = event_indicators.astype(bool, copy=False)
        if train_event_indicators is not None:
            train_event_indicators = train_event_indicators.astype(bool, copy=False)
    n_test = event_times.size

    if method in ["margin", "ipcw-t", "ipcw-d", "pseudo_obs"] and (
        train_event_times is None or train_event_indicators is None
    ):
        raise ValueError(
            f"If method is '{method}', training set values must be included."
        )

    # IPCW-D only needs censoring survival, not event-survival areas or weights.
    if method in ["margin", "ipcw-t", "pseudo_obs"]:
        km_model = KaplanMeierArea(train_event_times, train_event_indicators)
        km_linear_zero = km_model.km_linear_zero
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_model.survival_times)
        # predicted_times = np.clip(predicted_times, a_max=km_linear_zero, a_min=None)

        censor_times = event_times[~event_indicators]
        weights = np.ones(n_test)
        if weighted:
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

    # set the error function
    if error_type == "absolute":
        error_func = np.abs
    elif error_type == "squared":
        error_func = np.square
    else:
        raise TypeError(
            "Please enter one of 'absolute' or 'squared' for calculating error."
        )

    if method == "uncensored":
        # only use uncensored data
        if log_scale:
            errors = np.log(event_times[event_indicators]) - np.log(
                predicted_times[event_indicators]
            )
        else:
            errors = event_times[event_indicators] - predicted_times[event_indicators]
        return float(np.mean(error_func(errors), dtype=float))
    elif method == "hinge":
        # consider only the early prediction error
        # if prediction is higher than the censored time, it is not penalized
        weights = np.ones(predicted_times.size)
        if weighted:
            if train_event_times is None or train_event_indicators is None:
                raise ValueError(
                    "If 'weighted' is True for calculating Hinge, training set values must be included."
                )
            km_model = KaplanMeierArea(train_event_times, train_event_indicators)
            censor_times = event_times[~event_indicators]
            weights[~event_indicators] = 1 - km_model.predict(censor_times)

        if log_scale:
            errors = np.log(event_times) - np.log(predicted_times)
        else:
            errors = event_times - predicted_times
        errors[~event_indicators] = np.maximum(errors[~event_indicators], 0)
        return np.average(error_func(errors), weights=weights)
    elif method == "margin":
        # The L1-margin method proposed by https://www.jmlr.org/papers/v21/18-772.html
        # Calculate the best guess survival time given the KM curve and censoring time of that patient
        best_guesses = km_model.best_guess(censor_times)
        best_guesses[censor_times > km_linear_zero] = censor_times[
            censor_times > km_linear_zero
        ]

        if truncation_time:
            best_guesses = np.clip(best_guesses, a_max=truncation_time, a_min=None)
            predicted_times = np.clip(
                predicted_times, a_max=truncation_time, a_min=None
            )
            event_times = np.clip(event_times, a_max=truncation_time, a_min=None)

        errors = np.empty(predicted_times.size)
        if log_scale:
            errors[event_indicators] = np.log(event_times[event_indicators]) - np.log(
                predicted_times[event_indicators]
            )
            errors[~event_indicators] = np.log(best_guesses) - np.log(
                predicted_times[~event_indicators]
            )
        else:
            errors[event_indicators] = (
                event_times[event_indicators] - predicted_times[event_indicators]
            )
            errors[~event_indicators] = (
                best_guesses - predicted_times[~event_indicators]
            )
        return np.average(error_func(errors), weights=weights)
    elif method == "ipcw-t":
        # This is the IPCW-T method from https://arxiv.org/pdf/2306.01196.pdf
        # Calculate the best guess time (surrogate time) based on the subsequent uncensored subjects
        best_guesses = event_times.astype(float, copy=True)
        if censor_times.size:
            observed_train_times = np.sort(train_event_times[train_event_indicators])
            # Each query needs the mean of a suffix. Precompute suffix sums
            # once instead of scanning the training set for every censor time.
            tail_sums = np.append(
                np.cumsum(observed_train_times[::-1], dtype=float)[::-1], 0.0
            )
            starts = np.searchsorted(observed_train_times, censor_times, side="right")
            counts = observed_train_times.size - starts
            best_guesses[~event_indicators] = np.divide(
                tail_sums[starts],
                counts,
                out=np.full(censor_times.shape, np.nan),
                where=counts > 0,
            )

        # Exclude censor times with no strictly later training events.
        valid = ~np.isnan(best_guesses)
        predicted_times = predicted_times[valid]
        best_guesses = best_guesses[valid]
        weights = weights[valid]

        if truncation_time:
            best_guesses = np.clip(best_guesses, a_max=truncation_time, a_min=None)
            predicted_times = np.clip(
                predicted_times, a_max=truncation_time, a_min=None
            )

        if log_scale:
            errors = np.log(best_guesses) - np.log(predicted_times)
        else:
            errors = best_guesses - predicted_times
        return np.average(error_func(errors), weights=weights)
    elif method == "ipcw-d":
        # This is the IPCW-D method from https://arxiv.org/pdf/2306.01196.pdf
        # Using IPCW weights to transfer the censored subjects to uncensored subjects
        train_times, train_events = check_and_convert_event_data(
            train_event_times, train_event_indicators
        )
        if not np.any(event_indicators):
            raise ValueError("IPCW-D requires at least one observed event.")
        ipc_model = KaplanMeier(train_times, train_events, reverse=True)
        event_weights = _inverse_censoring_weights(
            ipc_model.predict(event_times[event_indicators], side="left")
        )

        if truncation_time:
            event_times = np.clip(event_times, a_max=truncation_time, a_min=None)
            predicted_times = np.clip(
                predicted_times, a_max=truncation_time, a_min=None
            )

        if log_scale:
            errors = np.log(event_times) - np.log(predicted_times)
        else:
            errors = event_times - predicted_times
        return (error_func(errors)[event_indicators] * event_weights).mean()
    elif method == "pseudo_obs":
        # Calculate the best guess time (surrogate time) by the contribution of the censored subjects to KM curve
        n_train = train_event_times.size

        events, population_counts = (
            km_model.events.copy(),
            km_model.population_count.copy(),
        )
        times = km_model.survival_times.copy()
        probs = km_model.survival_probabilities.copy()
        # get the discrete time points where the event happens, then calculate the area under those discrete time only
        # this doesn't make any difference for step function, but it does for trapezoid rule.
        unique_idx = np.where(events != 0)[0]
        if unique_idx[-1] != len(events) - 1:
            unique_idx = np.append(unique_idx, len(events) - 1)
        times = times[unique_idx]
        population_counts = population_counts[unique_idx]
        events = events[unique_idx]
        probs = probs[unique_idx]
        sub_expect_time = km_mean(times.copy(), probs.copy())

        # use the idea of dynamic programming to calculate the multiplier of the KM estimator in advances.
        # if we add a new time point to the KM curve, the multiplier before the new time point will be
        # 1 - event_counts / (population_counts + 1), and the multiplier after the new time point will be
        # the same as before.
        multiplier = 1 - events / population_counts
        multiplier_total = 1 - events / (population_counts + 1)
        best_guesses = event_times.copy().astype(float)

        for i in trange(
            n_test,
            desc="Calculating surrogate times for Pseudo_obs",
            disable=not verbose,
        ):
            if event_indicators[i] != 1:
                total_multiplier = multiplier.copy()
                insert_index = np.searchsorted(times, event_times[i], side="right")
                total_multiplier[:insert_index] = multiplier_total[:insert_index]
                survival_probabilities = np.cumprod(total_multiplier)
                if insert_index == len(times):
                    times_addition = np.append(times, event_times[i])
                    survival_probabilities_addition = np.append(
                        survival_probabilities, survival_probabilities[-1]
                    )
                    total_expect_time = km_mean(
                        times_addition, survival_probabilities_addition
                    )
                else:
                    total_expect_time = km_mean(times, survival_probabilities)
                best_guesses[i] = (
                    n_train + 1
                ) * total_expect_time - n_train * sub_expect_time

        if truncation_time:
            best_guesses = np.clip(best_guesses, a_max=truncation_time, a_min=None)
            predicted_times = np.clip(
                predicted_times, a_max=truncation_time, a_min=None
            )

        if log_scale:
            errors = np.log(best_guesses) - np.log(predicted_times)
        else:
            errors = best_guesses - predicted_times
        return np.average(error_func(errors), weights=weights)
    else:
        raise ValueError(
            "Method must be one of 'Uncensored', 'Hinge', 'Margin', 'IPCW-T', 'IPCW-D' "
            f"or 'Pseudo_obs'. Got '{method}' instead."
        )


def _prepare_interval_arrays(
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    predicted_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert interval bounds and point predictions to matching floating-point arrays.

    Parameters
    ----------
    left_bounds : ndarray
        Left observation bounds, one per sample.
    right_bounds : ndarray
        Right observation bounds, using positive infinity for right censoring.
    predicted_times : ndarray
        Predicted event times, one per sample.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        Floating-point left bounds, right bounds, and predictions with identical shapes.
    """
    L = np.asarray(left_bounds, float)
    R = np.asarray(right_bounds, float)
    t_hat = np.asarray(predicted_times, float)

    assert L.shape == R.shape == t_hat.shape, "shape mismatch"
    return L, R, t_hat


def _compute_inside_mask(
    left: np.ndarray,
    right: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify point predictions inside their observation intervals.

    Parameters
    ----------
    left : ndarray, shape (n_samples,)
        Left observation bounds.
    right : ndarray, shape (n_samples,)
        Right observation bounds; use infinity for right censoring.
    predicted : ndarray, shape (n_samples,)
        Predicted event times.

    Returns
    -------
    inside : ndarray, (n_samples,)
        Boolean mask for predictions inside their intervals.
    is_right_cens : ndarray, (n_samples,)
        Boolean mask for right-censored intervals.

    Notes
    -----
    Finite intervals use (left, right]; right-censored intervals use (left, infinity).
    Exact events (left == right) use equality within the function's absolute tolerance.
    """
    is_right_cens = np.isinf(right)
    is_exact = (~is_right_cens) & (left == right)
    inside_exact = is_exact & np.isclose(predicted, right)
    inside_finite = (
        (~is_right_cens) & (~is_exact) & (predicted > left) & (predicted <= right)
    )
    inside_right = is_right_cens & (predicted > left)
    inside = inside_exact | inside_finite | inside_right
    return inside, is_right_cens


def inclusion_rate(
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    predicted_times: np.ndarray,
) -> float:
    """
    Inclusion rate for interval-censored data: proportion of predictions inside their intervals.

    This is similar to p_out but reports the complement (1 - p_out).
    p_out is proposed in [1]_.

    Parameters
    ----------
    left_bounds : ndarray, (n_samples,)
        Left limits of the interval-censored data.
    right_bounds : ndarray, (n_samples,)
        Right limits of the interval-censored data (use np.inf for right-censor).
    predicted_times : ndarray, (n_samples,)
        Your point predictions (e.g., median predicted times, mean predicted times, etc).

    Returns
    -------
    inclusion_rate : float
        Proportion of samples with predicted times inside their intervals.

    References
    ----------
    .. [1] Avati et al. (2020). "Countdown Regression: Sharp and Calibrated Survival Predictions".
       Proceedings of Machine Learning Research, 115, 145-155.
       https://proceedings.mlr.press/v115/avati20a.html
    """
    L, R, t_hat = _prepare_interval_arrays(left_bounds, right_bounds, predicted_times)
    inside, _ = _compute_inside_mask(L, R, t_hat)
    return float(np.mean(inside, dtype=float))


def mean_error_ic(
    left_bounds: np.ndarray,
    right_bounds: np.ndarray,
    predicted_times: np.ndarray,
    error_type: str = "absolute",
    log_scale: bool = False,
) -> float:
    """
    Calculate mean prediction error outside interval-censored observation bounds.

    The hinge loss for interval-censored targets follows [1]_; [2]_ uses this as the d_out metric.

    Parameters
    ----------
    left_bounds : ndarray, (n_samples,)
        Left limits of the interval-censored data.
    right_bounds : ndarray, (n_samples,)
        Right limits of the interval-censored data (use np.inf for right-censor).
    predicted_times : ndarray, (n_samples,)
        Your point predictions (e.g., median predicted times, mean predicted times, etc).
    error_type : str, default="absolute"
        Type of mean error to use. Options are "absolute" and "squared".
    log_scale : bool, default=False
        Whether to use log scale for the loss function.

    Returns
    -------
    mean_error : float
        Mean error value.

    References
    ----------
    .. [1] Shivaswamy, Chu and Jansche (2007). "A Support Vector Approach to Censored
       Targets". ICDM, 655-660.
       https://research.google/pubs/a-support-vector-approach-to-censored-targets/
    .. [2] Avati et al. (2020). "Countdown Regression: Sharp and Calibrated Survival Predictions".
       Proceedings of Machine Learning Research, 115, 145-155.
       https://proceedings.mlr.press/v115/avati20a.html
    """
    L, R, t_hat = _prepare_interval_arrays(left_bounds, right_bounds, predicted_times)

    # set the error function
    error_type = error_type.lower()
    if error_type == "absolute":
        error_func = np.abs
    elif error_type == "squared":
        error_func = np.square
    else:
        raise TypeError(
            "Please enter one of 'absolute' or 'squared' for calculating error."
        )

    # TODO: We need to move the logarithm transformation later after we calculate a best-guess.
    if log_scale:
        t_hat = np.log(t_hat)
        L = np.log(L)
        R = np.log(R)

    inside, is_right_cens = _compute_inside_mask(L, R, t_hat)
    outside = ~inside
    errors = np.zeros(t_hat.shape)
    error_to_right = np.full(np.count_nonzero(outside), np.inf)
    np.subtract(
        t_hat[outside],
        R[outside],
        out=error_to_right,
        where=~is_right_cens[outside],
    )
    distances = np.minimum(np.abs(t_hat[outside] - L[outside]), np.abs(error_to_right))
    errors[outside] = error_func(distances)
    return float(np.mean(errors, dtype=float))
