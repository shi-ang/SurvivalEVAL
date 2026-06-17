import numpy as np
import pytest

from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.Evaluations._concordance_utils import _ConcordanceCounts
from SurvivalEVAL.Evaluations.TimeDependentConcordance import (
    _time_dependent_risk_counts,
    concordance_time_dependent,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import KaplanMeier


def _add_time_dependent_pair(
    counts,
    risk_scores,
    anchor_col_by_sample,
    anchor,
    candidate,
    weight=1.0,
    tied_tol=1e-8,
):
    anchor_col = anchor_col_by_sample[anchor]
    risk_diff = risk_scores[candidate, anchor_col] - risk_scores[anchor, anchor_col]
    if abs(risk_diff) <= tied_tol:
        counts.risk_tie_pairs += weight
    elif risk_diff < 0:
        counts.concordant += weight
    else:
        counts.discordant += weight


def _before_tau(time, tau):
    return tau is None or time < tau


def _brute_time_dependent_counts(
    event_indicators,
    event_times,
    risk_scores,
    sample_weights=None,
    anchor_pair_weights=None,
    tau=None,
):
    if sample_weights is None:
        sample_weights = np.ones(event_times.shape[0], dtype=float)

    event_indicators = event_indicators.astype(bool)
    anchor_indices = np.flatnonzero(event_indicators)
    anchor_col_by_sample = np.full(event_times.shape[0], -1, dtype=int)
    anchor_col_by_sample[anchor_indices] = np.arange(anchor_indices.shape[0])

    counts = _ConcordanceCounts()
    for i in range(event_times.shape[0]):
        for j in range(i + 1, event_times.shape[0]):
            if (
                event_indicators[i]
                and event_indicators[j]
                and event_times[i] == event_times[j]
                and _before_tau(event_times[i], tau)
            ):
                counts.time_tie_pairs += sample_weights[i] * sample_weights[j]

    for i in range(event_times.shape[0]):
        if not event_indicators[i] or not _before_tau(event_times[i], tau):
            continue

        for j in range(event_times.shape[0]):
            if i == j:
                continue
            is_later = event_times[j] > event_times[i]
            is_same_time_censored = (
                event_times[j] == event_times[i] and not event_indicators[j]
            )
            if not (is_later or is_same_time_censored):
                continue

            if anchor_pair_weights is None:
                weight = sample_weights[i] * sample_weights[j]
            else:
                weight = anchor_pair_weights[i]
            _add_time_dependent_pair(
                counts, risk_scores, anchor_col_by_sample, i, j, weight=weight
            )

    return counts


def test_time_dependent_antolini_perfect_and_reversed_rankings():
    event_times = np.array([1.0, 2.0, 3.0])
    event_indicators = np.array([1, 1, 1])
    perfect_scores = np.array(
        [
            [3.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    reversed_scores = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 0.0],
        ]
    )

    perfect = concordance_time_dependent(perfect_scores, event_times, event_indicators)
    reversed_result = concordance_time_dependent(
        reversed_scores, event_times, event_indicators
    )

    np.testing.assert_allclose(perfect, (1.0, 3.0, 3.0))
    np.testing.assert_allclose(reversed_result, (0.0, 0.0, 3.0))


def test_time_dependent_antolini_risk_ties_use_existing_tie_policy():
    risk_scores = np.ones((3, 3))
    event_times = np.array([1.0, 2.0, 3.0])
    event_indicators = np.array([1, 1, 1])

    with_risk_ties = concordance_time_dependent(
        risk_scores, event_times, event_indicators, ties="Risk"
    )
    without_ties = concordance_time_dependent(
        risk_scores, event_times, event_indicators, ties="None"
    )

    np.testing.assert_allclose(with_risk_ties, (0.5, 1.5, 3.0))
    assert np.isnan(without_ties[0])
    assert without_ties[1:] == (0.0, 0.0)


def test_time_dependent_antolini_time_ties_are_not_comparable_risk_pairs():
    risk_scores = np.array(
        [
            [3.0, 3.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    event_times = np.array([1.0, 1.0, 2.0])
    event_indicators = np.array([1, 1, 1])

    risk_only = concordance_time_dependent(
        risk_scores, event_times, event_indicators, ties="Risk"
    )
    with_time_ties = concordance_time_dependent(
        risk_scores, event_times, event_indicators, ties="Time"
    )

    np.testing.assert_allclose(risk_only, (1.0, 2.0, 2.0))
    np.testing.assert_allclose(with_time_ties, (5.0 / 6.0, 2.5, 3.0))


def test_private_time_dependent_counts_match_brute_force_for_random_small_inputs():
    rng = np.random.default_rng(2)

    for n_samples in range(2, 10):
        for _ in range(50):
            event_times = rng.integers(1, 6, size=n_samples).astype(float)
            event_indicators = rng.random(n_samples) < 0.65
            if not np.any(event_indicators):
                event_indicators[rng.integers(0, n_samples)] = True
            risk_scores = rng.integers(
                -2, 3, size=(n_samples, int(event_indicators.sum()))
            ).astype(float)
            sample_weights = rng.uniform(0.25, 2.0, size=n_samples)
            anchor_pair_weights = rng.uniform(0.25, 2.0, size=n_samples)
            tau = float(rng.integers(1, 6))

            actual = _time_dependent_risk_counts(
                risk_scores,
                event_times,
                event_indicators,
                sample_weights=sample_weights,
                anchor_pair_weights=anchor_pair_weights,
                tau=tau,
            )
            expected = _brute_time_dependent_counts(
                event_indicators,
                event_times,
                risk_scores,
                sample_weights=sample_weights,
                anchor_pair_weights=anchor_pair_weights,
                tau=tau,
            )

            assert np.isclose(actual.concordant, expected.concordant)
            assert np.isclose(actual.discordant, expected.discordant)
            assert np.isclose(actual.risk_tie_pairs, expected.risk_tie_pairs)
            assert np.isclose(actual.time_tie_pairs, expected.time_tie_pairs)


def test_time_dependent_ipcw_uses_squared_anchor_weights():
    event_times = np.array([1.0, 2.0, 3.0, 4.0])
    event_indicators = np.array([1, 1, 1, 1])
    risk_scores = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [2.0, 3.0, 4.0, 0.0],
            [1.0, 2.0, 3.0, 0.0],
        ]
    )
    train_event_times = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    train_event_indicators = np.array([1, 0, 1, 0, 1])

    c_index, concordant, total = concordance_time_dependent(
        risk_scores,
        event_times,
        event_indicators,
        train_event_times=train_event_times,
        train_event_indicators=train_event_indicators,
        method="IPCW",
    )

    censoring_model = KaplanMeier(
        train_event_times, ~train_event_indicators.astype(bool)
    )
    censoring_survival = censoring_model.predict(event_times)
    anchor_weights = 1 / np.square(censoring_survival)
    expected_total = 3 * anchor_weights[0] + 2 * anchor_weights[1] + anchor_weights[2]

    assert np.isclose(c_index, 1.0)
    assert np.isclose(concordant, expected_total)
    assert np.isclose(total, expected_total)


def test_time_dependent_ipcw_tau_excludes_boundary_anchor_and_allows_later_candidates():
    event_times = np.array([1.0, 2.0, 4.0])
    event_indicators = np.array([1, 1, 1])
    risk_scores = np.array(
        [
            [3.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )

    result = concordance_time_dependent(
        risk_scores,
        event_times,
        event_indicators,
        train_event_times=np.array([1.0, 2.0, 4.0]),
        train_event_indicators=np.array([1, 1, 1]),
        method="IPCW",
        tau=2.0,
    )

    np.testing.assert_allclose(result, (1.0, 2.0, 2.0))


def test_time_dependent_ipcw_ignores_zero_censoring_survival_for_discarded_final_time_ties():
    event_times = np.array([1.0, 2.0, 2.0])
    event_indicators = np.array([1, 1, 1])
    risk_scores = np.array(
        [
            [3.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    kwargs = {
        "risk_scores": risk_scores,
        "event_times": event_times,
        "event_indicators": event_indicators,
        "train_event_times": np.array([1.0, 2.0]),
        "train_event_indicators": np.array([1, 0]),
        "method": "IPCW",
    }

    default_ties = concordance_time_dependent(**kwargs)
    no_ties = concordance_time_dependent(**kwargs, ties="None")

    np.testing.assert_allclose(default_ties, (1.0, 2.0, 2.0))
    np.testing.assert_allclose(no_ties, (1.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="Censoring survival probability is zero"):
        concordance_time_dependent(**kwargs, ties="Time")
    with pytest.raises(ValueError, match="Censoring survival probability is zero"):
        concordance_time_dependent(**kwargs, ties="All")


def test_time_dependent_concordance_validates_inputs():
    event_times = np.array([1.0, 2.0, 3.0])
    event_indicators = np.array([1, 1, 1])
    risk_scores = np.ones((3, 3))

    with pytest.raises(ValueError, match="2D array"):
        concordance_time_dependent(np.ones(3), event_times, event_indicators)
    with pytest.raises(ValueError, match="same"):
        concordance_time_dependent(np.ones((2, 3)), event_times, event_indicators)
    with pytest.raises(ValueError, match="observed events"):
        concordance_time_dependent(np.ones((3, 2)), event_times, event_indicators)
    with pytest.raises(ValueError, match="no observed events"):
        concordance_time_dependent(np.ones((3, 0)), event_times, np.array([0, 0, 0]))
    with pytest.raises(ValueError, match="Unsupported method"):
        concordance_time_dependent(
            risk_scores, event_times, event_indicators, method="Harrell"
        )
    with pytest.raises(ValueError, match="must be provided"):
        concordance_time_dependent(
            risk_scores, event_times, event_indicators, method="IPCW"
        )


def test_survival_evaluator_time_dependent_concordance_end_to_end():
    time_grid = np.array([0.0, 1.0, 2.0, 3.0])
    hazards = np.array([0.5, 0.2, 0.1])
    pred_survs = np.exp(-hazards[:, None] * time_grid)

    fully_observed = SurvivalEvaluator(
        pred_survs=pred_survs,
        time_coordinates=time_grid,
        event_times=np.array([1.0, 2.0, 3.0]),
        event_indicators=np.array([1, 1, 1]),
    )

    survival_result = fully_observed.concordance_time_dependent(
        method="Antolini", risks="Survival"
    )
    hazard_result = fully_observed.concordance_time_dependent(
        method="Antolini", risks="Hazard"
    )

    np.testing.assert_allclose(survival_result, (1.0, 3.0, 3.0))
    np.testing.assert_allclose(hazard_result, (1.0, 3.0, 3.0))

    censored = SurvivalEvaluator(
        pred_survs=pred_survs,
        time_coordinates=time_grid,
        event_times=np.array([1.0, 2.0, 3.0]),
        event_indicators=np.array([1, 0, 1]),
        train_event_times=np.array([1.0, 2.0, 3.0]),
        train_event_indicators=np.array([1, 1, 1]),
    )

    ipcw_survival = censored.concordance_time_dependent(
        method="IPCW", risks="Survival", tau=2.0
    )
    ipcw_hazard = censored.concordance_time_dependent(
        method="IPCW", risks="Hazard", tau=2.0
    )

    np.testing.assert_allclose(ipcw_survival, (1.0, 2.0, 2.0))
    np.testing.assert_allclose(ipcw_hazard, (1.0, 2.0, 2.0))
