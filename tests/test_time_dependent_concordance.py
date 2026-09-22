import tracemalloc

import numpy as np
import pytest

import SurvivalEVAL.Evaluations.TimeDependentConcordance as td_concordance
from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.Evaluations._concordance_utils import (
    _ConcordanceCounts,
    _finalize_counts,
)
from SurvivalEVAL.Evaluations.TimeDependentConcordance import (
    _select_risk_anchors,
    _time_dependent_risk_counts,
    _time_dependent_risk_counts_from_predictions,
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


def _reference_risk_anchor_mask(event_times, event_indicators, tau):
    event_indicators = event_indicators.astype(bool)
    anchor_times = event_times[event_indicators]
    included_by_sample = np.zeros(event_times.shape[0], dtype=bool)
    for anchor_time in np.unique(anchor_times):
        if tau is not None and anchor_time >= tau:
            continue
        same_time = event_times == anchor_time
        has_candidate = np.any(event_times > anchor_time) or np.any(
            same_time & ~event_indicators
        )
        if has_candidate:
            included_by_sample[same_time & event_indicators] = True
    return anchor_times, included_by_sample[event_indicators]


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


def test_survival_evaluator_hazard_concordance_tau_skips_excluded_late_anchors():
    time_grid = np.array([0.0, 1.0, 2.0, 3.0])
    hazards = np.array([0.5, 0.3, 0.1, 0.05])
    pred_survs = np.exp(-hazards[:, None] * time_grid)

    evaluator = SurvivalEvaluator(
        pred_survs=pred_survs,
        time_coordinates=time_grid,
        event_times=np.array([1.0, 2.0, 10.0, 12.0]),
        event_indicators=np.array([1, 1, 1, 0]),
        train_event_times=np.array([1.0, 2.0, 3.0, 4.0]),
        train_event_indicators=np.array([1, 1, 1, 1]),
    )

    result = evaluator.concordance_time_dependent(
        method="IPCW", risks="Hazard", tau=3.0
    )

    np.testing.assert_allclose(result, (1.0, 5.0, 5.0))


def test_survival_evaluator_hazard_concordance_skips_final_anchor_without_candidates():
    time_grid = np.array([0.0, 1.0, 2.0, 3.0])
    hazards = np.array([0.5, 0.3, 0.1])
    pred_survs = np.exp(-hazards[:, None] * time_grid)

    evaluator = SurvivalEvaluator(
        pred_survs=pred_survs,
        time_coordinates=time_grid,
        event_times=np.array([1.0, 2.0, 10.0]),
        event_indicators=np.array([1, 1, 1]),
    )

    result = evaluator.concordance_time_dependent(method="Antolini", risks="Hazard")

    np.testing.assert_allclose(result, (1.0, 3.0, 3.0))


def test_select_risk_anchors_matches_reference_on_randomized_inputs():
    rng = np.random.default_rng(20260812)
    for _ in range(100):
        n_samples = int(rng.integers(1, 30))
        event_times = rng.integers(1, 8, size=n_samples).astype(float)
        event_indicators = rng.integers(0, 2, size=n_samples).astype(bool)
        event_indicators[int(rng.integers(0, n_samples))] = True
        tau = None if rng.random() < 0.5 else float(rng.integers(1, 9))

        expected_times, expected_mask = _reference_risk_anchor_mask(
            event_times, event_indicators, tau
        )
        actual_times, actual_mask = _select_risk_anchors(
            event_times, event_indicators, tau
        )

        np.testing.assert_array_equal(actual_times, expected_times)
        np.testing.assert_array_equal(actual_mask, expected_mask)


@pytest.mark.parametrize("risks", ["Survival", "Hazard"])
def test_evaluator_predicts_each_sample_once_at_unique_contributing_times(
    monkeypatch, risks
):
    time_grid = np.array([0.0, 1.0, 2.0, 3.0])
    hazards = np.array([0.5, 0.4, 0.2, 0.1])
    evaluator = SurvivalEvaluator(
        pred_survs=np.exp(-hazards[:, None] * time_grid),
        time_coordinates=time_grid,
        event_times=np.array([1.0, 1.0, 2.0, 3.0]),
        event_indicators=np.ones(4),
    )

    original_predict = evaluator._predict_risks_from_curve
    predicted_at = []

    def recording_predict(sample_index, target_times, risks):
        predicted_at.append((sample_index, target_times.copy()))
        return original_predict(sample_index, target_times, risks)

    monkeypatch.setattr(evaluator, "_predict_risks_from_curve", recording_predict)

    result = evaluator.concordance_time_dependent(risks=risks)

    assert len(predicted_at) == 4
    for (sample_index, target_times), expected_index, expected_times in zip(
        predicted_at, range(4), [[1.0], [1.0], [1.0, 2.0], [1.0, 2.0]]
    ):
        assert sample_index == expected_index
        np.testing.assert_array_equal(target_times, expected_times)
    np.testing.assert_allclose(result, (1.0, 5.0, 5.0))


@pytest.mark.parametrize("weighting", ["unweighted", "symmetric", "anchor"])
def test_streamed_counts_match_brute_force_on_randomized_inputs(weighting):
    rng = np.random.default_rng(20260908)
    time_grid = np.arange(1.0, 8.0)
    for n_samples in range(1, 20):
        for _ in range(20):
            event_times = rng.choice(time_grid, size=n_samples)
            event_indicators = rng.random(n_samples) < 0.65
            scores = rng.integers(-2, 3, size=(n_samples, time_grid.size)).astype(float)
            scores += rng.choice([0.0, 0.5e-8, 1e-8, 1.5e-8], size=scores.shape)
            sample_weights = (
                None if weighting == "unweighted" else rng.uniform(0.25, 2, n_samples)
            )
            anchor_pair_weights = (
                rng.uniform(0.25, 2, n_samples) if weighting == "anchor" else None
            )
            tau = None if rng.random() < 0.5 else float(rng.integers(1, 9))
            kwargs = dict(
                sample_weights=sample_weights,
                anchor_pair_weights=anchor_pair_weights,
                tau=tau,
            )

            def predict_risks(sample_index, target_times):
                return scores[sample_index, np.searchsorted(time_grid, target_times)]

            actual = _time_dependent_risk_counts_from_predictions(
                predict_risks, event_times, event_indicators, **kwargs
            )
            dense_scores = scores[
                :, np.searchsorted(time_grid, event_times[event_indicators])
            ]
            expected = _brute_time_dependent_counts(
                event_indicators, event_times, dense_scores, **kwargs
            )
            np.testing.assert_allclose(
                [
                    actual.concordant,
                    actual.discordant,
                    actual.risk_tie_pairs,
                    actual.time_tie_pairs,
                ],
                [
                    expected.concordant,
                    expected.discordant,
                    expected.risk_tie_pairs,
                    expected.time_tie_pairs,
                ],
                atol=1e-12,
            )


@pytest.mark.parametrize("weighting", ["unweighted", "symmetric", "anchor"])
@pytest.mark.parametrize("center", [-0.5, 0.0, 0.5])
def test_streamed_counts_preserve_subtraction_at_risk_tie_boundaries(weighting, center):
    tolerance = 1e-8
    lower, upper = center - tolerance, center + tolerance
    anchor_risks = np.repeat(
        [
            np.nextafter(lower, -np.inf),
            lower,
            np.nextafter(lower, np.inf),
            center,
            np.nextafter(upper, -np.inf),
            upper,
            np.nextafter(upper, np.inf),
        ],
        2,
    )
    candidate_risks = [
        center,
        np.nextafter(center, -np.inf),
        np.nextafter(center, np.inf),
    ]
    scores = np.concatenate((anchor_risks, candidate_risks))
    event_times = np.concatenate((np.ones(anchor_risks.size), [1.0, 2.0, 2.0]))
    event_indicators = np.arange(scores.size) < anchor_risks.size
    weights = (np.arange(scores.size) % 4).astype(float)
    kwargs = dict(
        sample_weights=None if weighting == "unweighted" else weights,
        anchor_pair_weights=weights[::-1] if weighting == "anchor" else None,
    )

    def predict_risks(sample_index, target_times):
        return np.full(target_times.size, scores[sample_index])

    actual = _time_dependent_risk_counts_from_predictions(
        predict_risks, event_times, event_indicators, **kwargs
    )
    expected = _brute_time_dependent_counts(
        event_indicators,
        event_times,
        np.repeat(scores[:, None], anchor_risks.size, axis=1),
        **kwargs,
    )

    # candidate - anchor must be compared directly: searching candidate +/-
    # tolerance can move a boundary by one ULP and misclassify these pairs.
    assert actual == expected


@pytest.mark.parametrize("weighting", ["unweighted", "symmetric", "anchor"])
@pytest.mark.parametrize("tau", [None, 2.0])
def test_streamed_nonfinite_risks_preserve_dense_pair_classification(weighting, tau):
    risk_values = np.array(
        [-np.inf, np.nan, -np.inf, -1.0, 0.0, np.inf, np.nan, np.inf]
    )
    scores = np.column_stack((np.tile(risk_values, 4), np.tile(risk_values[::-1], 4)))
    event_times = np.repeat([1.0, 1.0, 2.0, 3.0], risk_values.size)
    event_indicators = np.repeat([True, False, True, False], risk_values.size)
    weights = (np.arange(event_times.size) % 5).astype(float)
    kwargs = dict(
        sample_weights=None if weighting == "unweighted" else weights,
        anchor_pair_weights=weights[::-1] if weighting == "anchor" else None,
        tau=tau,
    )

    def predict_risks(sample_index, target_times):
        return scores[sample_index, target_times.astype(int) - 1]

    # Equal infinities and NaN risks produce NaN differences, which the
    # existing dense counter classifies as discordant, not tied or concordant.
    # Include both signs, same-time censorings, and later samples.
    with np.errstate(invalid="ignore"):
        actual = _time_dependent_risk_counts_from_predictions(
            predict_risks, event_times, event_indicators, **kwargs
        )
        dense_scores = scores[:, event_times[event_indicators].astype(int) - 1]
        expected = _brute_time_dependent_counts(
            event_indicators, event_times, dense_scores, **kwargs
        )
        dense = _time_dependent_risk_counts(
            dense_scores, event_times, event_indicators, **kwargs
        )

    assert actual == dense == expected


@pytest.mark.parametrize("weighting", ["symmetric", "anchor"])
def test_streamed_group_weights_are_independent_including_zero_weight_groups(weighting):
    event_times = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0])
    event_indicators = np.array([True] * 7 + [False])
    weights = np.array([1e16, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 1.0])
    scores = np.array(
        [[0.0, 0.0, 0.0]] * 2 + [[0.0, 1.0, 0.0]] * 3 + [[0.0, 0.0, 0.0]] * 3
    )
    kwargs = dict(
        sample_weights=weights,
        anchor_pair_weights=weights if weighting == "anchor" else None,
    )

    def predict_risks(sample_index, target_times):
        return scores[sample_index, target_times.astype(int) - 1]

    actual = _time_dependent_risk_counts_from_predictions(
        predict_risks, event_times, event_indicators, **kwargs
    )
    expected = _brute_time_dependent_counts(
        event_indicators,
        event_times,
        scores[:, event_times[event_indicators].astype(int) - 1],
        **kwargs,
    )

    # A global prefix sum would lose small weights after the first 1e16
    # group. The all-zero final group must not contribute directed pairs.
    assert actual.concordant == (12.0 if weighting == "anchor" else 4.0)
    assert actual == expected


@pytest.mark.parametrize("weighting", ["symmetric", "anchor"])
def test_streamed_small_concordant_weight_survives_large_risk_tie_weight(weighting):
    event_times = np.array([1.0, 1.0, 2.0])
    event_indicators = np.array([True, True, False])
    weights = np.array([1e16, 1.0, 1.0])
    scores = np.array([1.0, 2.0, 1.0])
    kwargs = dict(
        sample_weights=weights,
        anchor_pair_weights=weights if weighting == "anchor" else None,
    )

    def predict_risks(sample_index, target_times):
        return np.full(target_times.size, scores[sample_index])

    actual = _time_dependent_risk_counts_from_predictions(
        predict_risks, event_times, event_indicators, **kwargs
    )
    expected = _time_dependent_risk_counts(
        np.repeat(scores[:, None], 2, axis=1),
        event_times,
        event_indicators,
        **kwargs,
    )

    # Subtracting a 1e16 prefix from the group's total rounds the remaining
    # concordant weight to zero and makes the no-ties C-index undefined.
    assert actual == expected
    assert actual.concordant == 1.0
    assert actual.risk_tie_pairs == 1e16
    np.testing.assert_array_equal(_finalize_counts(actual, "None"), [1.0, 1.0, 1.0])


def test_streamed_rank_queries_scale_with_time_groups_not_event_anchors(monkeypatch):
    group_sizes = [512, 1024, 512]
    event_times = np.repeat([1.0, 2.0, 3.0], group_sizes)
    event_indicators = event_times < 3.0
    original_ranks = td_concordance._grouped_risk_ranks
    query_sizes = []

    def recording_ranks(anchor_risks, starts, ends, candidate_risks, tied_tol):
        query_sizes.append(candidate_risks.shape)
        ranks = original_ranks(anchor_risks, starts, ends, candidate_risks, tied_tol)
        assert ranks.shape == (2, *candidate_risks.shape)
        return ranks

    def predict_risks(sample_index, target_times):
        return np.full(target_times.size, -event_times[sample_index])

    monkeypatch.setattr(td_concordance, "_grouped_risk_ranks", recording_ranks)
    actual = _time_dependent_risk_counts_from_predictions(
        predict_risks, event_times, event_indicators
    )

    expected_pairs = (
        group_sizes[0] * sum(group_sizes[1:]) + group_sizes[1] * group_sizes[2]
    )
    assert actual.concordant == expected_pairs
    assert actual.discordant == actual.risk_tie_pairs == 0.0
    assert query_sizes
    assert max(columns for _, columns in query_sizes) <= 2
    assert max(rows * columns for rows, columns in query_sizes) <= event_times.size
    assert sum(rows * columns for rows, columns in query_sizes) <= 2 * event_times.size


def test_dense_scores_retain_distinct_columns_for_events_at_the_same_time():
    # The public dense-array API accepts different columns even for tied
    # event times; only a prediction callback guarantees a shared time risk.
    scores = np.array([[3.0, 0.0, 0.0], [3.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    result = concordance_time_dependent(
        scores, np.array([1.0, 1.0, 2.0]), np.ones(3), ties="Risk"
    )
    np.testing.assert_array_equal(result, [0.5, 1.0, 2.0])


@pytest.mark.parametrize("risks", ["Survival", "Hazard"])
@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
@pytest.mark.parametrize("curve_layout", ["shared_grid", "shared_curve", "individual"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_streamed_evaluator_matches_dense_for_crossing_curves(
    risks, interpolation, curve_layout, dtype
):
    rng = np.random.default_rng(16)
    n_samples = 14
    time_grid = np.linspace(0.0, 10.0, 12).astype(dtype)
    increments = rng.uniform(0.01, 0.4, (n_samples, time_grid.size - 1))
    curves = np.column_stack((np.ones(n_samples), np.exp(-increments.cumsum(axis=1))))
    curves = curves.astype(dtype)
    if curve_layout != "shared_grid":
        time_grid = time_grid * rng.uniform(1.0, 1.5, (n_samples, 1)).astype(dtype)
    if curve_layout == "shared_curve":
        curves = curves[0]

    event_times = np.array([4, 1, 4, 2, 5, 2, 8, 6, 7, 9, 2, 4, 9, 1], dtype=dtype)
    event_indicators = np.array([1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1])
    train_times = np.arange(1.0, 13.0)
    train_indicators = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    evaluator = SurvivalEvaluator(
        curves,
        time_grid,
        event_times,
        event_indicators,
        train_times,
        train_indicators,
        interpolation=interpolation,
    )
    anchor_times = event_times[event_indicators.astype(bool)]
    if risks == "Survival":
        dense_scores = -evaluator.predict_multi_probabilities_from_curve(anchor_times)
    else:
        dense_scores = evaluator.predict_multi_hazards_from_curve(anchor_times)
    # The original evaluator expanded its predictions into a float64 matrix.
    dense_scores = dense_scores.astype(float)

    for method in ["Antolini", "Naive", "IPCW"]:
        for ties in ["None", "Risk", "Time", "All"]:
            for tau in [None, 4.0]:
                expected = concordance_time_dependent(
                    dense_scores,
                    event_times,
                    event_indicators,
                    train_times,
                    train_indicators,
                    method=method,
                    ties=ties,
                    tau=tau,
                )
                actual = evaluator.concordance_time_dependent(
                    method=method, risks=risks, ties=ties, tau=tau
                )
                np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("interpolation", ["Linear", "Pchip"])
def test_streamed_survival_preserves_float32_prediction_ties(interpolation):
    half = np.float32(0.5)
    next_half = np.nextafter(half, np.float32(1.0))
    evaluator = SurvivalEvaluator(
        pred_survs=np.array([[1.0, half], [1.0, next_half]], dtype=np.float32),
        time_coordinates=np.array([0.0, 1.0]),
        event_times=np.array([0.5, 1.0]),
        event_indicators=np.ones(2),
        interpolation=interpolation,
    )
    # Unrounded interpolated risks differ by more than 1e-8, but both round
    # to 0.75 in the original float32 probability matrix.
    result = evaluator.concordance_time_dependent()
    np.testing.assert_allclose(result, (0.5, 0.5, 1.0))


@pytest.mark.parametrize("risks", ["Survival", "Hazard"])
@pytest.mark.parametrize("tau", [None, 500.0])
def test_streamed_evaluator_uses_linear_working_memory(risks, tau):
    n_samples = 1_000
    time_grid = np.array([0.0, n_samples])
    evaluator = SurvivalEvaluator(
        pred_survs=np.tile([1.0, 0.5], (n_samples, 1)),
        time_coordinates=time_grid,
        event_times=np.arange(1.0, n_samples + 1),
        event_indicators=np.ones(n_samples),
    )

    # Exclude the input curves. A float64 n-by-n matrix alone would require
    # 8 MB here; allow a generous budget for linear buffers and interpolation.
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    baseline_bytes, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        result = evaluator.concordance_time_dependent(risks=risks, tau=tau)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert peak_bytes - baseline_bytes < 1024 * n_samples
    assert result[0] == 0.5


@pytest.mark.parametrize("ties", ["None", "Risk", "Time", "All"])
def test_streamed_evaluator_counts_final_event_ties_without_predictions(
    monkeypatch, ties
):
    evaluator = SurvivalEvaluator(
        pred_survs=np.ones((3, 1)),
        time_coordinates=np.array([0.0]),
        event_times=np.ones(3),
        event_indicators=np.ones(3),
    )

    def unexpected_prediction(*args, **kwargs):
        pytest.fail("Final event-only ties do not require risk predictions.")

    monkeypatch.setattr(evaluator, "_predict_risks_from_curve", unexpected_prediction)
    actual = evaluator.concordance_time_dependent(risks="Hazard", ties=ties)
    expected = concordance_time_dependent(
        np.zeros((3, 3)), np.ones(3), np.ones(3), ties=ties
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_streamed_hazards_only_require_grid_support_for_contributing_pairs():
    evaluator = SurvivalEvaluator(
        pred_survs=np.exp(-np.array([[0.0, 1.0], [0.0, 0.8], [0.0, 0.4]])),
        time_coordinates=np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 2.0]]),
        event_times=np.array([1.0, 2.0, 3.0]),
        event_indicators=np.ones(3),
    )
    # Sample 0 is not a candidate for the event at time 2, so its hazard
    # beyond its own grid is irrelevant. The final event needs no own risk.
    result = evaluator.concordance_time_dependent(risks="Hazard")
    np.testing.assert_allclose(result, (1.0, 3.0, 3.0))
