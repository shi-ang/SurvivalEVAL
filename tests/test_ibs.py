"""Numerical checks for integrated interval-censored Brier scores."""

import pytest

from SurvivalEVAL import IntervalCenEvaluator


@pytest.mark.parametrize(
    "integration_method, early_weight, late_weight",
    [("trapz", 1 / 2, 1 / 3), ("simpson", 3 / 4, 1 / 4)],
)
@pytest.mark.parametrize(
    "left, right, method, early_score, late_score",
    [
        ([0.5, 2.0], [1.5, 4.0], "uncensored", 1 / 16, 1 / 16),
        ([0.5, 2.0], [1.5, 4.0], "Tsouprou-marginal", 1 / 16, 1 / 32),
        ([1.0, 2.0], [1.0, 2.0], "uncensored", 5 / 32, 5 / 32),
    ],
)
def test_interval_ibs_matches_hand_calculated_integral(
    left: list[float],
    right: list[float],
    method: str,
    early_score: float,
    late_score: float,
    integration_method: str,
    early_weight: float,
    late_weight: float,
) -> None:
    """Integrate known scores on an irregular grid, including exact observations."""
    evaluator = IntervalCenEvaluator(
        pred_survs=[[1.0, 0.5, 0.25], [1.0, 0.75, 0.5]],
        time_coordinates=[0.0, 1.0, 3.0],
        left_limits=left,
        right_limits=right,
        train_left_limits=[1.0, 2.0, 3.0, 4.0],
        train_right_limits=[1.0, 2.0, 3.0, 4.0],
    )
    # Scores at [0, 1, 3] are [0, early_score, late_score]. Exclusion leaves
    # one known status at each positive time in the first case. With marginal
    # imputation, the fitted upper survival curve has no mass in (0.5, 1.5],
    # so that interval is excluded at t=1; the other has status 1/2 at t=3.
    # The weights above integrate these three scores and divide by the span 3.
    assert evaluator.integrated_brier_score(
        target_times=[0.0, 1.0, 3.0],
        method=method,
        integration_method=integration_method,
    ) == pytest.approx(early_weight * early_score + late_weight * late_score)
