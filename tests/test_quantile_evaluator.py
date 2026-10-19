import numpy as np
import pytest

from SurvivalEVAL import QuantileRegEvaluator


@pytest.fixture
def quantile_evaluator() -> QuantileRegEvaluator:
    """Build two piecewise linear survival curves with medians two and four."""
    return QuantileRegEvaluator(
        pred_regs=[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
        quantile_levels=[0.25, 0.5, 0.75],
        event_times=[1.0, 2.0],
        event_indicators=[1, 1],
    )


def test_quantile_predictions_and_scores(
    quantile_evaluator: QuantileRegEvaluator,
) -> None:
    """Check quantile conversion and inherited metrics against known values."""
    evaluator = quantile_evaluator
    np.testing.assert_allclose(evaluator.predicted_event_times, [2.0, 4.0])
    np.testing.assert_allclose(
        evaluator.predict_probability_from_curve(2.0), [0.5, 0.75]
    )
    np.testing.assert_allclose(
        evaluator.predict_probability_from_curve(np.array([1.5, 3.0])), [0.625, 0.625]
    )
    np.testing.assert_allclose(
        evaluator.predict_multi_probabilities_from_curve(np.array([1.0, 2.0, 3.0])),
        [[0.75, 0.5, 0.25], [0.875, 0.75, 0.625]],
    )
    np.testing.assert_allclose(
        evaluator.predict_interval(cov_level=0.5), [[1, 3], [2, 6]]
    )
    np.testing.assert_allclose(evaluator.concordance(), (1, 1, 1))
    np.testing.assert_allclose(
        evaluator.residuals(method="CoxSnell"), -np.log([0.75, 0.75])
    )
    assert evaluator.brier_score(target_time=2.0) == pytest.approx(13 / 32)
    assert evaluator.mae() == pytest.approx(1.5)
    assert evaluator.mse() == pytest.approx(2.5)
    assert evaluator.rmse() == pytest.approx(np.sqrt(2.5))


def test_quantile_km_calibration(quantile_evaluator: QuantileRegEvaluator) -> None:
    """Check calibration with different time grids for the two quantile curves."""
    # At event times [1, 2], mean predictions are [13/16, 5/8] and KM is [1/2, 0].
    # Trapezoidal integration, including the origin, gives 75/256 over [0, 2].
    assert quantile_evaluator.km_calibration() == pytest.approx(75 / 512)


def test_quantile_evaluator_accepts_list_quantile_levels() -> None:
    """Preserve non-unit survival probabilities for events predicted at zero."""
    evaluator = QuantileRegEvaluator(
        pred_regs=np.array([[0.0, 2.0], [0.0, 3.0]]),
        quantile_levels=[0.25, 0.5],
        event_times=np.array([1.0, 2.0]),
        event_indicators=np.array([1, 1]),
    )
    np.testing.assert_allclose(evaluator.pred_survs, [0.75, 0.5])
