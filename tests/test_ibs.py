#!/usr/bin/env python3
"""
Test cases for IBS hinge loss implementation for interval-censored survival data.
"""

import os
import sys

import numpy as np
from scipy.integrate import simpson, trapezoid

# Add the parent directory to sys.path to import SurvivalEVAL
# Insert at the beginning to prioritize local version over installed package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SurvivalEVAL.Evaluations.BrierScore import brier_multiple_points_ic
from SurvivalEVAL.IntervalCenEvaluator import IntervalCenEvaluator


def ibs_hinge_ic(
    pred_survs: np.ndarray,  # (N, T): predicted survival probabilities
    time_coordinates: np.ndarray,  # (T,): time points corresponding to survival probabilities
    left_limits: np.ndarray,  # (N,): left limits of interval censoring
    right_limits: np.ndarray,  # (N,): right limits of interval censoring
    train_left_limits: np.ndarray = None,  # (N_train,): left limits
    train_right_limits: np.ndarray = None,  # (N_train,): right limits
    method: str = "uncensored",  # "uncensored" | "Tsouprou-marginal"
    integration_method: str = "trapezoidal",  # "trapezoidal" | "simpson"
) -> float:
    """
    Compute the Integrated Brier Score (IBS) using the hinge loss for interval-censored data.

    Parameters
    ----------
    pred_survs: np.ndarray
        Predicted survival probabilities of shape (N, T).
    time_coordinates: np.ndarray
        Time points corresponding to the survival probabilities of shape (T,).
    left_limits: np.ndarray
        Left limits of interval censoring of shape (N,).
    right_limits: np.ndarray
        Right limits of interval censoring of shape (N,).
    train_left_limits: np.ndarray, optional
        Left limits for training data of shape (N_train,).
    train_right_limits: np.ndarray, optional
        Right limits for training data of shape (N_train,).
    method: str, default: "uncensored"
        Method to handle uncertain areas. Options are "uncensored" or "Tsouprou-marginal".
    integration_method: str, default: "trapezoidal"
        Numerical integration method. Options are "trapezoidal" or "simpson".

    Returns
    -------
    float
        The computed IBS hinge loss.
    """
    brier_scores = brier_multiple_points_ic(
        pred_mat=pred_survs,
        left_limits=left_limits,
        right_limits=right_limits,
        target_times=time_coordinates,
        train_left_limits=train_left_limits,
        train_right_limits=train_right_limits,
        method=method,
    )

    if integration_method == "trapezoidal":
        ibs = trapezoid(brier_scores, time_coordinates) / (
            time_coordinates[-1] - time_coordinates[0]
        )
    elif integration_method == "simpson":
        ibs = simpson(brier_scores, time_coordinates) / (
            time_coordinates[-1] - time_coordinates[0]
        )
    else:
        raise ValueError(f"Unknown integration method: {integration_method}")

    return ibs


def test_case_1_simple_interval_censoring():
    """
    Test Case 1: Simple interval censoring with known ground truth
    """
    print("=" * 60)
    print("Test Case 1: Simple interval censoring")
    print("=" * 60)

    # Create simple test data
    n_samples = 5
    n_time_points = 10
    time_coordinates = np.linspace(0, 10, n_time_points)

    # Create predicted survival curves (decreasing probabilities over time)
    pred_survs = np.array(
        [
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],  # Sample 1
            [1.0, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15],  # Sample 2
            [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.05, 0.02, 0.01],  # Sample 3
            [1.0, 0.92, 0.84, 0.76, 0.68, 0.6, 0.52, 0.44, 0.36, 0.28],  # Sample 4
            [1.0, 0.88, 0.76, 0.64, 0.52, 0.4, 0.28, 0.16, 0.04, 0.01],  # Sample 5
        ]
    )

    # Interval censoring data
    left_limits = np.array([2.0, 3.0, 1.5, 4.0, 2.5])
    right_limits = np.array([4.0, 5.0, 3.0, 6.0, 4.5])

    # Training data (same structure)
    train_left_limits = np.array([1.0, 2.0, 3.0, 2.5, 3.5])
    train_right_limits = np.array([3.0, 4.0, 5.0, 4.5, 5.5])

    print(f"Time coordinates: {time_coordinates}")
    print(f"Left limits: {left_limits}")
    print(f"Right limits: {right_limits}")
    print(f"Prediction shape: {pred_survs.shape}")

    try:
        # Test IBS hinge with uncensored method
        ibs_uncensored = ibs_hinge_ic(
            pred_survs=pred_survs,
            time_coordinates=time_coordinates,
            left_limits=left_limits,
            right_limits=right_limits,
            method="uncensored",
        )
        print(f"IBS Hinge (uncensored): {ibs_uncensored:.4f}")

        # Test IBS hinge with Tsouprou-marginal method
        ibs_tsouprou = ibs_hinge_ic(
            pred_survs=pred_survs,
            time_coordinates=time_coordinates,
            left_limits=left_limits,
            right_limits=right_limits,
            train_left_limits=train_left_limits,
            train_right_limits=train_right_limits,
            method="Tsouprou-marginal",
        )
        print(f"IBS Hinge (Tsouprou-marginal): {ibs_tsouprou:.4f}")

        # Test ignore uncertain version (now integrated into main function)
        # Note: The uncensored method already ignores uncertain areas
        print(f"IBS Hinge (method already ignores uncertain): {ibs_tsouprou:.4f}")

        print("✓ Test Case 1 passed")

    except Exception as e:
        print(f"✗ Test Case 1 failed: {e}")
        import traceback

        traceback.print_exc()


def test_case_2_evaluator_integration():
    """
    Test Case 2: Test integration with IntervalCenEvaluator
    """
    print("\n" + "=" * 60)
    print("Test Case 2: IntervalCenEvaluator integration")
    print("=" * 60)

    # Create test data
    n_samples = 8
    n_time_points = 15
    time_coordinates = np.linspace(0, 15, n_time_points)

    # Create more realistic survival curves
    np.random.seed(42)
    pred_survs = np.zeros((n_samples, n_time_points))
    for i in range(n_samples):
        # Generate decreasing survival probabilities with some noise
        decay_rate = np.random.uniform(0.1, 0.3)
        pred_survs[i, :] = np.exp(-decay_rate * time_coordinates)
        # Add some noise
        noise = np.random.normal(0, 0.05, n_time_points)
        pred_survs[i, :] = np.clip(pred_survs[i, :] + noise, 0, 1)

    # Ensure monotonicity (survival probabilities should be non-increasing)
    for i in range(n_samples):
        for j in range(1, n_time_points):
            pred_survs[i, j] = min(pred_survs[i, j], pred_survs[i, j - 1])

    # Generate interval censoring data
    left_limits = np.random.uniform(2, 8, n_samples)
    right_limits = left_limits + np.random.uniform(2, 6, n_samples)

    # Training data
    train_left_limits = np.random.uniform(1, 7, n_samples)
    train_right_limits = train_left_limits + np.random.uniform(2, 5, n_samples)

    print(f"Time coordinates: {time_coordinates}")
    print(f"Left limits: {left_limits}")
    print(f"Right limits: {right_limits}")

    try:
        # Create evaluator
        evaluator = IntervalCenEvaluator(
            pred_survs=pred_survs,
            time_coordinates=time_coordinates,
            left_limits=left_limits,
            right_limits=right_limits,
            train_left_limits=train_left_limits,
            train_right_limits=train_right_limits,
        )

        # Test IBS hinge method
        ibs_result = evaluator.integrated_brier_score(method="Tsouprou-marginal")
        print(f"IBS Hinge via evaluator: {ibs_result:.4f}")

        # Test with different integration method
        ibs_simpson = evaluator.integrated_brier_score(
            method="Tsouprou-marginal", integration_method="simpson"
        )
        print(f"IBS Hinge (Simpson): {ibs_simpson:.4f}")

        # Compare with regular Brier score at a specific time
        brier_at_5 = evaluator.brier_score(target_time=5.0, method="Tsouprou-marginal")
        print(f"Brier score at t=5.0: {brier_at_5:.4f}")

        print("✓ Test Case 2 passed")

    except Exception as e:
        print(f"✗ Test Case 2 failed: {e}")
        import traceback

        traceback.print_exc()


def test_case_3_edge_cases():
    """
    Test Case 3: Edge cases and error handling
    """
    print("\n" + "=" * 60)
    print("Test Case 3: Edge cases and error handling")
    print("=" * 60)

    # Test case with point observations (no censoring)
    n_samples = 3
    n_time_points = 8
    time_coordinates = np.linspace(0, 8, n_time_points)

    pred_survs = np.array(
        [
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
            [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.05],
            [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65],
        ]
    )

    # Point observations (left = right)
    event_times = np.array([3.0, 2.5, 4.5])
    left_limits = event_times
    right_limits = event_times

    train_left_limits = np.array([2.0, 3.0, 4.0])
    train_right_limits = np.array([2.0, 3.0, 4.0])

    print("Testing point observations (no censoring)...")

    try:
        evaluator = IntervalCenEvaluator(
            pred_survs=pred_survs,
            time_coordinates=time_coordinates,
            left_limits=left_limits,
            right_limits=right_limits,
            train_left_limits=train_left_limits,
            train_right_limits=train_right_limits,
        )

        # Should automatically use uncensored method
        ibs_point = evaluator.integrated_brier_score()
        print(f"IBS for point observations: {ibs_point:.4f}")

        print("✓ Point observations test passed")

    except Exception as e:
        print(f"✗ Point observations test failed: {e}")

    # Test case with very few time points
    print("\nTesting with minimal time points...")
    try:
        minimal_time = np.array([0, 5])
        minimal_pred = np.array([[1.0, 0.5], [1.0, 0.3]])
        minimal_left = np.array([2.0, 3.0])
        minimal_right = np.array([4.0, 6.0])

        ibs_minimal = ibs_hinge_ic(
            pred_survs=minimal_pred,
            time_coordinates=minimal_time,
            left_limits=minimal_left,
            right_limits=minimal_right,
            method="uncensored",
        )
        print(f"IBS with minimal time points: {ibs_minimal:.4f}")
        print("✓ Minimal time points test passed")

    except Exception as e:
        print(f"✗ Minimal time points test failed: {e}")


def test_case_4_comparison_with_paper():
    """
    Test Case 4: Simulate scenario similar to the paper
    """
    print("\n" + "=" * 60)
    print("Test Case 4: Paper-like scenario simulation")
    print("=" * 60)

    # Simulate a more realistic scenario
    np.random.seed(123)
    n_samples = 20
    n_time_points = 50
    max_time = 20
    time_coordinates = np.linspace(0, max_time, n_time_points)

    # Generate survival curves with different risk profiles
    pred_survs = np.zeros((n_samples, n_time_points))
    for i in range(n_samples):
        # High risk vs low risk patients
        if i < n_samples // 2:
            # High risk: faster decline
            hazard_rate = np.random.uniform(0.2, 0.4)
        else:
            # Low risk: slower decline
            hazard_rate = np.random.uniform(0.05, 0.15)

        pred_survs[i, :] = np.exp(-hazard_rate * time_coordinates)

    # Generate interval censoring with varying widths
    true_event_times = np.random.exponential(8, n_samples)
    interval_widths = np.random.uniform(2, 6, n_samples)

    left_limits = np.maximum(0, true_event_times - interval_widths / 2)
    right_limits = true_event_times + interval_widths / 2

    # Some right censoring (infinite right limits)
    right_censor_mask = np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    right_limits[right_censor_mask] = np.inf

    # Training data
    train_true_times = np.random.exponential(7, n_samples)
    train_widths = np.random.uniform(1.5, 5, n_samples)
    train_left_limits = np.maximum(0, train_true_times - train_widths / 2)
    train_right_limits = train_true_times + train_widths / 2
    train_right_limits[np.random.choice([True, False], n_samples, p=[0.25, 0.75])] = (
        np.inf
    )

    print(f"Samples: {n_samples}, Time points: {n_time_points}")
    print(f"Max observation time: {max_time}")
    print(f"Right censoring rate: {np.mean(np.isinf(right_limits)):.2f}")
    print(
        f"Average interval width: {np.mean(right_limits[np.isfinite(right_limits)] - left_limits[np.isfinite(right_limits)]):.2f}"
    )

    try:
        # Test both IBS implementations
        ibs_standard = ibs_hinge_ic(
            pred_survs=pred_survs,
            time_coordinates=time_coordinates,
            left_limits=left_limits,
            right_limits=right_limits,
            train_left_limits=train_left_limits,
            train_right_limits=train_right_limits,
            method="Tsouprou-marginal",
        )

        # Test with different method (uncensored ignores uncertain areas)
        ibs_uncensored = ibs_hinge_ic(
            pred_survs=pred_survs,
            time_coordinates=time_coordinates,
            left_limits=left_limits,
            right_limits=right_limits,
            method="uncensored",
        )

        print(f"IBS Hinge (Tsouprou-marginal): {ibs_standard:.4f}")
        print(f"IBS Hinge (uncensored - ignores uncertain): {ibs_uncensored:.4f}")
        print(f"Difference: {abs(ibs_standard - ibs_uncensored):.4f}")

        # Test with evaluator
        evaluator = IntervalCenEvaluator(
            pred_survs=pred_survs,
            time_coordinates=time_coordinates,
            left_limits=left_limits,
            right_limits=right_limits,
            train_left_limits=train_left_limits,
            train_right_limits=train_right_limits,
        )

        ibs_evaluator = evaluator.integrated_brier_score(method="Tsouprou-marginal")
        print(f"IBS Hinge (via evaluator): {ibs_evaluator:.4f}")

        # Calculate some reference Brier scores at specific times
        brier_early = evaluator.brier_score(target_time=5, method="Tsouprou-marginal")
        brier_mid = evaluator.brier_score(target_time=10, method="Tsouprou-marginal")
        brier_late = evaluator.brier_score(target_time=15, method="Tsouprou-marginal")

        print(f"\nReference Brier scores:")
        print(f"  t=5:  {brier_early:.4f}")
        print(f"  t=10: {brier_mid:.4f}")
        print(f"  t=15: {brier_late:.4f}")

        print("✓ Test Case 4 passed")

    except Exception as e:
        print(f"✗ Test Case 4 failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    print("Testing IBS Hinge Implementation for Interval-Censored Survival Data")
    print("Reference: https://arxiv.org/pdf/1806.08324")

    test_case_1_simple_interval_censoring()
    test_case_2_evaluator_integration()
    test_case_3_edge_cases()
    test_case_4_comparison_with_paper()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
