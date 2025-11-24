import numpy as np
import pytest
from lifelines import AalenJohansenFitter

from SurvivalEVAL.NonparametricEstimator.CompetingRisks import AalenJohansenCompetingRisks


def _generate_data_no_cross_ties(n=200, random_state=12345):
    """
    Generate synthetic competing-risks data where:
        - 0 = censored, 1..K = causes
        - There are NO ties between different causes (>0),
          but ties ARE allowed within the same cause.

    Strategy:
        - Create a small discrete grid of base times.
        - Assign disjoint time grids to each cause (1 and 2).
        - Add some duplicate times within each cause to create same-cause ties.
    """
    rng = np.random.default_rng(random_state)

    # Two causes: 1 and 2
    base_times_c1 = np.array([1.0, 2.0, 3.0, 4.0])
    base_times_c2 = np.array([1.5, 2.5, 3.5, 4.5])

    # Sample indices with replacement to allow ties *within* each cause
    idx_c1 = rng.integers(0, len(base_times_c1), size=n // 3)
    idx_c2 = rng.integers(0, len(base_times_c2), size=n // 3)

    times_c1 = base_times_c1[idx_c1]
    times_c2 = base_times_c2[idx_c2]

    # Remaining subjects are censored; sample censoring times
    n_cens = n - times_c1.size - times_c2.size
    times_cens = rng.uniform(0.5, 5.0, size=n_cens)

    # Stack everything together
    times = np.concatenate([times_c1, times_c2, times_cens])
    events = np.concatenate([
        np.full(times_c1.shape, 1, dtype=int),  # cause 1
        np.full(times_c2.shape, 2, dtype=int),  # cause 2
        np.zeros(times_cens.shape, dtype=int),  # censored
    ])

    # Shuffle jointly
    order = rng.permutation(n)
    times = times[order]
    events = events[order]

    # Sanity check: no ties between different event types
    unique_times = np.unique(times)
    for t in unique_times:
        mask_t = times == t
        causes_at_t = np.unique(events[mask_t])
        active_causes = causes_at_t[causes_at_t > 0]
        assert active_causes.size <= 1

    return times, events


@pytest.mark.parametrize("event_of_interest", [1, 2])
def test_aj_matches_lifelines_no_cross_ties(event_of_interest):
    """
    In the absence of ties between different event types (but allowing ties
    within each event type), our Aalen-Johansen implementation and lifelines'
    AalenJohansenFitter should produce numerically identical CIF/S(t)
    (up to floating-point tolerance), provided that:
        - jitter_level=0 (no artificial jitter),
        - no left-truncation, no weights.

    We compare:
        - CIF for the chosen event_of_interest
        - overall survival S(t) = 1 - sum_k F_k(t)
    evaluated on lifelines' default timeline.
    """

    # 1. Generate data
    times, events = _generate_data_no_cross_ties(n=200, random_state=12345)

    # 2. Fit lifelines AJ for event_of_interest (no jitter, no variance)
    aj_life = AalenJohansenFitter(jitter_level=0.0, calculate_variance=False)
    aj_life.fit(times, events, event_of_interest=event_of_interest)

    # Lifelines' timeline and CIF for the event_of_interest
    timeline = aj_life.cumulative_density_.index.values.astype(float)
    cif_life = aj_life.cumulative_density_.iloc[:, 0].values  # column is event_of_interest

    # 3. Fit our AJ competing-risks implementation on the same data
    aj_ours = AalenJohansenCompetingRisks()
    aj_ours.fit(times, events)

    # 4. Evaluate our CIF and survival on the *same* timeline
    cif_ours_full = aj_ours.predict_cif(timeline)  # shape (len(timeline), n_causes)
    cif_ours_event = cif_ours_full[:, event_of_interest - 1]
    surv_ours = aj_ours.predict_surv(timeline)

    # 5. Compare with tight tolerances
    np.testing.assert_allclose(
        cif_ours_event, cif_life, rtol=1e-12, atol=1e-12,
        err_msg="CIF for event_of_interest does not match lifelines."
    )


def test_aj_mass_conservation_all_causes():
    """
    Check that, for all causes simultaneously, the total probability mass is
    conserved for both our implementation:

        S(t) + sum_k F_k(t) ≈ 1  for all t,

    in the setting with:
        - no ties between different event types,
        - ties allowed within each cause,
        - jitter_level=0, no left truncation, no weights.
    """

    times, events = _generate_data_no_cross_ties(n=200, random_state=12345)
    K = int(events.max())
    assert K >= 1

    # Use a common explicit timeline: distinct event times (any cause)
    timeline = np.unique(times[events > 0])

    # --- Our implementation ---
    aj_ours = AalenJohansenCompetingRisks()
    aj_ours.fit(times, events)

    surv_ours = aj_ours.predict_surv(timeline)           # shape (T,)
    cif_ours_all = aj_ours.predict_cif(timeline)         # shape (T, K)
    sum_prob_ours = surv_ours + cif_ours_all.sum(axis=1)

    np.testing.assert_allclose(
        sum_prob_ours, np.ones_like(sum_prob_ours),
        rtol=1e-12, atol=1e-12,
        err_msg="Our AJ implementation does not conserve probability mass."
    )