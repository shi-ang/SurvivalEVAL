import numpy as np

from SurvivalEVAL.Evaluations.Concordance import _get_comparable_ic


def test_get_comparable_ic_returns_directed_precedence_relation():
    left = np.array([0.0, 2.0, 4.0])
    right = np.array([1.0, 3.0, 5.0])

    comparable = _get_comparable_ic(left, right)

    expected = np.array([
        [False, True, True],
        [False, False, True],
        [False, False, False],
    ])
    np.testing.assert_array_equal(comparable, expected)


def test_get_comparable_ic_respects_touching_endpoint_inclusivity():
    # (0, 1] precedes (1, 2], but overlaps the exact event [1, 1].
    left = np.array([0.0, 1.0, 1.0])
    right = np.array([1.0, 2.0, 1.0])

    comparable = _get_comparable_ic(left, right)

    assert comparable[0, 1]
    assert not comparable[1, 0]
    assert not comparable[0, 2]
    assert not comparable[2, 0]
