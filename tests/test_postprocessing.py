"""Tests for postprocessing module."""

import numpy as np

from chess_detector.data.postprocessing import gen_diff, reorder_points


def test_gen_diff():
    """Test gen_diff function."""
    before = np.full((10, 10), 100, dtype=np.uint8)
    after = np.full((10, 10), 50, dtype=np.uint8)

    # Standard absolute difference
    diff = gen_diff(before, after, binary=False)
    assert diff.shape == (10, 10)
    assert np.all(diff == 50)

    # Binary threshold difference
    diff_binary = gen_diff(before, after, binary=True, binary_threshold=30)
    assert diff_binary.shape == (10, 10)
    assert np.all(diff_binary == 255)


def test_reorder_points():
    # Randomly ordered square corners (x, y)
    """Test reorder_points function."""
    pts = np.array(
        [
            [10, 10],  # bottom-right
            [0, 0],  # top-left
            [0, 10],  # bottom-left
            [10, 0],  # top-right
        ]
    )

    ordered = reorder_points(pts)

    # Expected order: top-left, top-right, bottom-right, bottom-left
    expected = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)

    np.testing.assert_array_equal(ordered, expected)
