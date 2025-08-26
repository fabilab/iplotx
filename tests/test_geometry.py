import pytest
import numpy as np
from functools import partial
import matplotlib as mpl
from iplotx.utils import geometry


@pytest.fixture
def three_points():
    return np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
        ]
    )


@pytest.fixture
def four_points():
    return np.array(
        [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ]
    )


# 13: two tests fail with AssertionError
def assert_array_equal_upon_cycling(points1, points2):
    """Determine array equality but allow order to be different by cycling.

    Parameters:
        points1: First array of points.
        points2: Second array of points.
    Returns:
        True if the arrays are equal upon cycling, False otherwise.
    """

    def _fun(points1, points2):
        if len(points1) != len(points2):
            return False

        points1 = np.asarray(points1)
        points2 = np.asarray(points2).astype(points1.dtype)

        if (points1.ndim != 1) or (points2.ndim != 1):
            raise ValueError("Only 1D arrays are supported.")

        if (points1 == points2).all():
            return True

        # Cycle
        points2 = np.concatenate([points2, points2])
        i0 = np.flatnonzero((points2 == points1[0]))
        if len(i0) != 2:
            return False
        i0 = i0[0]
        points2 = points2[i0 : i0 + len(points1)]

        # Check again
        return bool((points1 == points2).all())

    assert _fun(points1, points2)


def test_squared_bezier(three_points):
    ev = partial(geometry._evaluate_squared_bezier, three_points)
    np.testing.assert_array_almost_equal(ev(0), three_points[0])
    np.testing.assert_array_almost_equal(ev(1), three_points[2])
    np.testing.assert_array_almost_equal(
        ev(0.5),
        [0.75, 0.25],
    )


def test_cubic_bezier(four_points):
    ev = partial(geometry._evaluate_cubic_bezier, four_points)
    np.testing.assert_array_almost_equal(ev(0), four_points[0])
    np.testing.assert_array_almost_equal(ev(1), four_points[3])
    np.testing.assert_array_almost_equal(
        ev(0.5),
        [0.75, 0.5],
    )


def test_hull_empty():
    hull = geometry.convex_hull([])
    np.testing.assert_array_equal(hull, [])


def test_hull_one(three_points):
    hull = geometry.convex_hull(three_points[:1])
    np.testing.assert_array_equal(hull, [0])


def test_hull_two(three_points):
    hull = geometry.convex_hull(three_points[:2])
    np.testing.assert_array_equal(hull, [0, 1])


def test_hull_three(three_points):
    hull = geometry.convex_hull(three_points)
    assert_array_equal_upon_cycling(hull, [2, 1, 0])


def test_hull_four(four_points):
    hull = geometry.convex_hull(four_points)
    assert_array_equal_upon_cycling(hull, [3, 2, 1, 0])


def test_hull_internal_three(three_points):
    hull = geometry._convex_hull_Graham_scan(three_points)
    # NOTE: It's ok to start from a different point, as layout_small_ring
    # as the subsequent is the same
    np.testing.assert_array_equal(hull, [0, 2, 1])


def test_hull_internal_four(four_points):
    hull = geometry._convex_hull_Graham_scan(four_points)
    # NOTE: It's ok to start from a different point, as layout_small_ring
    # as the subsequent is the same
    np.testing.assert_array_equal(hull, [0, 3, 2, 1])


def test_hull_internal_repetition(four_points):
    points = np.vstack([four_points[:3], four_points[2:]])
    hull = geometry._convex_hull_Graham_scan(points)
    # NOTE: It's ok to start from a different point, as layout_small_ring
    # as the subsequent is the same
    np.testing.assert_array_equal(hull, [0, 4, 3, 1])


def test_hull_internal_repetition_internal(four_points):
    points = np.vstack([four_points[:4], 2 * four_points[2:3], four_points[3:]])
    hull = geometry._convex_hull_Graham_scan(points)
    # NOTE: It's ok to start from a different point, as layout_small_ring
    # as the subsequent is the same
    np.testing.assert_array_equal(hull, [0, 5, 4, 1])


def test_hull_internal_repetition_internal_backtrack():
    points = np.array(
        [
            (0, 0),
            (1, 0),
            (0.5, 0.5),
            # This has a larger angle but created a concave bay that needs backtracking
            (1, 1.1),
            (0, 1),
        ]
    )
    hull = geometry._convex_hull_Graham_scan(points)
    # NOTE: It's ok to start from a different point, as layout_small_ring
    # as the subsequent is the same
    np.testing.assert_array_equal(hull, [0, 4, 3, 1])


def test_hull_internal_repetition_internal_dlast():
    points = np.array(
        [
            (0, 0),
            (1, 0),
            (1, 1),
            # This has a same angle as last but shorter distance, should be ignored
            (0.5, 0.5),
            (0, 1),
        ]
    )
    hull = geometry._convex_hull_Graham_scan(points)
    # NOTE: It's ok to start from a different point, as layout_small_ring
    # as the subsequent is the same
    np.testing.assert_array_equal(hull, [0, 4, 2, 1])


def test_vertexpadding_none(three_points):
    hull_idx = geometry.convex_hull(three_points[:1])
    hull = three_points[hull_idx]
    padding = geometry._compute_group_path_with_vertex_padding(
        hull,
        np.array([three_points[0]] * 31),
        mpl.transforms.IdentityTransform(),
        vertexpadding=0,
    )
    assert len(padding) == 31


def test_vertexpadding_zero():
    padding = geometry._compute_group_path_with_vertex_padding(
        [],
        np.array([]),
        mpl.transforms.IdentityTransform(),
        vertexpadding=0,
    )
    assert padding is None


def test_vertexpadding_one(three_points):
    hull_idx = geometry.convex_hull(three_points[:1])
    hull = three_points[hull_idx]
    padding = geometry._compute_group_path_with_vertex_padding(
        hull,
        np.array([three_points[0]] * 31),
        mpl.transforms.IdentityTransform(),
        vertexpadding=10,
    )
    assert len(padding) == 31


def test_vertexpadding_two(three_points):
    hull_idx = geometry.convex_hull(three_points[:2])
    hull = three_points[hull_idx]
    padding = geometry._compute_group_path_with_vertex_padding(
        hull,
        np.array([three_points[0]] * 61),
        mpl.transforms.IdentityTransform(),
        vertexpadding=10,
    )
    assert len(padding) == 61


def test_vertexpadding_three(three_points):
    hull_idx = geometry.convex_hull(three_points)
    hull = three_points[hull_idx]
    padding = geometry._compute_group_path_with_vertex_padding(
        hull,
        np.array(
            [three_points[0]] * 30
            + [three_points[1]] * 30
            + [three_points[2]] * 30
            + [three_points[0]]
        ),
        mpl.transforms.IdentityTransform(),
        vertexpadding=10,
    )

    # 3 points, each with 30 points + 1 for the closing point
    assert len(padding) == 3 * 30 + 1
