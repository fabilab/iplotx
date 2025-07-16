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
    np.testing.assert_array_equal(hull, [2, 1, 0])


def test_hull_four(four_points):
    hull = geometry.convex_hull(four_points)
    np.testing.assert_array_equal(hull, [3, 2, 1, 0])


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
