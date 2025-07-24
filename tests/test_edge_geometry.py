import pytest
import numpy as np
import matplotlib as mpl
from iplotx.edge import geometry


def test_loops_per_angle_empty():
    res = geometry._compute_loops_per_angle(10, [])
    assert res == [(0, 2 * np.pi, 10)]


def test_shorter_edge_invalid():
    path = mpl.path.Path(np.array([[0, 0], [1, 1]]))
    with pytest.raises(ValueError):
        geometry._get_shorter_edge_coords(
            path,
            5,
            -0.5,
        )


def test_compute_path_straight_invalid():
    with pytest.raises(ValueError):
        geometry._compute_edge_path_straight(
            None,
            None,
            None,
            None,
            None,
            layout_coordinate_system="invalid",
        )


@pytest.fixture
def args_make_paths():
    vpath = mpl.path.Path(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]))
    transform = mpl.transforms.IdentityTransform()
    trans = transform.transform
    trans_inv = transform.inverted().transform
    return (vpath, trans, trans_inv)


def test_compute_path_straight_polar(args_make_paths):
    vpath, trans, trans_inv = args_make_paths
    epath, angles = geometry._compute_edge_path_straight(
        [[0, 0], [10, 0]],
        [vpath] * 2,
        (2, 2),
        trans,
        trans_inv,
        layout_coordinate_system="polar",
    )
    assert isinstance(epath, mpl.path.Path)
    assert len(angles) == 2
    assert isinstance(angles[0], float)
    assert isinstance(angles[1], float)
    assert np.round(angles[0] - angles[1] + np.pi, 2) == 0.0


@pytest.mark.parametrize("waypoints,expected", [["y0x1", [10, 0]], ["x0y1", [0, 5]], [[0, 5]] * 2])
def test_compute_waypoints(args_make_paths, waypoints, expected):
    vpath, trans, trans_inv = args_make_paths
    epath, angles = geometry._compute_edge_path_waypoints(
        waypoints,
        [[0, 0], [10, 5]],
        [vpath] * 2,
        (2, 2),
        trans,
        trans_inv,
    )
    assert isinstance(epath, mpl.path.Path)
    assert len(epath.vertices) == 3
    assert list(epath.vertices[1]) == expected


def test_compute_waypoints_port(args_make_paths):
    vpath, trans, trans_inv = args_make_paths
    epath, angles = geometry._compute_edge_path_waypoints(
        "x0y1",
        [[0, 0], [10, 5]],
        [vpath] * 2,
        (2, 2),
        trans,
        trans_inv,
        ports=("n", "w"),
    )
    assert isinstance(epath, mpl.path.Path)
    assert len(epath.vertices) == 3
    assert list(epath.vertices[1]) == [0, 5]


@pytest.mark.parametrize(
    "waypoints,expected",
    [
        ["xmidy0,xmidy1", [[5, 0], [5, 5]]],
        ["x0ymid,x1ymid", [[0, 2.5], [10, 2.5]]],
    ],
)
def test_double_waypoints(args_make_paths, waypoints, expected):
    vpath, trans, trans_inv = args_make_paths
    epath, angles = geometry._compute_edge_path_waypoints(
        waypoints,
        [[0, 0], [10, 5]],
        [vpath] * 2,
        (2, 2),
        trans,
        trans_inv,
    )
    assert isinstance(epath, mpl.path.Path)
    assert len(epath.vertices) == 4
    assert list(epath.vertices[1]) == expected[0]
    assert list(epath.vertices[2]) == expected[1]


@pytest.mark.parametrize(
    "waypoints,expected,port",
    [
        ["xmidy0,xmidy1", [[5, 0], [5, 5]], "w"],
        ["x0ymid,x1ymid", [[0, 2.5], [10, 2.5]], "s"],
    ],
)
def test_double_waypoints_ports(args_make_paths, waypoints, expected, port):
    vpath, trans, trans_inv = args_make_paths
    epath, angles = geometry._compute_edge_path_waypoints(
        waypoints,
        [[0, 0], [10, 5]],
        [vpath] * 2,
        (2, 2),
        trans,
        trans_inv,
        ports=(None, port),
    )
    assert isinstance(epath, mpl.path.Path)
    assert len(epath.vertices) == 4
    assert list(epath.vertices[1]) == expected[0]
    assert list(epath.vertices[2]) == expected[1]
