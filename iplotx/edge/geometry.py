"""
Support module with geometry- and path-related functions for edges.

3D geometry is in a separate module.
"""

from typing import (
    Optional,
    Sequence,
)
from math import atan2, tan, pi
import numpy as np
import matplotlib as mpl

from ..typing import (
    Pair,
)
from .ports import _get_port_unit_vector


def _compute_loops_per_angle(nloops, angles):
    if len(angles) == 0:
        return [(0, 2 * pi, nloops)]

    angles_sorted_closed = list(sorted(angles))
    angles_sorted_closed.append(angles_sorted_closed[0] + 2 * pi)
    deltas = np.diff(angles_sorted_closed)

    # Now we have the deltas and the total number of loops
    # 1. Assign all loops to the largest wedge
    idx_dmax = deltas.argmax()
    if nloops == 1:
        return [
            (
                angles_sorted_closed[idx_dmax],
                angles_sorted_closed[idx_dmax + 1],
                nloops,
            )
        ]

    # 2. Check if any other wedges are larger than this
    # If not, we are done (this is the algo in igraph)
    dsplit = deltas[idx_dmax] / nloops
    if (deltas > dsplit).sum() < 2:
        return [
            (
                angles_sorted_closed[idx_dmax],
                angles_sorted_closed[idx_dmax + 1],
                nloops,
            )
        ]

        # 3. Check how small the second-largest wedge would become
    idx_dsort = np.argsort(deltas)
    return [
        (
            angles_sorted_closed[idx_dmax],
            angles_sorted_closed[idx_dmax + 1],
            nloops - 1,
        ),
        (
            angles_sorted_closed[idx_dsort[-2]],
            angles_sorted_closed[idx_dsort[-2] + 1],
            1,
        ),
    ]


def _get_shorter_edge_coords(vpath, vsize, theta, shrink=0):
    """Get the coordinates of an edge tip such that it touches the vertex border.

    Parameters:
        vpath: the vertex path, in figure coordinates (so scaled by dpi).
        vsize: the vertex max size, in figure coordinates (so scaled by dpi).
        theta: the angle of the edge inpinging into the vertex, in radians, in figure coordinates.
        shrink: additional shrinking of the edge, in figure coordinates (so scaled by dpi).
    """
    # Bound theta from -pi to pi (why is that not guaranteed?)
    theta = (theta + pi) % (2 * pi) - pi

    # Size zero vertices need no shortening
    if vsize == 0:
        ve = np.array([0, 0])
    else:
        for i in range(len(vpath)):
            v1 = vpath.vertices[i]
            v2 = vpath.vertices[(i + 1) % len(vpath)]
            theta1 = atan2(*((v1)[::-1]))
            theta2 = atan2(*((v2)[::-1]))

            # atan2 ranges ]-3.14, 3.14]
            # so it can be that theta1 is -3 and theta2 is +3
            # therefore we need two separate cases, one that cuts at pi and one at 0
            cond1 = theta1 <= theta <= theta2
            cond2 = (
                (theta1 + 2 * pi) % (2 * pi)
                <= (theta + 2 * pi) % (2 * pi)
                <= (theta2 + 2 * pi) % (2 * pi)
            )
            if cond1 or cond2:
                break
        else:
            raise ValueError("Angle for patch not found")

        # The edge meets the patch of the vertex on the v1-v2 size,
        # at angle theta from the center
        mtheta = tan(theta)
        if v2[0] == v1[0]:
            xe = v1[0]
        else:
            m12 = (v2[1] - v1[1]) / (v2[0] - v1[0])
            xe = (v1[1] - m12 * v1[0]) / (mtheta - m12)
        ye = mtheta * xe
        ve = np.array([xe, ye])

    ve = ve * vsize

    # Shrink (assuming dpi scaling is already applied to the shrink)
    ve += shrink * np.array([np.cos(theta), np.sin(theta)])

    return ve


def _fix_parallel_edges_straight(
    paths,
    indices,
    indices_inv,
    trans,
    trans_inv,
    paralleloffset=3,
):
    """Offset parallel edges along the same path."""
    ntot = len(indices) + len(indices_inv)

    # This is straight so two vertices anyway
    # NOTE: all paths will be the same, which is why we need to offset them
    vs, ve = trans(paths[indices[0]].vertices)

    # Move orthogonal to the line
    fracs = (vs - ve) / np.sqrt(((vs - ve) ** 2).sum()) @ np.array([[0, 1], [-1, 0]])

    # NOTE: for now treat both direction the same
    for i, idx in enumerate(indices + indices_inv):
        # Offset the path
        paths[idx].vertices = trans_inv(
            trans(paths[idx].vertices) + fracs * paralleloffset * (i - ntot / 2)
        )


def _compute_loop_path(
    vcoord_fig,
    vpath,
    vsize,
    angle1,
    angle2,
    trans_inv,
    looptension,
    shrink=0,
):
    # Shorten at starting angle
    start = _get_shorter_edge_coords(vpath, vsize, angle1, shrink) + vcoord_fig
    # Shorten at end angle
    end = _get_shorter_edge_coords(vpath, vsize, angle2, shrink) + vcoord_fig

    aux1 = (start - vcoord_fig) * looptension + vcoord_fig
    aux2 = (end - vcoord_fig) * looptension + vcoord_fig

    vertices = np.vstack(
        [
            start,
            aux1,
            aux2,
            end,
        ]
    )
    codes = ["MOVETO"] + ["CURVE4"] * 3

    # Offset to place and transform to data coordinates
    vertices = trans_inv(vertices)
    codes = [getattr(mpl.path.Path, x) for x in codes]
    path = mpl.path.Path(
        vertices,
        codes=codes,
    )
    return path


def _compute_edge_path_straight(
    vcoord_data,
    vpath_fig,
    vsize_fig,
    trans,
    trans_inv,
    layout_coordinate_system: str = "cartesian",
    shrink: float = 0,
    **kwargs,
):
    if layout_coordinate_system not in ("cartesian", "polar"):
        raise ValueError(
            f"Layout coordinate system not supported for straight edges: {layout_coordinate_system}.",
        )

    if layout_coordinate_system == "polar":
        r0, theta0 = vcoord_data[0]
        r1, theta1 = vcoord_data[1]
        vcoord_data_cart = np.array(
            [
                [r0 * np.cos(theta0), r0 * np.sin(theta0)],
                [r1 * np.cos(theta1), r1 * np.sin(theta1)],
            ]
        )
    else:
        vcoord_data_cart = vcoord_data

    # Coordinates in figure (default) coords
    vcoord_fig = trans(vcoord_data_cart)

    points = []

    # Angle of the straight line
    theta = atan2(*((vcoord_fig[1] - vcoord_fig[0])[::-1]))

    # Shorten at starting vertex
    vs = _get_shorter_edge_coords(vpath_fig[0], vsize_fig[0], theta, shrink) + vcoord_fig[0]
    points.append(vs)

    # Shorten at end vertex
    ve = _get_shorter_edge_coords(vpath_fig[1], vsize_fig[1], theta + pi, shrink) + vcoord_fig[1]
    points.append(ve)

    codes = ["MOVETO", "LINETO"]
    path = mpl.path.Path(
        points,
        codes=[getattr(mpl.path.Path, x) for x in codes],
    )
    path.vertices = trans_inv(path.vertices)
    return path, (theta, theta + np.pi)


def _compute_edge_path_waypoints(
    waypoints,
    vcoord_data,
    vpath_fig,
    vsize_fig,
    trans,
    trans_inv,
    layout_coordinate_system: str = "cartesian",
    points_per_curve: int = 30,
    ports: Pair[Optional[str]] = (None, None),
    shrink: float = 0,
    **kwargs,
):
    if not isinstance(waypoints, str):
        # Only cartesian coordinates supported for numerical waypoints for now
        assert layout_coordinate_system == "cartesian"

        waypoints = trans(np.array(waypoints, ndmin=2))

        # Coordinates in figure (default) coords
        vcoord_fig = trans(vcoord_data)

        # Angles of the straight lines
        thetas = [None, None]
        vshorts = [None, None]
        for i in range(2):
            # This picks always the first waypoint for i == 0,
            # the last waypoint for i == 1. They might be the same.
            waypoint = waypoints[-i]
            if ports[i] is None:
                thetas[i] = atan2(*((waypoint - vcoord_fig[i])[::-1]))
            else:
                thetas[i] = atan2(*(_get_port_unit_vector(ports[i], trans_inv)[::-1]))

            # Shorten at vertex border
            vshorts[i] = (
                _get_shorter_edge_coords(vpath_fig[i], vsize_fig[i], thetas[i], shrink)
                + vcoord_fig[i]
            )

        points = [vshorts[0]] + list(waypoints) + [vshorts[1]]
        codes = ["MOVETO"] + ["LINETO"] * len(waypoints) + ["LINETO"]
        angles = tuple(thetas)

    elif waypoints in ("x0y1", "y0x1"):
        assert layout_coordinate_system == "cartesian"

        # Coordinates in figure (default) coords
        vcoord_fig = trans(vcoord_data)

        if waypoints == "x0y1":
            waypoint = np.array([vcoord_fig[0][0], vcoord_fig[1][1]])
        else:
            waypoint = np.array([vcoord_fig[1][0], vcoord_fig[0][1]])

        # Angles of the straight lines
        thetas = [None, None]
        vshorts = [None, None]
        for i in range(2):
            if ports[i] is None:
                thetas[i] = atan2(*((waypoint - vcoord_fig[i])[::-1]))
            else:
                thetas[i] = atan2(*(_get_port_unit_vector(ports[i], trans_inv)[::-1]))

            # Shorten at vertex border
            vshorts[i] = (
                _get_shorter_edge_coords(vpath_fig[i], vsize_fig[i], thetas[i], shrink)
                + vcoord_fig[i]
            )

        # Shorten waypoints to keep the angles right
        if waypoints == "x0y1":
            waypoint[0] = vshorts[0][0]
            waypoint[1] = vshorts[1][1]
        else:
            waypoint[1] = vshorts[0][1]
            waypoint[0] = vshorts[1][0]

        points = [vshorts[0], waypoint, vshorts[1]]
        codes = ["MOVETO", "LINETO", "LINETO"]
        angles = tuple(thetas)
    elif waypoints in ("xmidy0,xmidy1", "x0ymid,x1ymid"):
        # S-shaped orthogonal line
        assert layout_coordinate_system == "cartesian"

        # Coordinates in figure (default) coords
        vcoord_fig = trans(vcoord_data)

        if waypoints == "xmidy0,xmidy1":
            xmid = 0.5 * (vcoord_fig[0][0] + vcoord_fig[1][0])
            waypoint_array = np.array(
                [
                    [xmid, vcoord_fig[0][1]],
                    [xmid, vcoord_fig[1][1]],
                ]
            )
        else:
            ymid = 0.5 * (vcoord_fig[0][1] + vcoord_fig[1][1])
            waypoint_array = np.array(
                [
                    [vcoord_fig[0][0], ymid],
                    [vcoord_fig[1][0], ymid],
                ]
            )

        # Angles of the straight lines
        thetas = []
        vshorts = []
        for i in range(2):
            if ports[i] is None:
                theta = atan2(*((waypoint_array[i] - vcoord_fig[i])[::-1]))
            else:
                theta = atan2(*(_get_port_unit_vector(ports[i], trans_inv)[::-1]))

            # Shorten at vertex border
            vshort = (
                _get_shorter_edge_coords(vpath_fig[i], vsize_fig[i], theta, shrink) + vcoord_fig[i]
            )
            thetas.append(theta)
            vshorts.append(vshort)

        points = [vshorts[0], waypoint_array[0], waypoint_array[1], vshorts[1]]
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO"]
        angles = tuple(thetas)

    elif waypoints == "r0a1":
        assert layout_coordinate_system == "polar"

        r0, alpha0 = vcoord_data[0]
        r1, alpha1 = vcoord_data[1]
        idx_inner = np.argmin([r0, r1])
        idx_outer = 1 - idx_inner
        alpha_outer = [alpha0, alpha1][idx_outer]

        betas = np.linspace(alpha0, alpha1, points_per_curve)
        waypoints = [r0, r1][idx_inner] * np.vstack([np.cos(betas), np.sin(betas)]).T
        endpoint = [r0, r1][idx_outer] * np.array([np.cos(alpha_outer), np.sin(alpha_outer)])
        points = np.array(list(waypoints) + [endpoint])
        points = trans(points)
        codes = ["MOVETO"] + ["LINETO"] * len(waypoints)
        angles = (alpha0 + pi / 2, alpha1)

    else:
        raise NotImplementedError(
            f"Edge shortening with waypoints not implemented yet: {waypoints}.",
        )

    path = mpl.path.Path(
        points,
        codes=[getattr(mpl.path.Path, x) for x in codes],
    )

    path.vertices = trans_inv(path.vertices)
    return path, angles


def _compute_edge_path_curved(
    tension,
    vcoord_data,
    vpath_fig,
    vsize_fig,
    trans,
    trans_inv,
    ports: Pair[Optional[str]] = (None, None),
    shrink: float = 0,
):
    """Shorten the edge path along a cubic Bezier between the vertex centres.

    The most important part is that the derivative of the Bezier at the start
    and end point towards the vertex centres: people notice if they do not.
    """

    # Coordinates in figure (default) coords
    vcoord_fig = trans(vcoord_data)

    dv = vcoord_fig[1] - vcoord_fig[0]
    edge_straight_length = np.sqrt((dv**2).sum())

    auxs = [None, None]
    for i in range(2):
        if ports[i] is not None:
            der = _get_port_unit_vector(ports[i], trans_inv)
            auxs[i] = der * edge_straight_length * tension + vcoord_fig[i]

    # Both ports defined, just use them and hope for the best
    # Obviously, if the user specifies ports that make no sense,
    # this is going to be a (technically valid) mess.
    if all(aux is not None for aux in auxs):
        pass

    # If no ports are specified (the most common case), compute
    # the Bezier and shorten it
    elif all(aux is None for aux in auxs):
        # Put auxs along the way
        auxs = np.array(
            [
                vcoord_fig[0] + 0.33 * dv,
                vcoord_fig[1] - 0.33 * dv,
            ]
        )
        # Right rotation from the straight edge
        dv_rot = -0.1 * dv @ np.array([[0, 1], [-1, 0]])
        # Shift the auxs orthogonal to the straight edge
        auxs += dv_rot * tension

    # First port is defined
    elif (auxs[0] is not None) and (auxs[1] is None):
        auxs[1] = auxs[0]

    # Second port is defined
    else:
        auxs[0] = auxs[1]

    vs = [None, None]
    thetas = [None, None]
    for i in range(2):
        thetas[i] = atan2(*((auxs[i] - vcoord_fig[i])[::-1]))
        vs[i] = (
            _get_shorter_edge_coords(vpath_fig[i], vsize_fig[i], thetas[i], shrink) + vcoord_fig[i]
        )

    path = {
        "vertices": [
            vs[0],
            auxs[0],
            auxs[1],
            vs[1],
        ],
        "codes": ["MOVETO"] + ["CURVE4"] * 3,
    }

    path = mpl.path.Path(
        path["vertices"],
        codes=[getattr(mpl.path.Path, x) for x in path["codes"]],
    )

    # Return to data transform
    path.vertices = trans_inv(path.vertices)
    return path, tuple(thetas)


def _compute_edge_path(
    *args,
    tension: float = 0,
    waypoints: str | tuple[float, float] | Sequence[tuple[float, float]] | np.ndarray = "none",
    ports: Pair[Optional[str]] = (None, None),
    layout_coordinate_system: str = "cartesian",
    **kwargs,
):
    """Compute the edge path in a few different ways."""
    if (waypoints != "none") and (tension != 0):
        raise ValueError("Waypoints not supported for curved edges.")

    if waypoints != "none":
        return _compute_edge_path_waypoints(
            waypoints,
            *args,
            layout_coordinate_system=layout_coordinate_system,
            ports=ports,
            **kwargs,
        )

    if tension == 0:
        return _compute_edge_path_straight(
            *args,
            layout_coordinate_system=layout_coordinate_system,
            **kwargs,
        )

    return _compute_edge_path_curved(
        tension,
        *args,
        ports=ports,
        **kwargs,
    )
