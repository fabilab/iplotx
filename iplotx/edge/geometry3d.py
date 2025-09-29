"""
Support for computing edge paths in 3D.
"""

from typing import (
    Optional,
    Sequence,
)
import numpy as np
import matplotlib as mpl

from ..typing import (
    Pair,
)


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
    """Compute straight edge path between two vertices, in 3D.

    Parameters:
        vcoord_data: Vertex coordinates in data coordinates, shape (2, 3).
        vpath_fig: Vertex path in figure coordinates.
        vsize_fig: Vertex size in figure coordinates.
        trans: Transformation from data to figure coordinates.
        trans_inv: Inverse transformation from figure to data coordinates.
        layout_coordinate_system: The coordinate system of the layout.
        shrink: Amount to shorten the edge at each end, in figure coordinates.
        **kwargs: Additional keyword arguments (not used).
    Returns:
        A pair with the path and a tuple of angles of exit and entry, in radians.

    """

    if layout_coordinate_system not in ("cartesian"):
        raise ValueError(
            f"Layout coordinate system not supported for straight edges in 3D: {layout_coordinate_system}.",
        )

    vcoord_data_cart = vcoord_data

    # Coordinates in figure (default) coords
    vcoord_fig = trans(vcoord_data_cart)

    points = []

    # Angles of the straight line
    # FIXME: In 2D, this is only used to make space for loops
    # let's ignore for now
    # theta = atan2(*((vcoord_fig[1] - vcoord_fig[0])[::-1]))
    theta = 0

    # TODO: Shorten at starting vertex (?)
    vs = vcoord_fig[0]
    points.append(vs)

    # TODO: Shorten at end vertex (?)
    ve = vcoord_fig[1]
    points.append(ve)

    codes = ["MOVETO", "LINETO"]
    path = mpl.path.Path(
        points,
        codes=[getattr(mpl.path.Path, x) for x in codes],
    )
    path.vertices = trans_inv(path.vertices)
    return path, (theta, theta + np.pi)


def _compute_edge_path_3d(
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
        raise NotImplementedError("Waypoints not implemented for 3D edges.")
        # return _compute_edge_path_waypoints(
        #    waypoints,
        #    *args,
        #    layout_coordinate_system=layout_coordinate_system,
        #    ports=ports,
        #    **kwargs,
        # )

    if np.isscalar(tension) and (tension == 0):
        return _compute_edge_path_straight(
            *args,
            layout_coordinate_system=layout_coordinate_system,
            **kwargs,
        )

    raise NotImplementedError("Curved edges not implemented for 3D edges.")
    # return _compute_edge_path_curved(
    #    tension,
    #    *args,
    #    ports=ports,
    #    **kwargs,
    # )
