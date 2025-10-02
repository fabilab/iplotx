"""
Support for computing edge paths in 3D.
"""

from typing import (
    Optional,
    Sequence,
)
import numpy as np

from ...typing import (
    Pair,
)


def _compute_edge_segments_straight(
    vcoord_data,
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

    segments = [vcoord_data[0], vcoord_data[1]]
    return segments


def _compute_edge_segments(
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
        return _compute_edge_segments_straight(
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
