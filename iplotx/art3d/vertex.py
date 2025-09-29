"""
Module containing code to manipulate vertex visualisations, especially the VertexCollection class.
"""

from typing import (
    Sequence,
)
import numpy as np
from mpl_toolkits.mplot3d.art3d import Patch3DCollection

from ..utils.matplotlib import (
    _forwarder,
)
from ..vertex import (
    VertexCollection,
)


@_forwarder(
    (
        "set_clip_path",
        "set_clip_box",
        "set_snap",
        "set_sketch_params",
        "set_animated",
        "set_picker",
    )
)
class Vertex3DCollection(Patch3DCollection, VertexCollection):
    """Collection of vertex patches for plotting."""

    pass


def vertex_collection_2d_to_3d(
    col: VertexCollection,
    zs: np.ndarray | float | Sequence[float] = 0,
    zdir: str = "z",
    depthshade: bool = True,
    axlim_clip: bool = False,
):
    """Convert a 2D VertexCollection to a 3D Vertex3DCollection.

    Parameters:
        col: The 2D VertexCollection to convert.
        zs: The z coordinate(s) to use for the 3D vertices.
        zdir: The axis to use as the z axis (default is "z").
        depthshade: Whether to apply depth shading (default is True).
        axlim_clip: Whether to clip the vertices to the axes limits (default is False).
    """
    if not isinstance(col, VertexCollection):
        raise TypeError("vertices must be a VertexCollection")

    col.__class__ = Vertex3DCollection
    col._depthshade = depthshade
    col._in_draw = False
    col.set_3d_properties(zs, zdir, axlim_clip)
