"""
Module containing code to manipulate edge visualisations in 3D, especially the Edge3DCollection class.
"""

from typing import (
    Sequence,
)
import numpy as np
from matplotlib import (
    cbook,
)
from mpl_toolkits.mplot3d.art3d import Patch3DCollection

from ..utils.matplotlib import (
    _forwarder,
)
from ..edge import (
    EdgeCollection,
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
class Edge3DCollection(EdgeCollection, Patch3DCollection):
    """Collection of vertex patches for plotting."""

    def draw(self, renderer) -> None:
        """Draw the collection of vertices in 3D.

        Parameters:
            renderer: The renderer to use for drawing.
        """
        with self._use_zordered_offset():
            with cbook._setattr_cm(self, _in_draw=True):
                EdgeCollection.draw(self, renderer)


def edge_collection_2d_to_3d(
    col: EdgeCollection,
    zs: np.ndarray | float | Sequence[float] = 0,
    zdir: str = "z",
    depthshade: bool = True,
    axlim_clip: bool = False,
):
    """Convert a 2D EdgeCollection to a 3D Edge3DCollection.

    Parameters:
        col: The 2D EdgeCollection to convert.
        zs: The z coordinate(s) to use for the 3D vertices.
        zdir: The axis to use as the z axis (default is "z").
        depthshade: Whether to apply depth shading (default is True).
        axlim_clip: Whether to clip the vertices to the axes limits (default is False).
    """
    if not isinstance(col, EdgeCollection):
        raise TypeError("vertices must be a VertexCollection")

    col.__class__ = Edge3DCollection
    col._offset_zordered = None
    col._depthshade = depthshade
    col._in_draw = False

    # FIXME: This should be actually fixed...
    col.set_3d_properties(zs, zdir, axlim_clip)
