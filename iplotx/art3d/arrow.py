"""
Module containing code to manipulate arrow visualisations in 3D, especially the EdgeArrow3DCollection class.
"""

from typing import (
    Sequence,
)
import numpy as np
from matplotlib import (
    cbook,
)
from mpl_toolkits.mplot3d.art3d import (
    Path3DCollection,
)

from ..utils.matplotlib import (
    _forwarder,
)
from ..edge.arrow import (
    EdgeArrowCollection,
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
class EdgeArrow3DCollection(EdgeArrowCollection, Path3DCollection):
    """Collection of vertex patches for plotting."""

    def draw(self, renderer) -> None:
        """Draw the collection of vertices in 3D.

        Parameters:
            renderer: The renderer to use for drawing.
        """
        with self._use_zordered_offset():
            with cbook._setattr_cm(self, _in_draw=True):
                EdgeArrowCollection.draw(self, renderer)


def arrow_collection_2d_to_3d(
    col: EdgeArrowCollection,
    zs: np.ndarray | float | Sequence[float] = 0,
    zdir: str = "z",
    depthshade: bool = True,
    axlim_clip: bool = False,
):
    """Convert a 2D EdgeArrowCollection to a 3D EdgeArrow3DCollection.

    Parameters:
        col: The 2D EdgeArrowCollection to convert.
        zs: The z coordinate(s) to use for the 3D vertices.
        zdir: The axis to use as the z axis (default is "z").
        depthshade: Whether to apply depth shading (default is True).
        axlim_clip: Whether to clip the vertices to the axes limits (default is False).
    """
    if not isinstance(col, EdgeArrowCollection):
        raise TypeError("vertices must be a EdgeArrowCollection")

    col.__class__ = EdgeArrow3DCollection
    col._offset_zordered = None
    col._depthshade = depthshade
    col._in_draw = False
    col.set_3d_properties(zs, zdir, axlim_clip)
