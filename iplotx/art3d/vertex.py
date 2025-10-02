"""
Module containing code to manipulate vertex visualisations in 3D, especially the Vertex3DCollection class.
"""

from typing import (
    Sequence,
)
import numpy as np
from matplotlib import (
    cbook,
)
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import (
    Path3DCollection,
    text_2d_to_3d,
)

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
class Vertex3DCollection(VertexCollection, Path3DCollection):
    """Collection of vertex patches for plotting."""

    def _update_before_draw(self) -> None:
        """Update the collection before drawing."""
        # Set the sizes according to the current figure dpi
        VertexCollection._update_before_draw(self)

        if isinstance(self.axes, Axes3D) and hasattr(self, "do_3d_projection"):
            self.do_3d_projection()

    def draw(self, renderer) -> None:
        """Draw the collection of vertices in 3D.

        Parameters:
            renderer: The renderer to use for drawing.
        """
        with self._use_zordered_offset():
            with cbook._setattr_cm(self, _in_draw=True):
                VertexCollection.draw(self, renderer)


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
        depthshade: Whether to aply depth shading (default is True).
        axlim_clip: Whether to clip the vertices to the axes limits (default is False).
    """
    if not isinstance(col, VertexCollection):
        raise TypeError("vertices must be a VertexCollection")

    col.__class__ = Vertex3DCollection
    col._offset_zordered = None
    col._depthshade = depthshade
    col._in_draw = False
    col.set_3d_properties(zs, zdir, axlim_clip)

    # Labels if present
    if col.get_labels() is not None:
        for z, art in zip(zs, col.get_labels()._labelartists):
            # zdir=None means the text is always horizontal facing the camera
            text_2d_to_3d(art, z, zdir=None, axlim_clip=axlim_clip)
