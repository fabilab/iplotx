"""
Module containing code to manipulate arrow visualisations in 3D, especially the EdgeArrow3DCollection class.
"""

from typing import (
    Sequence,
)
from math import atan2, cos, sin
import numpy as np
from matplotlib import (
    cbook,
)
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import (
    Path3DCollection,
)

from ...utils.matplotlib import (
    _forwarder,
)
from ...edge.arrow import (
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

    def _update_before_draw(self) -> None:
        """Update the collection before drawing."""
        if (
            isinstance(self.axes, Axes3D)
            and hasattr(self, "do_3d_projection")
            and (self.axes.M is not None)
        ):
            self.do_3d_projection()

        # The original EdgeArrowCollection method for
        # _update_before_draw cannot be used because it
        # relies on paths, whereas edges are now a
        # Line3DCollection which uses segments.
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)

        if (not hasattr(self, "_z_markers_idx")) or (
            not isinstance(self._z_markers_idx, np.ndarray)
        ):
            return

        trans = self.get_offset_transform().transform

        # The do_3d_projection method above reorders the
        # arrow offsets in some way, so we might have to figure out
        # what edge index corres
        for i, ie in enumerate(self._z_markers_idx):
            segments_2d = self._edge_collection.get_segments()[ie]

            # We could reset the 3d projection here, might be a way to
            # skip the function call above.
            v2 = trans(segments_2d[-1])
            v1 = trans(segments_2d[-2])
            dv = v2 - v1
            theta = atan2(*(dv[::-1]))
            theta_old = self._angles[i]
            dtheta = theta - theta_old
            mrot = np.array([[cos(dtheta), sin(dtheta)], [-sin(dtheta), cos(dtheta)]])

            apath = self._paths[i]
            apath.vertices = apath.vertices @ mrot
            self._angles[i] = theta

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
