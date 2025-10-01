"""
Module containing code to manipulate edge visualisations in 3D, especially the Edge3DCollection class.
"""

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import (
    Line3DCollection,
)

from ..utils.matplotlib import (
    _forwarder,
)
from ..edge import (
    EdgeCollection,
)
from .arrow import (
    arrow_collection_2d_to_3d,
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
class Edge3DCollection(Line3DCollection):
    """Collection of vertex patches for plotting."""

    def get_children(self) -> tuple:
        children = []
        if hasattr(self, "_subedges"):
            children.append(self._subedges)
        if hasattr(self, "_arrows"):
            children.append(self._arrows)
        if hasattr(self, "_label_collection"):
            children.append(self._label_collection)
        return tuple(children)

    def set_figure(self, fig) -> None:
        super().set_figure(fig)
        for child in self.get_children():
            # NOTE: This sets the sizes with correct dpi scaling in the arrows
            child.set_figure(fig)

    @property
    def axes(self):
        return Line3DCollection.axes.__get__(self)

    @axes.setter
    def axes(self, new_axes):
        Line3DCollection.axes.__set__(self, new_axes)
        for child in self.get_children():
            child.axes = new_axes

    def draw(self, renderer) -> None:
        """Draw the collection of vertices in 3D.

        Parameters:
            renderer: The renderer to use for drawing.
        """
        # Render the Line3DCollection first
        super().draw(renderer)

        # Now attempt to draw the children
        for child in self.get_children():
            if isinstance(child.axes, Axes3D) and hasattr(child, "do_3d_projection"):
                child.do_3d_projection()
            if hasattr(child, "_update_before_draw"):
                child._update_before_draw()
            child.draw(renderer)


def edge_collection_2d_to_3d(
    col: EdgeCollection,
    zdir: str = "z",
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

    # TODO: if we make Edge3DCollection a dynamic drawer, this will need to change
    # fundamentally. Also, this currently does not handle labels properly.
    vinfo = col._get_adjacent_vertices_info()

    segments3d = []
    for offset1, offset2 in vinfo["offsets"]:
        segment = [tuple(offset1), tuple(offset2)]
        segments3d.append(segment)

    # NOTE: after this line, none of the EdgeCollection methods will work
    # It's become a static drawer now
    col.__class__ = Edge3DCollection

    col.set_segments(segments3d)
    col._axlim_clip = axlim_clip

    # Convert the arrow collection if present
    if hasattr(col, "_arrows"):
        # Fix the x and y to the center of the target vertex (for now)
        col._arrows._offsets[:] = [segment[0][:2] for segment in segments3d]
        zs = [segment[-1][2] for segment in segments3d]
        arrow_collection_2d_to_3d(
            col._arrows,
            zs=zs,
            zdir=zdir,
            depthshade=False,
            axlim_clip=axlim_clip,
        )
