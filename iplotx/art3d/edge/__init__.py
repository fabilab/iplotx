"""
Module containing code to manipulate edge visualisations in 3D, especially the Edge3DCollection class.
"""

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import (
    Line3DCollection,
)

from ...utils.matplotlib import (
    _forwarder,
)
from ...edge import (
    EdgeCollection,
)
from .arrow import (
    arrow_collection_2d_to_3d,
)
from .geometry import (
    _compute_edge_segments as _compute_single_edge_segments,
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
            child.set_figure(fig)

    @property
    def axes(self):
        return Line3DCollection.axes.__get__(self)

    @axes.setter
    def axes(self, new_axes):
        Line3DCollection.axes.__set__(self, new_axes)
        for child in self.get_children():
            child.axes = new_axes

    _get_adjacent_vertices_info = EdgeCollection._get_adjacent_vertices_info

    def _compute_edge_segments(self):
        """Compute the edge segments for all edges."""
        vinfo = self._get_adjacent_vertices_info()

        segments3d = []
        for vcoord_data in vinfo["offsets"]:
            segment = _compute_single_edge_segments(
                vcoord_data,
            )
            segments3d.append(segment)
        self.set_segments(segments3d)

    def _update_before_draw(self) -> None:
        """Update the collection before drawing."""
        if isinstance(self.axes, Axes3D) and hasattr(self, "do_3d_projection"):
            self.do_3d_projection()

        # TODO: Here's where we would shorten the edges to fit the vertex
        # projections from 3D onto 2D, if we wanted to do that. Because edges
        # in 3D are chains of segments rathen than splines, the shortening
        # needs to be done in a different way to how it's done in 2D.

    def draw(self, renderer) -> None:
        """Draw the collection of vertices in 3D.

        Parameters:
            renderer: The renderer to use for drawing.
        """
        # Prepare the collection for drawing
        self._update_before_draw()

        # Render the Line3DCollection
        # NOTE: we are NOT calling EdgeCollection.draw here
        super().draw(renderer)

        # This sets the labels offsets
        # TODO: implement labels in 3D (one could copy the function from 2D,
        # but would also need to promote the 2D labels into 3D labels similarly to
        # how it's done for 3D vertices).
        # self._update_labels()

        # Now attempt to draw the arrows
        for child in self.get_children():
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

    # NOTE: after this line, none of the EdgeCollection methods will work
    # It's become a static drawer now. It uses segments instead of paths.
    col.__class__ = Edge3DCollection
    col._compute_edge_segments()

    col._axlim_clip = axlim_clip

    # Convert the arrow collection if present
    if hasattr(col, "_arrows"):
        segments3d = col._segments3d

        # Fix the x and y to the center of the target vertex (for now)
        col._arrows._offsets[:] = [segment[-1][:2] for segment in segments3d]
        zs = [segment[-1][2] for segment in segments3d]
        arrow_collection_2d_to_3d(
            col._arrows,
            zs=zs,
            zdir=zdir,
            depthshade=False,
            axlim_clip=axlim_clip,
        )
