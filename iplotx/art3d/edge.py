"""
Module containing code to manipulate edge visualisations in 3D, especially the Edge3DCollection class.
"""

from mpl_toolkits.mplot3d.art3d import (
    Line3DCollection,
)

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
class Edge3DCollection(Line3DCollection):
    """Collection of vertex patches for plotting."""

    pass


def edge_collection_2d_to_3d(
    col: EdgeCollection,
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
