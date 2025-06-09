from typing import (
    Optional,
)

import numpy as np
import pandas as pd
import matplotlib as mpl

from .style import (
    get_style,
    rotate_style,
)
from .utils.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _build_cmap_fun,
)
from .ingest import (
    ingest_tree_data,
)
from .vertex import (
    VertexCollection,
)
from .edge import (
    EdgeCollection,
    make_stub_patch as make_undirected_edge_patch,
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
class TreeArtist(mpl.artist.Artist):
    """Artist for plotting trees."""

    def __init__(
        self,
        tree,
        layout="horizontal",
        direction="right",
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        offset_transform: Optional[mpl.transforms.Transform] = None,
    ):
        self.tree = tree

        self._ipx_internal_data = ingest_tree_data(
            tree,
            layout,
            vertex_labels=vertex_labels,
            edge_labels=edge_labels,
        )

        super().__init__()

        # This is usually the identity (which scales poorly with dpi)
        self.set_transform(transform)

        # This is usually transData
        self.set_offset_transform(offset_transform)

        zorder = get_style(".network").get("zorder", 1)
        self.set_zorder(zorder)

        self._add_vertices()
        self._add_edges()

    def get_children(self):
        return (self._vertices, self._edges)

    def set_figure(self, figure):
        super().set_figure(figure)
        for child in self.get_children():
            child.set_figure(figure)

    def get_offset_transform(self):
        """Get the offset transform (for vertices/edges)."""
        return self._offset_transform

    def set_offset_transform(self, offset_transform):
        """Set the offset transform (for vertices/edges)."""
        self._offset_transform = offset_transform

    def get_vertices(self):
        """Get VertexCollection artist."""
        return self._vertices

    def get_edges(self):
        """Get EdgeCollection artist."""
        return self._edges

    def get_vertex_labels(self):
        """Get list of vertex label artists."""
        return self._vertices.get_labels()

    def get_edge_labels(self):
        """Get list of edge label artists."""
        return self._edges.get_labels()

    def _add_vertices(self):
        """Add vertices to the tree."""
        self._vertices = VertexCollection(
            layout=self.get_layout(),
            offset_transform=self.get_offset_transform(),
            transform=self.get_transform(),
            style=get_style(".vertex"),
            labels=self._get_label_series("vertex"),
        )

    def _add_edges(self):
        """Add edges to the network artist.

        NOTE: UndirectedEdgeCollection and ArrowCollection are both subclasses of
        PatchCollection. When used with a cmap/norm, they set their facecolor
        according to the cmap, even though most likely we only want the edgecolor
        set that way. It can make for funny looking plots that are not uninteresting
        but mostly niche at this stage. Therefore we sidestep the whole cmap thing
        here.
        """

        labels = self._get_label_series("edge")
        edge_style = get_style(".edge")

        vertex_layout_df = self.get_layout()

        if "cmap" in edge_style:
            cmap_fun = _build_cmap_fun(
                edge_style["color"],
                edge_style["cmap"],
            )
        else:
            cmap_fun = None

        edge_df = self._ipx_internal_data["edge_df"].set_index(
            ["_ipx_source", "_ipx_target"]
        )

        if "cmap" in edge_style:
            colorarray = []
        edgepatches = []
        adjacent_vertex_ids = []
        for i, (vid1, vid2) in enumerate(edge_df.index):
            # Get the vertices for this edge

            edge_stylei = rotate_style(edge_style, index=i, id=(vid1, vid2))

            # FIXME:: Improve this logic. We have three layers of priority:
            # 1. Explicitely set in the style of "plot"
            # 2. Internal through network attributes
            # 3. Default styles
            # Because 1 and 3 are merged as a style context on the way in,
            # it's hard to squeeze 2 in the middle. For now, we will assume
            # the priority order is 2-1-3 instead (internal property is
            # highest priority).
            # This is also why we cannot shift this logic further into the
            # EdgeCollection class, which is oblivious of NetworkArtist's
            # internal data. In fact, one would argue this needs to be
            # pushed outwards to deal with the wrong ordering.
            _update_from_internal(edge_stylei, edge_df.iloc[i], kind="edge")

            if cmap_fun is not None:
                colorarray.append(edge_style["color"])
                edge_stylei["color"] = cmap_fun(edge_stylei["color"])

            # These are not the actual edges drawn, only stubs to establish
            # the styles which are then fed into the dynamic, optimised
            # factory (the collection) below
            patch = make_undirected_edge_patch(
                **edge_stylei,
            )
            edgepatches.append(patch)
            adjacent_vertex_ids.append((vid1, vid2))

        # TODO:: deal with "ports" a la graphviz

        if "cmap" in edge_style:
            vmin = np.min(colorarray)
            vmax = np.max(colorarray)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            edge_style["norm"] = norm

        self._edges = EdgeCollection(
            edgepatches,
            labels=labels,
            vertex_ids=adjacent_vertex_ids,
            vertex_collection=self._vertices,
            transform=self.get_offset_transform(),
            style=edge_style,
            directed=self._ipx_internal_data["directed"],
        )

    @_stale_wrapper
    def draw(self, renderer):
        super().draw(renderer)
        for child in self.get_children():
            child.draw(renderer)
