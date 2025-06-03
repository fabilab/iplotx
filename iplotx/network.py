from typing import Union, Sequence
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
from pandas._libs.lib import is_scalar

from .typing import (
    GraphType,
    LayoutType,
)
from .style import (
    get_style,
    rotate_style,
)
from .utils.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _get_label_width_height,
    _build_cmap_fun,
)
from .ingest import (
    ingest_network_data,
)
from .vertex import (
    VertexCollection,
    make_patch as make_vertex_patch,
)
from .edge.undirected import (
    UndirectedEdgeCollection,
    make_stub_patch as make_undirected_edge_patch,
)
from .edge.directed import (
    DirectedEdgeCollection,
    make_arrow_patch,
)


@_forwarder(
    (
        "set_clip_path",
        "set_clip_box",
        "set_transform",
        "set_snap",
        "set_sketch_params",
        "set_figure",
        "set_animated",
        "set_picker",
    )
)
class NetworkArtist(mpl.artist.Artist):
    def __init__(
        self,
        network: GraphType,
        layout: LayoutType = None,
        vertex_labels: Union[None, list, dict, pd.Series] = None,
        edge_labels: Union[None, Sequence] = None,
    ):
        """Network container artist that groups all plotting elements.

        Parameters:
            network (networkx.Graph or igraph.Graph): The network to plot.
            layout (array-like): The layout of the network. If None, this function will attempt to
                infer the layout from the network metadata, using heuristics. If that fails, an
                exception will be raised.
            vertex_labels (list, dict, or pandas.Series): The labels for the vertices. If None, no vertex labels
                will be drawn. If a list, the labels are taken from the list. If a dict, the keys
                should be the vertex IDs and the values should be the labels.
            elge_labels (sequence): The labels for the edges. If None, no edge labels will be drawn.
        """
        super().__init__()

        self.network = network
        self._ipx_internal_data = ingest_network_data(
            network,
            layout,
            vertex_labels=vertex_labels,
            edge_labels=edge_labels,
        )
        self._clear_state()

    def _clear_state(self):
        self._vertices = None
        self._edges = None
        self._groups = None

    def get_children(self):
        artists = []
        # Collect edges first. This way vertices are on top of edges,
        # since vertices are drawn later. That is what most people expect.
        if self._edges is not None:
            artists.append(self._edges)
        if self._vertices is not None:
            artists.append(self._vertices)
        return tuple(artists)

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

    def get_datalim(self, transData, pad=0.05):
        """Get limits on x/y axes based on the graph layout data.

        Parameters:
            transData (Transform): The transform to use for the data.
            pad (float): Padding to add to the limits. Default is 0.05.
                Units are a fraction of total axis range before padding.
        """
        # FIXME: transData works here, but it's probably kind of broken in general
        import numpy as np

        layout_columns = [
            f"_ipx_layout_{i}" for i in range(self._ipx_internal_data["ndim"])
        ]
        layout = self._ipx_internal_data["vertex_df"][layout_columns].values

        if len(layout) == 0:
            mins = np.array([0, 0])
            maxs = np.array([1, 1])
            return mpl.transforms.Bbox([mins, maxs])

        # Use the layout as a base, and expand using bboxes from other artists
        mins = np.min(layout, axis=0).astype(float)
        maxs = np.max(layout, axis=0).astype(float)

        # NOTE: unlike other Collections, the vertices are basically a
        # PatchCollection with an offset transform using transData. Therefore,
        # care should be taken if one wants to include it here
        if self._vertices is not None:
            trans = transData.transform
            trans_inv = transData.inverted().transform
            verts = self._vertices
            for size, offset in zip(verts.get_sizes(), verts.get_offsets()):
                imax = trans_inv(trans(offset) + size)
                imin = trans_inv(trans(offset) - size)
                mins = np.minimum(mins, imin)
                maxs = np.maximum(maxs, imax)

        if self._edges is not None:
            for path in self._edges.get_paths():
                bbox = path.get_extents()
                mins = np.minimum(mins, bbox.min)
                maxs = np.maximum(maxs, bbox.max)

        if self._groups is not None:
            for path in self._groups.get_paths():
                bbox = path.get_extents()
                mins = np.minimum(mins, bbox.min)
                maxs = np.maximum(maxs, bbox.max)

        # 5% padding, on each side
        pad = (maxs - mins) * pad
        mins -= pad
        maxs += pad

        return mpl.transforms.Bbox([mins, maxs])

    def _get_layout_dataframe(self):
        layout_columns = [
            f"_ipx_layout_{i}" for i in range(self._ipx_internal_data["ndim"])
        ]
        vertex_layout_df = self._ipx_internal_data["vertex_df"][layout_columns]
        return vertex_layout_df

    def _get_label_series(self, kind):
        if "label" in self._ipx_internal_data[f"{kind}_df"].columns:
            labels = self._ipx_internal_data[f"{kind}_df"]["label"]
        else:
            return None

    def _add_vertices(self):
        """Draw the vertices"""

        self._vertices = VertexCollection(
            layout=self._get_layout_dataframe(),
            offset_transform=self.axes.transData,
            transform=mpl.transforms.IdentityTransform(),
            style=get_style(".vertex"),
            labels=self._get_label_series("vertex"),
        )

    def _add_edges(self):
        """Draw the edges.

        NOTE: UndirectedEdgeCollection and ArrowCollection are both subclasses of
        PatchCollection. When used with a cmap/norm, they set their facecolor
        according to the cmap, even though most likely we only want the edgecolor
        set that way. It can make for funny looking plots that are not uninteresting
        but mostly niche at this stage. Therefore we sidestep the whole cmap thing
        here.
        """

        labels = self._get_label_series("edge")
        edge_style = get_style(".edge")
        if "cmap" in edge_style:
            cmap_fun = _build_cmap_fun(
                edge_style["color"],
                edge_style["cmap"],
            )
        else:
            cmap_fun = None

        if self._ipx_internal_data["directed"]:
            return self._add_directed_edges(
                labels=labels,
                edge_style=edge_style,
                cmap_fun=cmap_fun,
            )
        return self._add_undirected_edges(
            labels=labels,
            edge_style=edge_style,
            cmap_fun=cmap_fun,
        )

    def _add_directed_edges(self, edge_style, labels=None, cmap_fun=None):
        """Draw directed edges."""
        vertex_layout_df = self._get_layout_dataframe()
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
            # NOTE: Not sure this is the best way but can't think of another one
            # Prioritise network internal styles
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

        self._edges = DirectedEdgeCollection(
            edges=edgepatches,
            labels=labels,
            vertex_ids=adjacent_vertex_ids,
            vertex_collection=self._vertices,
            transform=self.axes.transData,
            style=edge_style,
        )

    def _add_undirected_edges(self, edge_style, labels=None, cmap_fun=None):
        """Draw undirected edges."""

        vertex_layout_df = self._get_layout_dataframe()
        edge_df = self._ipx_internal_data["edge_df"].set_index(
            ["_ipx_source", "_ipx_target"]
        )

        edgepatches = []
        adjacent_vertex_ids = []
        for i, (vid1, vid2) in enumerate(edge_df.index):
            # Get the vertices for this edge
            edge_stylei = rotate_style(edge_style, index=i, id=(vid1, vid2))
            # NOTE: Not sure this is the best way but can't think of another one
            # Prioritise network internal styles
            _update_from_internal(edge_stylei, edge_df.iloc[i], kind="edge")

            if cmap_fun is not None:
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

        self._edges = UndirectedEdgeCollection(
            patches=edgepatches,
            labels=labels,
            vertex_ids=adjacent_vertex_ids,
            vertex_collection=self._vertices,
            transform=self.axes.transData,
            style=edge_style,
        )

    def _process(self):
        self._clear_state()

        # TODO: some more things might be plotted before this

        # NOTE: we plot vertices first to get size etc. for edge shortening
        # but when the mpl engine runs down all children artists for actual
        # drawing it uses get_children() to get the order. Whatever is last
        # in that order will get drawn on top (vis-a-vis zorder).
        self._add_vertices()
        self._add_edges()

        # TODO: callbacks for stale vertices/edges

        # Forward mpl properties to children
        # TODO sort out all of the things that need to be forwarded
        for child in self.get_children():
            # set the figure & axes on child, this ensures each artist
            # down the hierarchy knows where to draw
            if hasattr(child, "set_figure"):
                child.set_figure(self.figure)
            child.axes = self.axes

            # forward the clippath/box to the children need this logic
            # because mpl exposes some fast-path logic
            clip_path = self.get_clip_path()
            if clip_path is None:
                clip_box = self.get_clip_box()
                child.set_clip_box(clip_box)
            else:
                child.set_clip_path(clip_path)

    @_stale_wrapper
    def draw(self, renderer, *args, **kwds):
        """Draw each of the children, with some buffering mechanism."""
        if not self.get_visible():
            return

        if not self.get_children():
            self._process()

        # NOTE: looks like we have to manage the zorder ourselves
        # this is kind of funny actually
        children = list(self.get_children())
        children.sort(key=lambda x: x.zorder)
        for art in children:
            art.draw(renderer, *args, **kwds)


def _update_from_internal(style, row, kind):
    """Update single vertex/edge style from internal data."""
    if "color" in row:
        style["color"] = row["color"]
    if "facecolor" in row:
        style["facecolor"] = row["facecolor"]
    if "edgecolor" in row:
        if kind == "vertex":
            style["edgecolor"] = row["edgecolor"]
        else:
            style["color"] = row["edgecolor"]

    if "linewidth" in row:
        style["linewidth"] = row["linewidth"]
    if "linestyle" in row:
        style["linestyle"] = row["linestyle"]
    if "alpha" in row:
        style["alpha"] = row["alpha"]
