from typing import (
    Optional,
    Sequence,
    Self,
)
import numpy as np
import pandas as pd
import matplotlib as mpl

from ..typing import (
    GraphType,
    LayoutType,
)
from ..style import (
    get_style,
    rotate_style,
)
from ..utils.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _build_cmap_fun,
)
from ..ingest import (
    ingest_network_data,
)
from ..vertex import (
    VertexCollection,
)
from ..edge import (
    EdgeCollection,
    make_stub_patch as make_undirected_edge_patch,
)
from ..art3d.vertex import (
    vertex_collection_2d_to_3d,
)
from ..art3d.edge import (
    Edge3DCollection,
    edge_collection_2d_to_3d,
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
class NetworkArtist(mpl.artist.Artist):
    def __init__(
        self,
        network: GraphType,
        layout: Optional[LayoutType] = None,
        vertex_labels: Optional[list | dict | pd.Series] = None,
        edge_labels: Optional[Sequence] = None,
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        offset_transform: Optional[mpl.transforms.Transform] = None,
    ):
        """Network container artist that groups all plotting elements.

        Parameters:
            network: The network to plot.
            layout: The layout of the network. If None, this function will attempt to
                infer the layout from the network metadata, using heuristics. If that fails, an
                exception will be raised.
            vertex_labels: The labels for the vertices. If None, no vertex labels
                will be drawn. If a list, the labels are taken from the list. If a dict, the keys
                should be the vertex IDs and the values should be the labels.
            edge_labels: The labels for the edges. If None, no edge labels will be drawn.
            transform: The transform to use for the vertices. Default is IdentityTransform.
            offset_transform: The transform to use as offset transform for the vertices and main
                transform for the edges. Default is None, but this should eventually be set to
                ax.transData once the artist is added to an Axes.

        """
        self.network = network

        super().__init__()

        # This is usually the identity (which scales poorly with dpi)
        self.set_transform(transform)

        # This is usually transData
        self.set_offset_transform(offset_transform)

        zorder = get_style(".network").get("zorder", 1)
        self.set_zorder(zorder)

        if network is not None:
            self._ipx_internal_data = ingest_network_data(
                network,
                layout,
                vertex_labels=vertex_labels,
                edge_labels=edge_labels,
            )

            self._add_vertices()
            self._add_edges()

    @classmethod
    def from_other(
        cls: "NetworkArtist",  # NOTE: This is fixed in Python 3.14
        other: Self,
    ) -> Self:
        """Create a NetworkArtist as a copy of another one.

        Parameters:
            other: The other NetworkArtist.

        Returns:
            An instantiated NetworkArtist.
        """
        self = cls.from_edgecollection(other._edges)
        self.network = other.network
        if hasattr(other, "_ipx_internal_data"):
            self._ipx_internal_data = other._ipx_internal_data
        return self

    @classmethod
    def from_edgecollection(
        cls: "NetworkArtist",  # NOTE: This is fixed in Python 3.14
        edge_collection: EdgeCollection | Edge3DCollection,
    ) -> Self:
        """Create a NetworkArtist from iplotx artists.

        Parameters:
            edge_collection: The edge collection to use to initialise the artist. Vertices will
              be obtained automatically.

        Returns:
            The initialised NetworkArtist.
        """
        vertex_collection = edge_collection._vertex_collection
        layout = vertex_collection._layout
        transform = vertex_collection.get_transform()
        offset_transform = vertex_collection.get_offset_transform()

        # Follow the steps in the normal constructor
        self = cls(
            network=None,
            layout=layout,
            transform=transform,
            offset_transform=offset_transform,
        )
        # TODO: should we make copies here?
        self._vertices = vertex_collection
        self._edges = edge_collection

        return self

    def get_children(self):
        if hasattr(self, "_edges"):
            return (self._vertices, self._edges)
        else:
            return (self._vertices,)

    def set_figure(self, fig):
        super().set_figure(fig)
        for child in self.get_children():
            child.set_figure(fig)

    @property
    def axes(self):
        return mpl.artist.Artist.axes.__get__(self)

    @axes.setter
    def axes(self, new_axes):
        mpl.artist.Artist.axes.__set__(self, new_axes)
        for child in self.get_children():
            child.axes = new_axes
        self.set_figure(new_axes.figure)

    def get_offset_transform(self):
        """Get the offset transform (for vertices/edges)."""
        return self._offset_transform

    def set_offset_transform(self, offset_transform):
        """Set the offset transform (for vertices/edges)."""
        self._offset_transform = offset_transform
        if hasattr(self, "_vertices"):
            self._vertices.set_offset_transform(offset_transform)
        if hasattr(self, "_edges"):
            self._edges.set_transform(offset_transform)

    def get_vertices(self):
        """Get VertexCollection artist."""
        return self._vertices

    get_nodes = get_vertices

    def get_edges(self):
        """Get EdgeCollection artist."""
        return self._edges

    def get_vertex_labels(self):
        """Get list of vertex label artists."""
        return self._vertices.get_labels()

    get_node_labels = get_vertex_labels

    def get_edge_labels(self):
        """Get list of edge label artists."""
        return self._edges.get_labels()

    def get_datalim(self, transData, pad=0.15):
        """Get limits on x/y axes based on the graph layout data.

        Parameters:
            transData (Transform): The transform to use for the data.
            pad (float): Padding to add to the limits. Default is 0.05.
                Units are a fraction of total axis range before padding.
        """
        layout = self.get_layout().values

        if len(layout) == 0:
            return mpl.transforms.Bbox([[0, 0], [1, 1]])

        bboxes = [
            self._vertices.get_datalim(transData),
        ]
        if hasattr(self, "_edges"):
            bboxes.append(
                self._edges.get_datalim(transData),
            )
        bbox = mpl.transforms.Bbox.union(bboxes)

        bbox = bbox.expanded(sw=(1.0 + pad), sh=(1.0 + pad))
        return bbox

    def autoscale_view(self, tight=False):
        """Recompute data limits from this artist and set autoscale based on them."""
        bbox = self.get_datalim(self.axes.transData)
        self.axes.update_datalim(bbox)
        self.axes.autoscale_view(tight=tight)

    def get_layout(self):
        """Get the vertex layout.

        Returns:
            The vertex layout as a DataFrame.
        """
        layout_columns = [f"_ipx_layout_{i}" for i in range(self.get_ndim())]
        vertex_layout_df = self._ipx_internal_data["vertex_df"][layout_columns]
        return vertex_layout_df

    def get_ndim(self):
        """Get the dimensionality of the layout.

        Returns:
            The dimensionality of the layout (2 or 3).
        """
        return self._ipx_internal_data["ndim"]

    def _get_label_series(self, kind):
        # Equivalence vertex/node
        if kind == "node":
            kind = "vertex"

        if "label" in self._ipx_internal_data[f"{kind}_df"].columns:
            return self._ipx_internal_data[f"{kind}_df"]["label"]
        else:
            return None

    def _add_vertices(self):
        """Add vertices to the network artist."""

        self._vertices = VertexCollection(
            layout=self.get_layout(),
            layout_coordinate_system=self._ipx_internal_data.get(
                "layout_coordinate_system", "cartesian"
            ),
            style=get_style(".vertex"),
            labels=self._get_label_series("vertex"),
            transform=self.get_transform(),
            offset_transform=self.get_offset_transform(),
        )

        if self.get_ndim() == 3:
            depthshade = get_style(".vertex").get("depthshade", True)
            vertex_collection_2d_to_3d(
                self._vertices,
                zs=self.get_layout().iloc[:, 2].values,
                depthshade=depthshade,
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

        edge_df = self._ipx_internal_data["edge_df"].set_index(["_ipx_source", "_ipx_target"])

        if len(edge_df) == 0:
            return

        if "cmap" in edge_style:
            cmap_fun = _build_cmap_fun(
                edge_style,
                "color",
                edge_style.get("norm", None),
                internal=edge_df,
            )
        else:
            cmap_fun = None

        if "cmap" in edge_style:
            colorarray = []
        edgepatches = []
        adjacent_vertex_ids = []
        for i, (vid1, vid2) in enumerate(edge_df.index):
            edge_stylei = rotate_style(edge_style, index=i, key=(vid1, vid2))

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
                colorarray.append(edge_stylei["color"])
                edge_stylei["color"] = cmap_fun(edge_stylei["color"])

            # These are not the actual edges drawn, only stubs to establish
            # the styles which are then fed into the dynamic, optimised
            # factory (the collection) below
            patch = make_undirected_edge_patch(
                **edge_stylei,
            )
            edgepatches.append(patch)
            adjacent_vertex_ids.append((vid1, vid2))

        if ("cmap" in edge_style) and ("norm" not in edge_style):
            vmin = np.min(colorarray)
            vmax = np.max(colorarray)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            edge_style["norm"] = norm

        self._edges = EdgeCollection(
            edgepatches,
            vertex_ids=adjacent_vertex_ids,
            vertex_collection=self._vertices,
            labels=labels,
            transform=self.get_offset_transform(),
            style=edge_style,
            directed=self._ipx_internal_data["directed"],
        )
        if "cmap" in edge_style:
            self._edges.set_array(colorarray)

        if self.get_ndim() == 3:
            depthshade = get_style(".edge").get("depthshade", True)
            edge_collection_2d_to_3d(
                self._edges,
                depthshade=depthshade,
            )

    @_stale_wrapper
    def draw(self, renderer):
        """Draw each of the children, with some buffering mechanism."""
        if not self.get_children():
            self._add_vertices()
            self._add_edges()

        if not self.get_visible():
            return

        # Handle zorder manually, just like in AxesBase in mpl
        children = list(self.get_children())
        children.sort(key=lambda x: x.zorder)
        for child in children:
            child.draw(renderer)


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
