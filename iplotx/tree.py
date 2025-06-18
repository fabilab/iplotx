from typing import (
    Optional,
    Sequence,
)
from collections.abc import Hashable

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
    data_providers,
)
from .vertex import (
    VertexCollection,
)
from .edge import (
    EdgeCollection,
    make_stub_patch as make_undirected_edge_patch,
)
from .label import (
    LabelCollection,
)
from .network import (
    _update_from_internal,
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
        orientation="right",
        directed: bool | str = False,
        vertex_labels: Optional[
            bool | list[str] | dict[Hashable, str] | pd.Series
        ] = None,
        edge_labels: Optional[Sequence] = None,
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        offset_transform: Optional[mpl.transforms.Transform] = None,
    ):
        """Initialize the TreeArtist.

        Parameters:
            tree: The tree to plot.
            layout: The layout to use for the tree. Can be "horizontal", "vertical", or "radial".
            orientation: The orientation of the tree layout. Can be "right" or "left" (for
                horizontal and radial layouts) and "descending" or "ascending" (for vertical
                layouts).
            directed: Whether the tree is directed. Can be a boolean or a string with the
                following choices: "parent" or "child".
            vertex_labels: Labels for the vertices. Can be a list, dictionary, or pandas Series.
            edge_labels: Labels for the edges. Can be a sequence of strings.
            transform: The transform to apply to the tree artist. This is usually the identity.
            offset_transform: The offset transform to apply to the tree artist. This is
                usually `ax.transData`.
        """

        self.tree = tree
        self._ipx_internal_data = ingest_tree_data(
            tree,
            layout,
            orientation=orientation,
            directed=directed,
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

        if "cascading" in self.get_vertices().get_style():
            self._add_cascades()

    def get_children(self) -> tuple[mpl.artist.Artist]:
        """Get the children of this artist.

        Returns:
            The artists for vertices and edges.
        """
        children = [self._vertices, self._edges]
        if hasattr(self, "_cascades"):
            children.append(self._cascades)
        return tuple(children)

    def set_figure(self, fig) -> None:
        """Set the figure for this artist and its children.

        Parameters:
            fig: the figure to set for this artist and its children.
        """
        super().set_figure(fig)
        for child in self.get_children():
            child.set_figure(fig)

    def get_offset_transform(self):
        """Get the offset transform (for vertices/edges)."""
        return self._offset_transform

    def set_offset_transform(self, offset_transform):
        """Set the offset transform (for vertices/edges)."""
        self._offset_transform = offset_transform

    def get_layout(self, kind="vertex"):
        """Get vertex or edge layout."""
        layout_columns = [
            f"_ipx_layout_{i}" for i in range(self._ipx_internal_data["ndim"])
        ]

        if kind == "vertex":
            layout = self._ipx_internal_data["vertex_df"][layout_columns]
            return layout

        elif kind == "edge":
            return self._ipx_internal_data["edge_df"][layout_columns]
        else:
            raise ValueError(f"Unknown layout kind: {kind}. Use 'vertex' or 'edge'.")

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

        bbox = self._vertices.get_datalim(transData)

        edge_bbox = self._edges.get_datalim(transData)
        bbox = mpl.transforms.Bbox.union([bbox, edge_bbox])

        bbox = bbox.expanded(sw=(1.0 + pad), sh=(1.0 + pad))
        return bbox

    def _get_label_series(self, kind: str) -> Optional[pd.Series]:
        if "label" in self._ipx_internal_data[f"{kind}_df"].columns:
            return self._ipx_internal_data[f"{kind}_df"]["label"]
        else:
            return None

    def get_vertices(self) -> VertexCollection:
        """Get VertexCollection artist."""
        return self._vertices

    def get_edges(self) -> EdgeCollection:
        """Get EdgeCollection artist."""
        return self._edges

    def get_vertex_labels(self) -> LabelCollection:
        """Get list of vertex label artists."""
        return self._vertices.get_labels()

    def get_edge_labels(self) -> LabelCollection:
        """Get list of edge label artists."""
        return self._edges.get_labels()

    def _add_vertices(self) -> None:
        """Add vertices to the tree."""
        self._vertices = VertexCollection(
            layout=self.get_layout(),
            layout_coordinate_system=self._ipx_internal_data.get(
                "layout_coordinate_system",
                "catesian",
            ),
            style=get_style(".vertex"),
            labels=self._get_label_series("vertex"),
            transform=self.get_transform(),
            offset_transform=self.get_offset_transform(),
        )

    def _add_cascades(self) -> None:
        """Add cascading patches."""

        # NOTE: there is a weird bug in pandas when using generic Hashable-s
        # with .loc. Seems like doing .T[...] works for individual index
        # elements only though
        def get_node_coords(node):
            return self._vertices.get_layout().T[node].values

        def get_leaves_coords(leaves):
            return np.array(
                [get_node_coords(leaf) for leaf in leaves],
            )

        style = self.get_vertices().get_style()["cascading"]
        provider = data_providers["tree"][self._ipx_internal_data["tree_library"]]

        # FIXME: get the nodes to draw patches from as a single dictionary/list
        # instead of from facecolor
        nodes_unordered = list(style["facecolor"].keys())

        # Draw the patches from the closest to the root (earlier drawing)
        # to the closer to the leaves (later drawing).
        drawing_order = []
        for node in provider(self.tree).preorder():
            if node in nodes_unordered:
                drawing_order.append(node)

        layout_name = self._ipx_internal_data["layout_name"]
        orientation = self._ipx_internal_data["orientation"]
        extend = style.get("extend", False)
        if layout_name not in ("horizontal", "vertical"):
            raise NotImplementedError(
                "Cascading patches not implemented for radial layout.",
            )

        if layout_name == "horizontal":
            if orientation == "right":
                depth_max = self.get_vertices().get_layout().values[:, 0].max()
            else:
                depth_max = self.get_vertices().get_layout().values[:, 0].min()
        elif layout_name == "vertical":
            if orientation == "descending":
                depth_max = self.get_vertices().get_layout().values[:, 1].min()
            else:
                depth_max = self.get_vertices().get_layout().values[:, 1].max()
        elif layout_name == "radial":
            # layout values are: r, theta
            depth_max = self.get_vertices().get_layout().values[:, 0].max()

        cascading_patches = []
        for node in drawing_order:
            stylei = rotate_style(style, key=node)
            del stylei["extend"]
            provider_node = provider(node)

            bl = provider_node.get_branch_length_default_to_one(node)
            node_coords = get_node_coords(node).copy()
            leaves_coords = get_leaves_coords(provider_node.get_leaves())
            if len(leaves_coords) == 0:
                leaves_coords = np.array([node_coords])

            if layout_name in ("horizontal", "vertical"):
                if layout_name == "horizontal":
                    ybot = leaves_coords[:, 1].min() - 0.5
                    ytop = leaves_coords[:, 1].max() + 0.5
                    if orientation == "right":
                        xleft = node_coords[0] - bl
                        xright = depth_max if extend else leaves_coords[:, 0].max()
                    else:
                        xleft = depth_max if extend else leaves_coords[:, 0].min()
                        xright = node_coords[0] + bl
                elif layout_name == "vertical":
                    xleft = leaves_coords[:, 0].min() - 0.5
                    xright = leaves_coords[:, 0].max() + 0.5
                    if orientation == "descending":
                        ytop = node_coords[1] + bl
                        ybot = depth_max if extend else leaves_coords[:, 1].min()
                    else:
                        ytop = depth_max if extend else leaves_coords[:, 1].max()
                        ybot = node_coords[1] - bl

                patch = mpl.patches.Rectangle(
                    (xleft, ybot),
                    xright - xleft,
                    ytop - ybot,
                    **stylei,
                )
            elif layout_name == "radial":
                points = []
                path = mpl.path.Path(
                    points,
                    codes=[getattr(mpl.path.Path, code) for code in codes],
                )
                patch = mpl.patches.PathPath(
                    path,
                    **stylei,
                )

            cascading_patches.append(patch)

        self._cascades = mpl.collections.PatchCollection(
            cascading_patches,
            transform=self.get_offset_transform(),
            match_original=True,
        )

    def _add_edges(self) -> None:
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
        waypoints = []
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

            # Tree layout determines waypoints
            waypointsi = edge_stylei.pop("waypoints", None)
            if waypointsi is None:
                layout_name = self._ipx_internal_data["layout_name"]
                if layout_name == "horizontal":
                    waypointsi = "x0y1"
                elif layout_name == "vertical":
                    waypointsi = "y0y0"
                elif layout_name == "radial":
                    waypointsi = "r0a1"
                else:
                    waypointsi = "none"
            waypoints.append(waypointsi)

            # These are not the actual edges drawn, only stubs to establish
            # the styles which are then fed into the dynamic, optimised
            # factory (the collection) below
            patch = make_undirected_edge_patch(
                **edge_stylei,
            )
            edgepatches.append(patch)
            adjacent_vertex_ids.append((vid1, vid2))

        if "cmap" in edge_style:
            vmin = np.min(colorarray)
            vmax = np.max(colorarray)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            edge_style["norm"] = norm

        edge_style["waypoints"] = waypoints

        # NOTE: Trees are directed is their "directed" property is True, "child", or "parent"
        self._edges = EdgeCollection(
            edgepatches,
            vertex_ids=adjacent_vertex_ids,
            vertex_collection=self._vertices,
            labels=labels,
            transform=self.get_offset_transform(),
            style=edge_style,
            directed=bool(self._ipx_internal_data["directed"]),
        )
        if "cmap" in edge_style:
            self._edges.set_array(colorarray)

    @_stale_wrapper
    def draw(self, renderer) -> None:
        """Draw each of the children, with some buffering mechanism."""
        if not self.get_visible():
            return

        # FIXME: Callbacks on stale vertices/edges??

        # NOTE: looks like we have to manage the zorder ourselves
        # this is kind of funny actually
        children = list(self.get_children())
        children.sort(key=lambda x: x.zorder)
        for art in children:
            art.draw(renderer)
