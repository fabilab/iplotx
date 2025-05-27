from typing import Union, Sequence
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.transforms import Affine2D
from pandas._libs.lib import is_scalar

from .typing import (
    GraphType,
    LayoutType,
)
from .importing import (
    igraph,
    networkx,
)
from .styles import (
    get_style,
    rotate_style,
)
from .heuristics import (
    network_library,
    normalise_layout,
    detect_directedness,
)
from .utils.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _get_label_width_height,
    _build_cmap_fun,
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
        self._ipx_internal_data = _create_internal_data(
            network,
            layout,
            vertex_labels=vertex_labels,
            edge_labels=edge_labels,
        )
        self._clear_state()

    def _clear_state(self):
        self._vertices = None
        self._edges = None
        self._vertex_labels = []
        self._edge_labels = []

    def get_children(self):
        artists = []
        # Collect edges first. This way vertices are on top of edges,
        # since vertices are drawn later. That is what most people expect.
        if self._edges is not None:
            artists.append(self._edges)
        if self._vertices is not None:
            artists.append(self._vertices)
        artists.extend(self._edge_labels)
        artists.extend(self._vertex_labels)
        return tuple(artists)

    def get_vertices(self):
        """Get VertexCollection artist."""
        return self._vertices

    def get_edges(self):
        """Get EdgeCollection artist."""
        return self._edges

    def get_vertex_labels(self):
        """Get list of vertex label artists."""
        return self._vertex_labels

    def get_edge_labels(self):
        """Get list of edge label artists."""
        return self._edge_labels

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
            for path, offset in zip(verts.get_paths(), verts._offsets):
                bbox = path.get_extents()
                mins = np.minimum(mins, trans_inv(bbox.min + trans(offset)))
                maxs = np.maximum(maxs, trans_inv(bbox.max + trans(offset)))

        ## FIXME: edge extents are currently broken
        # if self._edges is not None:
        #    for path in self._edges.get_paths():
        #        bbox = path.get_extents()
        #        mins = np.minimum(mins, bbox.min)
        #        maxs = np.maximum(maxs, bbox.max)

        if hasattr(self, "_groups") and self._groups is not None:
            for path in self._groups.get_paths():
                bbox = path.get_extents()
                mins = np.minimum(mins, bbox.min)
                maxs = np.maximum(maxs, bbox.max)

        # 5% padding, on each side
        pad = (maxs - mins) * pad
        mins -= pad
        maxs += pad

        return mpl.transforms.Bbox([mins, maxs])

    def _add_vertices(self):
        """Draw the vertices"""
        vertex_style = get_style(".vertex")

        layout_columns = [
            f"_ipx_layout_{i}" for i in range(self._ipx_internal_data["ndim"])
        ]
        vertex_layout_df = self._ipx_internal_data["vertex_df"][layout_columns]
        if vertex_style.get("size") == "label":
            if "label" not in self._ipx_internal_data["vertex_df"].columns:
                warnings.warn(
                    "No labels found, cannot resize vertices based on labels."
                )
                vertex_style["size"] = get_style("default.vertex")["size"]
            else:
                vertex_labels = self._ipx_internal_data["vertex_df"]["label"]

        # FIXME:: this would be better off in the VertexCollection itself, like we do for groups
        offsets = []
        patches = []
        for i, (vid, row) in enumerate(vertex_layout_df.iterrows()):
            # Centre of the vertex
            offsets.append(list(row[layout_columns].values))

            if vertex_style.get("size") == "label":
                # NOTE: it's ok to overwrite the dict here
                vertex_style["size"] = _get_label_width_height(
                    str(vertex_labels[vid]), **vertex_style.get("label", {})
                )

            vertex_stylei = rotate_style(vertex_style, index=i, id=vid)

            # Shape of the vertex (Patch)
            art = make_vertex_patch(**vertex_stylei)
            patches.append(art)

        art = VertexCollection(
            patches,
            offsets=offsets if offsets else None,
            offset_transform=self.axes.transData,
            transform=Affine2D(),
            match_original=True,
        )
        self._vertices = art

    def _add_vertex_labels(self):
        """Draw vertex labels."""
        label_style = get_style(".vertex.label")
        forbidden_props = ["hpadding", "vpadding"]
        for prop in forbidden_props:
            if prop in label_style:
                del label_style[prop]

        texts = []
        vertex_labels = self._ipx_internal_data["vertex_df"]["label"]
        for offset, label in zip(self._vertices._offsets, vertex_labels):
            text = mpl.text.Text(
                offset[0],
                offset[1],
                label,
                transform=self.axes.transData,
                **label_style,
            )
            texts.append(text)
        self._vertex_labels = texts

    def _add_edges(self):
        """Draw the edges.

        NOTE: UndirectedEdgeCollection and ArrowCollection are both subclasses of
        PatchCollection. When used with a cmap/norm, they set their facecolor
        according to the cmap, even though most likely we only want the edgecolor
        set that way. It can make for funny looking plots that are not uninteresting
        but mostly niche at this stage. Therefore we sidestep the whole cmap thing
        here.
        """
        if "label" in self._ipx_internal_data["edge_df"].columns:
            labels = self._ipx_internal_data["edge_df"]["label"]
        else:
            labels = None

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
        arrow_style = get_style(".arrow")

        layout_columns = [
            f"_ipx_layout_{i}" for i in range(self._ipx_internal_data["ndim"])
        ]
        vertex_layout_df = self._ipx_internal_data["vertex_df"][layout_columns]
        edge_df = self._ipx_internal_data["edge_df"].set_index(
            ["_ipx_source", "_ipx_target"]
        )

        # This contains the patches for vertices, for edge shortening and such
        vertex_paths = self._vertices._paths
        vertex_indices = pd.Series(
            np.arange(len(vertex_layout_df)), index=vertex_layout_df.index
        )

        if "cmap" in edge_style:
            edgearray = []
        edgepatches = []
        arrowpatches = []
        adjacent_vertex_ids = []
        adjecent_vertex_centers = []
        adjecent_vertex_paths = []
        for i, (vid1, vid2) in enumerate(edge_df.index):
            # Get the vertices for this edge
            vcenter1 = vertex_layout_df.loc[vid1, layout_columns].values
            vcenter2 = vertex_layout_df.loc[vid2, layout_columns].values
            vpath1 = vertex_paths[vertex_indices[vid1]]
            vpath2 = vertex_paths[vertex_indices[vid2]]

            edge_stylei = rotate_style(edge_style, index=i, id=(vid1, vid2))
            # NOTE: Not sure this is the best way but can't think of another one
            # Prioritise network internal styles
            _update_from_internal(edge_stylei, edge_df.iloc[i], kind="edge")
            if cmap_fun is not None:
                edgearray.append(edge_style["color"])
                edge_stylei["color"] = cmap_fun(edge_stylei["color"])

            # These are not the actual edges drawn, only stubs to establish
            # the styles which are then fed into the dynamic, optimised
            # factory (the collection) below
            patch = make_undirected_edge_patch(
                **edge_stylei,
            )
            edgepatches.append(patch)
            adjacent_vertex_ids.append((vid1, vid2))
            adjecent_vertex_centers.append((vcenter1, vcenter2))
            adjecent_vertex_paths.append((vpath1, vpath2))

            arrow_style["color"] = edge_stylei.get("color", "black")
            if "alpha" in edge_stylei:
                arrow_style["alpha"] = edge_stylei["alpha"]
            if "linewidth" in edge_stylei:
                arrow_style["linewidth"] = edge_stylei["linewidth"]
            arrow_patch = make_arrow_patch(
                **arrow_style,
            )
            arrowpatches.append(arrow_patch)

        adjecent_vertex_centers = np.array(adjecent_vertex_centers)
        # NOTE: the paths might have different number of sides, so it cannot be recast

        # TODO:: deal with "ports" a la graphviz

        if "cmap" in edge_style:
            vmin = np.min(edgearray)
            vmax = np.max(edgearray)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            edge_style["norm"] = norm

        art = DirectedEdgeCollection(
            edges=edgepatches,
            arrows=arrowpatches,
            labels=labels,
            vertex_ids=adjacent_vertex_ids,
            vertex_paths=adjecent_vertex_paths,
            vertex_centers=adjecent_vertex_centers,
            transform=self.axes.transData,
            style=edge_style,
        )
        self._edges = art

    def _add_undirected_edges(self, edge_style, labels=None, cmap_fun=None):
        """Draw undirected edges."""

        layout_columns = [
            f"_ipx_layout_{i}" for i in range(self._ipx_internal_data["ndim"])
        ]
        vertex_layout_df = self._ipx_internal_data["vertex_df"][layout_columns]
        edge_df = self._ipx_internal_data["edge_df"].set_index(
            ["_ipx_source", "_ipx_target"]
        )

        # This contains the patches for vertices, for edge shortening and such
        vertex_paths = self._vertices._paths
        vertex_indices = pd.Series(
            np.arange(len(vertex_layout_df)), index=vertex_layout_df.index
        )

        edgepatches = []
        adjacent_vertex_ids = []
        adjecent_vertex_centers = []
        adjecent_vertex_paths = []
        for i, (vid1, vid2) in enumerate(edge_df.index):
            # Get the vertices for this edge
            vcenter1 = vertex_layout_df.loc[vid1, layout_columns].values
            vcenter2 = vertex_layout_df.loc[vid2, layout_columns].values
            vpath1 = vertex_paths[vertex_indices[vid1]]
            vpath2 = vertex_paths[vertex_indices[vid2]]

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
            adjecent_vertex_centers.append((vcenter1, vcenter2))
            adjecent_vertex_paths.append((vpath1, vpath2))

        adjecent_vertex_centers = np.array(adjecent_vertex_centers)
        # NOTE: the paths might have different number of sides, so it cannot be recast

        # TODO:: deal with "ports" a la graphviz

        art = UndirectedEdgeCollection(
            edgepatches,
            labels=labels,
            vertex_ids=adjacent_vertex_ids,
            vertex_paths=adjecent_vertex_paths,
            vertex_centers=adjecent_vertex_centers,
            transform=self.axes.transData,
            style=edge_style,
        )
        self._edges = art

    def _process(self):
        self._clear_state()

        # TODO: some more things might be plotted before this

        # NOTE: we plot vertices first to get size etc. for edge shortening
        # but when the mpl engine runs down all children artists for actual
        # drawing it uses get_children() to get the order. Whatever is last
        # in that order will get drawn on top (vis-a-vis zorder).
        self._add_vertices()
        self._add_edges()
        if "label" in self._ipx_internal_data["vertex_df"].columns:
            self._add_vertex_labels()

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


# INTERNAL ROUTINES
def _create_internal_data(
    network,
    layout=None,
    vertex_labels=None,
    edge_labels=None,
):
    """Create internal data for the network."""
    nl = network_library(network)
    directed = detect_directedness(network)

    # Recast vertex_labels=False as vertex_labels=None
    if np.isscalar(vertex_labels) and (not vertex_labels):
        vertex_labels = None

    if nl == "networkx":
        # Vertices are indexed by node ID
        vertex_df = normalise_layout(
            layout,
            network=network,
        ).loc[pd.Index(network.nodes)]
        ndim = vertex_df.shape[1]
        vertex_df.columns = [f"_ipx_layout_{i}" for i in range(ndim)]

        # Vertex internal properties
        tmp = pd.DataFrame(dict(network.nodes.data())).T
        # Arrays become a single column, which we have already anyway
        if isinstance(layout, str) and (layout in tmp.columns):
            del tmp[layout]
        for col in tmp.columns:
            vertex_df[col] = tmp[col]
        del tmp

        # Vertex labels
        if vertex_labels is None:
            if "label" in vertex_df:
                del vertex_df["label"]
        else:
            if (
                np.isscalar(vertex_labels)
                and (not vertex_labels)
                and ("label" in vertex_df)
            ):
                del vertex_df["label"]
            elif vertex_labels is True:
                if "label" not in vertex_df:
                    vertex_df["label"] = vertex_df.index
            elif (not np.isscalar(vertex_labels)) and (
                len(vertex_labels) != len(vertex_df)
            ):
                raise ValueError(
                    "Vertex labels must be the same length as the number of vertices."
                )
            elif isinstance(vertex_labels, networkx.classes.reportviews.NodeDataView):
                vertex_df["label"] = pd.Series(dict(vertex_labels))
            else:
                vertex_df["label"] = vertex_labels

        # Edges are a list of tuples, because of multiedges
        tmp = []
        for u, v, d in network.edges.data():
            row = {"_ipx_source": u, "_ipx_target": v}
            row.update(d)
            tmp.append(row)
        if len(tmp):
            edge_df = pd.DataFrame(tmp)
        else:
            edge_df = pd.DataFrame(columns=["_ipx_source", "_ipx_target"])
        del tmp

        # Edge labels
        # Even though they could exist in the dataframe, request the to be explicitely mentioned
        if (edge_labels is None) and ("label" in edge_df):
            del edge_df["label"]
        elif edge_labels is not None:
            if np.isscalar(edge_labels):
                if (not edge_labels) and ("label" in edge_df):
                    del edge_df["label"]
                if (edge_labels is True) and ("label" not in edge_df):
                    edge_df["label"] = [str(i) for i in edge_df.index]
            else:
                if len(edge_labels) != len(edge_df):
                    raise ValueError(
                        "Edge labels must be the same length as the number of edges."
                    )
                edge_df["label"] = edge_labels

    else:
        # Vertices are ordered integers, no gaps
        vertex_df = normalise_layout(layout, network=network)
        ndim = vertex_df.shape[1]
        vertex_df.columns = [f"_ipx_layout_{i}" for i in range(ndim)]

        # Vertex labels
        if vertex_labels is not None:
            if len(vertex_labels) != len(vertex_df):
                raise ValueError(
                    "Vertex labels must be the same length as the number of vertices."
                )
            vertex_df["label"] = vertex_labels

        # Edges are a list of tuples, because of multiedges
        tmp = []
        for edge in network.es:
            row = {"_ipx_source": edge.source, "_ipx_target": edge.target}
            row.update(edge.attributes())
            tmp.append(row)
        if len(tmp):
            edge_df = pd.DataFrame(tmp)
        else:
            edge_df = pd.DataFrame(columns=["_ipx_source", "_ipx_target"])
        del tmp

        # Edge labels
        if edge_labels is not None:
            if len(edge_labels) != len(edge_df):
                raise ValueError(
                    "Edge labels must be the same length as the number of edges."
                )
            edge_df["label"] = edge_labels

    internal_data = {
        "vertex_df": vertex_df,
        "edge_df": edge_df,
        "directed": directed,
        "network_library": nl,
        "ndim": ndim,
    }
    return internal_data


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
