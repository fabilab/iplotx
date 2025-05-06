import pandas as pd
import matplotlib as mpl
from matplotlib.transforms import Affine2D

from .styles import (
    get_style,
    get_stylename,
)
from .heuristics import (
    network_library,
    normalise_layout,
    detect_directedness,
)
from .tools.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _additional_set_methods,
)
from .vertex import (
    VertexCollection,
    make_patch as make_vertex_patch,
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
    def __init__(self, network, layout=None):
        super().__init__()

        self.network = network
        self._ipx_internal_data = _create_internal_data(network, layout)
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

    def get_datalim(self, pad=0.05):
        """Get limits on x/y axes based on the graph layout data.

        Parameters:
            pad (float): Padding to add to the limits. Default is 0.05.
                Units are a fraction of total axis range before padding.
        """
        import numpy as np

        layout_columns = [f"_ipx_layout_{i}" for i in range(self._ipx_internal_data["ndim"])]
        layout = self._ipx_internal_data["vertex_df"].values

        if len(layout) == 0:
            mins = np.array([0, 0])
            maxs = np.array([1, 1])
            return (mins, maxs)

        # Use the layout as a base, and expand using bboxes from other artists
        mins = np.min(layout, axis=0).astype(float)
        maxs = np.max(layout, axis=0).astype(float)

        # NOTE: unlike other Collections, the vertices are basically a
        # PatchCollection with an offset transform using transData. Therefore,
        # care should be taken if one wants to include it here
        if self._vertices is not None:
            trans = self.axes.transData.transform
            trans_inv = self.axes.transData.inverted().transform
            verts = self._vertices
            for path, offset in zip(verts.get_paths(), verts._offsets):
                bbox = path.get_extents()
                mins = np.minimum(mins, trans_inv(bbox.min + trans(offset)))
                maxs = np.maximum(maxs, trans_inv(bbox.max + trans(offset)))

        if self._edges is not None:
            for path in self._edges.get_paths():
                bbox = path.get_extents()
                mins = np.minimum(mins, bbox.min)
                maxs = np.maximum(maxs, bbox.max)

        if hasattr(self, "_groups") and self._groups is not None:
            for path in self._groups.get_paths():
                bbox = path.get_extents()
                mins = np.minimum(mins, bbox.min)
                maxs = np.maximum(maxs, bbox.max)

        # 5% padding, on each side
        pad = (maxs - mins) * pad
        mins -= pad
        maxs += pad

        return (mins, maxs)

    def _draw_vertices(self):
        """Draw the vertices"""
        vertex_style = get_style(get_stylename()+".vertex")

        layout_columns = [f"_ipx_layout_{i}" for i in range(self._ipx_internal_data["ndim"])]
        vertex_layout_df = self._ipx_internal_data["vertex_df"][layout_columns]

        # TODO:
        offsets = []
        patches = []
        for vid, row in vertex_layout_df.iterrows():
            # Centre of the vertex
            offsets.append(list(row[layout_columns].values))

            # Shape of the vertex (Patch)
            # FIXME: this needs to be rescaled in figure points, not data points
            art = make_vertex_patch(**vertex_style)
            patches.append(art)

        art = VertexCollection(
            patches,
            offsets=offsets if offsets else None,
            offset_transform=self.axes.transData,
            match_original=True,
            transform=Affine2D(),
        )
        self._vertices = art

    #def _draw_edges(self):
    #    """Draw the edges"""
    #    graph = self.graph
    #    vertex_builder = self._vertex_builder
    #    edge_drawer = self._edge_drawer
    #    edge_builder = self._edge_builder
    #    edge_order = self._edge_order

    #    es = graph.es
    #    if edge_order is None:
    #        # Default edge order
    #        edge_coord_iter = zip(es, edge_builder)
    #    else:
    #        # Specified edge order
    #        edge_coord_iter = ((es[i], edge_builder[i]) for i in edge_order)

    #    directed = graph.is_directed()

    #    visual_vertices = []
    #    edgepatches = []
    #    arrow_sizes = []
    #    arrow_widths = []
    #    loop_sizes = []
    #    curved = []
    #    for edge, visual_edge in edge_coord_iter:
    #        edge_vertices = [vertex_builder[v] for v in edge.tuple]
    #        art = edge_drawer.build_patch(visual_edge, *edge_vertices)
    #        edgepatches.append(art)
    #        visual_vertices.append(edge_vertices)
    #        arrow_sizes.append(visual_edge.arrow_size)
    #        arrow_widths.append(visual_edge.arrow_width)
    #        loop_sizes.append(visual_edge.loop_size)
    #        curved.append(visual_edge.curved)

    #    art = EdgeCollection(
    #        edgepatches,
    #        visual_vertices=visual_vertices,
    #        directed=directed,
    #        arrow_sizes=arrow_sizes,
    #        arrow_widths=arrow_widths,
    #        loop_sizes=loop_sizes,
    #        curved=curved,
    #        transform=self.axes.transData,
    #    )
    #    self._edges = art

    def _process(self):
        self._clear_state()

        # TODO: some more things might be plotted before this

        # NOTE: we plot vertices first to get size etc. for edge shortening
        # but when the mpl engine runs down all children artists for actual
        # drawing it uses get_children() to get the order. Whatever is last
        # in that order will get drawn on top (vis-a-vis zorder).
        self._draw_vertices()
        #self._draw_edges()
        #self._draw_vertex_labels()
        #self._draw_edge_labels()

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
def _create_internal_data(network, layout=None):
    """Create internal data for the network."""
    nl = network_library(network)
    directed = detect_directedness(network)

    if nl == "networkx":
        # Vertices are indexed by node ID
        # NOTE: unclear we need all these data for just plotting
        vertex_df = pd.DataFrame(index=pd.Index(network.nodes))
        # Add layout
        layout = normalise_layout(layout, vertex_ids=vertex_df.index)
        ndim = layout.shape[1]
        for i, layouti in enumerate(layout.T):
            vertex_df[f'_ipx_layout_{i}'] = layouti

        # Edges are a list of tuples, because of multiedges
        tmp = []
        for u, v, d in network.edges.data():
            row = {'_ipx_source': u, '_ipx_target': v}
            row.update(d)
            tmp.append(row)
        edge_df = pd.DataFrame(tmp)
        del tmp

    else:
        # Vertices are ordered integers, no gaps
        layout = normalise_layout(layout)
        ndim = layout.shape[1]

        # All that's left to do is to check their attributes
        vertex_df = pd.DataFrame(layout, columns=[f'_ipx_layout_{i}' for i in range(ndim)])

        # Edges are a list of tuples, because of multiedges
        tmp = []
        for edge in network.es:
            row = {'_ipx_source': edge.source, '_ipx_target': edge.target}
            row.update(edge.attributes())
            tmp.append(row)
        edge_df = pd.DataFrame(tmp)
        del tmp

    internal_data = {
        "vertex_df": vertex_df,
        "edge_df": edge_df,
        "directed": directed,
        "network_library": nl,
        "ndim": ndim,
    }
    return internal_data
