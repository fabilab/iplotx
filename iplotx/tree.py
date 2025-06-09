import numpy as np
import pandas as pd
import matplotlib as mpl


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

    def _add_vertices(self):
        # TODO:
        self._vertices = VertexCollection

    def _add_edges(self):
        # TODO:
        self._edges = EdgeCollection

    def draw(self, renderer):
        super().draw(renderer)
        for child in self.get_children():
            child.draw(renderer)
