"""
Module containing leaf edges, i.e. special edges of tree visualisations
that connect leaf vertices to the deepest leaf (typically for labeling).
"""

from typing import (
    Sequence,
    Optional,
    Any,
)
import numpy as np
import pandas as pd
import matplotlib as mpl

from ..utils.matplotlib import (
    _forwarder,
)
from ..vertex import VertexCollection
from iplotx.edge import EdgeCollection


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
class LeafEdgeCollection(EdgeCollection):
    """Artist for leaf edges in tree visualisations."""

    def __init__(
        self,
        patches: Sequence[mpl.patches.Patch],
        vertex_leaf_ids: Sequence[tuple],
        vertex_collection: VertexCollection,
        leaf_collection: VertexCollection,
        *args,
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        arrow_transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        directed: bool = False,
        style: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialise a LeafEdgeCollection.

        Parameters:
            patches: A sequence (usually, list) of matplotlib `Patch`es describing the edges.
            vertex_ids: A sequence of pairs `(v1, v2)`, each defining the ids of vertices at the
                end of an edge.
            vertex_collection: The VertexCollection instance containing the Artist for the
                vertices. This is needed to compute vertex borders and adjust edges accordingly.
            transform: The matplotlib transform for the edges, usually transData.
            arrow_transform: The matplotlib transform for the arrow patches. This is not the
                *offset_transform* of arrows, which is set equal to the edge transform (previous
                parameter). Instead, it specifies how arrow size scales, similar to vertex size.
                This is usually the identity transform.
            directed: Whether the graph is directed (in which case arrows are drawn, possibly
                with zero size or opacity to obtain an "arrowless" effect).
            style: The edge style (subdictionary: "edge") to use at creation.
        """
        self._leaf_collection = leaf_collection
        super().__init__(
            patches=patches,
            vertex_ids=vertex_leaf_ids,
            vertex_collection=vertex_collection,
            *args,
            transform=transform,
            arrow_transform=arrow_transform,
            directed=directed,
            style=style,
            **kwargs,
        )

    def _get_adjacent_vertices_info(self):
        lindex = self._leaf_collection.get_index()
        lindex = pd.Series(
            np.arange(len(lindex)),
            index=lindex,
        )
        vindex = self._vertex_collection.get_index()
        vindex = pd.Series(
            np.arange(len(vindex)),
            index=vindex,
        ).loc[lindex.index]

        voffsets = []
        vpaths = []
        vsizes = []
        for vid in self._vertex_ids:
            # NOTE: these are in the original layout coordinate system
            # not cartesianised yet.
            offset1 = self._vertex_collection.get_layout().values[vindex[vid]]
            offset2 = self._leaf_collection.get_layout().values[lindex[vid]]
            voffsets.append((offset1, offset2))

            path1 = self._vertex_collection.get_paths()[vindex[vid]]
            path2 = self._leaf_collection.get_paths()[lindex[vid]]
            vpaths.append((path1, path2))

            # NOTE: This needs to be computed here because the
            # VertexCollection._transforms are reset each draw in order to
            # accomodate for DPI changes on the canvas
            size1 = self._vertex_collection.get_sizes_dpi()[vindex[vid]]
            size2 = self._leaf_collection.get_sizes_dpi()[lindex[vid]]
            vsizes.append((size1, size2))

        return {
            "ids": [(vid, vid) for vid in self._vertex_ids],
            "offsets": voffsets,
            "paths": vpaths,
            "sizes": vsizes,
            "loops": False,
        }
