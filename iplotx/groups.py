from typing import Union, Sequence
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.collections import PatchCollection


from .importing import igraph
from .typing import (
    GroupingType,
    LayoutType,
)
from .heuristics import normalise_layout, normalise_grouping
from .styles import get_style
from .tools.geometry import convex_hull


class GroupingArtist(PatchCollection):
    def __init__(
        self,
        grouping: GroupingType,
        layout: LayoutType,
        *args,
        **kwargs,
    ):
        """Container artist for vertex groupings, e.g. covers or clusterings.

        Parameters:
            grouping: This can be a sequence of sets (a la networkx), an igraph Clustering
                or Cover instance (including VertexClustering/VertexCover), or a sequence
                of integers/strings indicating memberships for each vertex.
            layout: The layout of the vertices. If this object has no keys/index, the
                vertices are assumed to have IDs corresponding to integers starting from
                zero.
        """
        patches, grouping, layout = self._create_patches(grouping, layout, **kwargs)
        self._grouping = grouping
        self._layout = layout
        kwargs["match_original"] = True

        super().__init__(patches, *args, **kwargs)

    def _create_patches(self, grouping, layout, **kwargs):
        layout = normalise_layout(layout)
        grouping = normalise_grouping(grouping, layout)
        style = get_style(".grouping")

        style.update(kwargs)

        patches = []
        for name, vids in grouping.items():
            if len(vids) == 0:
                continue
            vids = np.array(list(vids))
            coords = layout.loc[vids].values
            idx_hull = convex_hull(coords)
            coords_hull = coords[idx_hull]

            patches.append(
                mpl.patches.Polygon(
                    coords_hull,
                    label=name,
                    # NOTE: the transform is set later on
                    **style,
                )
            )
        return patches, grouping, layout

    def _process(self):
        self.set_transform(self.axes.transData)
