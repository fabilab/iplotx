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
from .tools.geometry import _compute_group_path_with_vertex_padding


class GroupingArtist(PatchCollection):
    def __init__(
        self,
        grouping: GroupingType,
        layout: LayoutType,
        vertexpadding: Union[None, int] = None,
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
        if vertexpadding is not None:
            self._vertexpadding = vertexpadding
        else:
            style = get_style(".grouping")
            self._vertexpadding = style.get("vertexpadding", 10)
        patches, grouping, layout = self._create_patches(grouping, layout, **kwargs)
        self._grouping = grouping
        self._layout = layout
        kwargs["match_original"] = True

        super().__init__(patches, *args, **kwargs)

    def _create_patches(self, grouping, layout, **kwargs):
        layout = normalise_layout(layout)
        grouping = normalise_grouping(grouping, layout)
        style = get_style(".grouping")
        style.pop("vertexpadding", None)

        style.update(kwargs)

        patches = []
        for name, vids in grouping.items():
            if len(vids) == 0:
                continue
            vids = np.array(list(vids))
            coords = layout.loc[vids].values
            idx_hull = convex_hull(coords)
            coords_hull = coords[idx_hull]

            # NOTE: the transform is set later on
            patch = _compute_group_patch_stub(
                coords_hull,
                self._vertexpadding,
                label=name,
                **style,
            )

            patches.append(patch)
        return patches, grouping, layout

    def _process(self):
        self.set_transform(self.axes.transData)
        if self._vertexpadding > 0:
            for i, path in enumerate(self._paths):
                self._paths[i].vertices = _compute_group_path_with_vertex_padding(
                    path.vertices,
                    self.get_transform(),
                    vertexpadding=self._vertexpadding,
                )


def _compute_group_patch_stub(
    points,
    vertexpadding,
    **kwargs,
):
    if vertexpadding == 0:
        return mpl.patches.Polygon(
            points,
            **kwargs,
        )

    vertices = []
    codes = []
    for point in points:
        vertices.extend([point] * 3)
        codes.extend(["LINETO", "CURVE3", "CURVE3"])
    codes[0] = "MOVETO"
    vertices.append(vertices[0])
    codes.append("CLOSEPOLY")

    codes = [getattr(mpl.path.Path, x) for x in codes]
    patch = mpl.patches.PathPatch(
        mpl.path.Path(
            vertices,
            codes=codes,
        ),
        **kwargs,
    )

    return patch
