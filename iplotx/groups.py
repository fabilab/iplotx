from typing import Union, Sequence
from copy import deepcopy
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
from .ingest.heuristics import (
    normalise_layout,
    normalise_grouping,
)
from .styles import get_style, rotate_style
from .utils.geometry import (
    convex_hull,
    _compute_group_path_with_vertex_padding,
)


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
        for i, (name, vids) in enumerate(grouping.items()):
            if len(vids) == 0:
                continue
            vids = np.array(list(vids))
            coords = layout.loc[vids].values
            idx_hull = convex_hull(coords)
            coords_hull = coords[idx_hull]

            stylei = rotate_style(style, i)

            # NOTE: the transform is set later on
            patch = _compute_group_patch_stub(
                coords_hull,
                self._vertexpadding,
                label=name,
                **stylei,
            )

            patches.append(patch)
        return patches, grouping, layout

    def _compute_paths(self):
        if self._vertexpadding > 0:
            for i, path in enumerate(self._paths):
                self._paths[i].vertices = _compute_group_path_with_vertex_padding(
                    path.vertices,
                    self.get_transform(),
                    vertexpadding=self._vertexpadding,
                )

    def _process(self):
        self.set_transform(self.axes.transData)
        self._compute_paths()

    def draw(self, renderer):
        self._compute_paths()
        super().draw(renderer)


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

    # NOTE: Closing point: mpl is a bit quirky here
    vertices = []
    codes = []
    if len(points) == 0:
        vertices = np.zeros((0, 2))
    elif len(points) == 1:
        vertices = [points[0]] * 9
        codes = ["MOVETO"] + ["CURVE3"] * 8
    elif len(points) == 2:
        vertices = [points[0]] * 5 + [points[1]] * 5 + [points[0]]
        codes = ["MOVETO"] + ["CURVE3"] * 4 + ["LINETO"] + ["CURVE3"] * 4 + ["LINETO"]
    else:
        for point in points:
            vertices.extend([point] * 3)
            codes.extend(["LINETO", "CURVE3", "CURVE3"])
        vertices.append(vertices[0])
        codes.append("LINETO")
        codes[0] = "MOVETO"

    codes = [getattr(mpl.path.Path, x) for x in codes]
    patch = mpl.patches.PathPatch(
        mpl.path.Path(
            vertices,
            codes=codes,
        ),
        **kwargs,
    )

    return patch
