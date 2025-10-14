"""
Module for vertex groupings code, especially the GroupingCollection class.
"""

from typing import Union
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.collections import PatchCollection


from ..typing import (
    GroupingType,
    LayoutType,
)
from ..ingest.heuristics import (
    normalise_layout,
    normalise_grouping,
)
from ..style import get_style, rotate_style
from ..utils.geometry import (
    convex_hull,
    _compute_group_path_with_vertex_padding,
)


class GroupingCollection(PatchCollection):
    """Matplotlib artist for a vertex grouping (clustering/cover).

    This class is used to plot patches surrounding groups of vertices in a network.
    """

    _factor = 1.0

    def __init__(
        self,
        grouping: GroupingType,
        layout: LayoutType,
        vertexpadding: Union[None, int] = None,
        points_per_curve: int = 30,
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        *args,
        **kwargs,
    ) -> None:
        """Container artist for vertex groupings, e.g. covers or clusterings.

        Parameters:
            grouping: This can be a sequence of sets (a la networkx), an igraph Clustering
                or Cover instance (including VertexClustering/VertexCover), or a sequence
                of integers/strings indicating memberships for each vertex.
            layout: The layout of the vertices. If this object has no keys/index, the
                vertices are assumed to have IDs corresponding to integers starting from
                zero.
            vertexpadding: How may points of padding to leave around each vertex centre.
            points_per_curve: How many points to use to approximate a round envelope around
                each convex hull vertex.
            transform: The matplotlib transform to use for the patches (typically transData).
        """
        if vertexpadding is not None:
            self._vertexpadding = vertexpadding
        else:
            style = get_style(".grouping")
            self._vertexpadding = style.get("vertexpadding", 10)

        self._points_per_curve = points_per_curve

        network = kwargs.pop("network", None)
        self.layout = layout = normalise_layout(layout, network=network)
        self.ndim = layout.shape[1]

        patches, grouping, coords_hulls = self._create_patches(
            grouping,
            layout,
            network,
            **kwargs,
        )
        if "network" in kwargs:
            del kwargs["network"]
        self._grouping = grouping
        self._coords_hulls = coords_hulls
        kwargs["match_original"] = True

        super().__init__(patches, *args, **kwargs)

        zorder = get_style(".grouping").get("zorder", 1)
        self.set_zorder(zorder)

        self.set_transform(transform)

    def set_figure(self, figure) -> None:
        """Set the figure for the grouping, recomputing the paths depending on the figure's dpi."""
        ret = super().set_figure(figure)
        self._compute_paths(self.get_figure(root=True).dpi)
        return ret

    @property
    def axes(self):
        return PatchCollection.axes.__get__(self)

    @axes.setter
    def axes(self, new_axes):
        PatchCollection.axes.__set__(self, new_axes)
        for child in self.get_children():
            child.axes = new_axes
        self.set_figure(new_axes.figure)

    def get_layout(self) -> pd.DataFrame:
        """Get the layout used for this grouping."""
        return self.layout

    def get_vertexpadding(self) -> float:
        """Get the vertex padding of each group."""
        return self._vertexpadding

    def get_vertexpadding_dpi(self, dpi: float = 72.0) -> float:
        """Get vertex padding of each group, scaled by dpi of the figure."""
        return self.get_vertexpadding() * dpi / 72.0 * self._factor

    def _create_patches(self, grouping, layout, network, **kwargs):
        grouping = normalise_grouping(grouping, layout)
        style = get_style(".grouping")
        style.pop("vertexpadding", None)

        style.update(kwargs)

        patches = []
        coords_hulls = []
        for i, (name, vids) in enumerate(grouping.items()):
            if len(vids) == 0:
                continue
            vids = np.array(list(vids))
            coords = layout.loc[vids].values
            idx_hull = convex_hull(coords)
            coords_hull = coords[idx_hull]
            coords_hulls.append(coords_hull)

            stylei = rotate_style(style, i)

            # NOTE: the transform is set later on
            patch = _compute_group_patch_stub(
                coords_hull,
                self._vertexpadding,
                label=name,
                **stylei,
            )

            patches.append(patch)
        return patches, grouping, coords_hulls

    def _compute_paths(self, dpi: float = 72.0) -> None:
        for i, hull in enumerate(self._coords_hulls):
            _compute_group_path_with_vertex_padding(
                hull,
                self._paths[i].vertices,
                self.get_transform(),
                vertexpadding=self.get_vertexpadding_dpi(dpi),
            )

    def _process(self) -> None:
        self._compute_paths()

    def draw(self, renderer) -> None:
        """Draw or re-draw the grouping patches.

        Parameters:
            renderer: The renderer to use for drawing the patches.
        """
        # FIXME: this kind of breaks everything since the vertices' magical "_transforms" does
        # not really scale from 72 pixels but rather from the screen's or something.
        # Conclusion: using this keeps consistency across dpis but breaks proportionality of
        # vertexpadding and vertex_size (for now).
        # NOTE: this might be less bad than initially thought in the sense that even perfect
        # scaling does not seem to align the center of the perimeter of the group with the
        # center of the perimeter of the vertex when of the same exact size. So we are
        # probably ok winging it as users will adapt.
        self._compute_paths(self.get_figure(root=True).dpi)
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
    vertices = np.zeros(
        (1 + 30 * len(points), 2),
    )
    codes = ["MOVETO"] + ["LINETO"] * (len(vertices) - 2) + ["CLOSEPOLY"]
    codes = [getattr(mpl.path.Path, x) for x in codes]
    patch = mpl.patches.PathPatch(
        mpl.path.Path(
            vertices,
            codes=codes,
        ),
        **kwargs,
    )

    return patch
