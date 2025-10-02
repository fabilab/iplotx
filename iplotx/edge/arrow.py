"""
Module for edge arrows in iplotx.
"""

from typing import Never, Optional

from math import atan2, cos, sin
import numpy as np
import matplotlib as mpl
from matplotlib.patches import PathPatch

from ..style import (
    get_style,
    rotate_style,
)


class EdgeArrowCollection(mpl.collections.PatchCollection):
    """Collection of arrow patches for plotting directed edgs."""

    _factor = 1.0

    def __init__(
        self,
        edge_collection,
        *args,
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        **kwargs,
    ) -> None:
        """Initialize the edge arrow collection.

        Parameters:
            edge_collection: The edge collection to which these arrows belong.
            transform: The transform to apply to the arrows. This related to the arrow size
                scaling, not the arrow tip position which is controlled by set_offset_transform.
        """

        self._edge_collection = edge_collection
        self._style = get_style(".edge.arrow")

        patches, sizes = self._create_artists()

        if "cmap" in self._edge_collection._style:
            kwargs["cmap"] = self._edge_collection._style["cmap"]
            kwargs["norm"] = self._edge_collection._style["norm"]

        super().__init__(
            patches,
            offsets=np.zeros((len(patches), 2)),
            offset_transform=self.get_offset_transform(),
            transform=transform,
            match_original=True,
            *args,
            **kwargs,
        )
        self._angles = np.zeros(len(self._paths))

        # Compute _transforms like in _CollectionWithScales for dpi issues
        self.set_sizes(sizes)

    def get_sizes(self):
        """Get arrow sizes (max of width and height), not scaled by dpi."""
        return self._sizes

    def get_sizes_dpi(self):
        """Get arrow sizes (max of width and height) scaled by dpi."""
        return self._transforms[:, 0, 0]

    def set_sizes(self, sizes, dpi=72.0):
        """Set vertex sizes.

        This rescales the current vertex symbol/path linearly, using this
        value as the largest of width and height.

        @param sizes: A sequence of vertex sizes or a single size.
        """
        if sizes is None:
            self._sizes = np.array([])
            self._transforms = np.empty((0, 3, 3))
        else:
            self._sizes = np.asarray(sizes)
            self._transforms = np.zeros((len(self._sizes), 3, 3))
            scale = self._sizes * dpi / 72.0 * self._factor
            self._transforms[:, 0, 0] = scale
            self._transforms[:, 1, 1] = scale
            self._transforms[:, 2, 2] = 1.0
        self.stale = True

    get_size = get_sizes
    set_size = set_sizes

    def set_figure(self, fig) -> None:
        """Set the figure for this artist and all children."""
        super().set_figure(fig)
        self._update_before_draw()

    def get_offset_transform(self):
        """Get offset transform for the edge arrows. This sets the tip of each arrow."""
        return self._edge_collection.get_transform()

    get_size = get_sizes
    set_size = set_sizes

    def _create_artists(self):
        style = self._style if self._style is not None else {}

        patches = []
        sizes = []
        for i in range(len(self._edge_collection._vertex_ids)):
            stylei = rotate_style(style, index=i)
            if ("facecolor" not in stylei) and ("color" not in stylei):
                stylei["facecolor"] = self._edge_collection.get_edgecolors()[i][:3]
            if ("edgecolor" not in stylei) and ("color" not in stylei):
                stylei["edgecolor"] = self._edge_collection.get_edgecolors()[i][:3]
            if "alpha" not in stylei:
                stylei["alpha"] = self._edge_collection.get_edgecolors()[i][3]
            if "linewidth" not in stylei:
                stylei["linewidth"] = self._edge_collection.get_linewidths()[i]

            patch, size = make_arrow_patch(
                **stylei,
            )
            patches.append(patch)
            sizes.append(size)

        return patches, sizes

    def _update_before_draw(self) -> None:
        """Update the arrow paths and directions before drawing, based on the edge collection."""
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)

        trans = self.get_offset_transform().transform

        for i, epath in enumerate(self._edge_collection.get_paths()):
            # Offset the arrow to point to the end of the edge
            self._offsets[i] = epath.vertices[-1]

            # Rotate the arrow to point in the direction of the edge
            apath = self._paths[i]
            # NOTE: because the tip of the arrow is at (0, 0) in patch space,
            # in theory it will rotate around that point already
            v2 = trans(epath.vertices[-1])
            v1 = trans(epath.vertices[-2])
            dv = v2 - v1
            theta = atan2(*(dv[::-1]))
            theta_old = self._angles[i]
            dtheta = theta - theta_old
            mrot = np.array([[cos(dtheta), sin(dtheta)], [-sin(dtheta), cos(dtheta)]])
            apath.vertices = apath.vertices @ mrot
            self._angles[i] = theta

    def set_array(self, A: np.ndarray) -> Never:
        """Set the array for cmap/norm coloring, but keep the facecolors as set (usually 'none')."""
        raise ValueError("Setting an array for arrows directly is not supported.")

    def set_colors(self, colors: np.ndarray) -> None:
        """Set arrow colors (edge and/or face) based on a colormap.

        Parameters:
            colors: Color array to apply. This must be an Nx3 or Nx4 vector of RGB or RGBA colors.
                This function will NOT attempt to convert other color descriptions to RGB/RGBA.
        """
        # NOTE: facecolors is always an array because we come from patches
        # It can have zero alpha (i.e. if we choose "none", or a hollow marker)
        self.set_edgecolor(colors)
        has_facecolor = self._facecolors[:, 3] > 0
        self._facecolors[has_facecolor] = colors[has_facecolor]

    @mpl.artist.allow_rasterization
    def draw(self, renderer):
        self._update_before_draw()
        super().draw(renderer)


def make_arrow_patch(
    marker: str = "|>",
    width: float = 8,
    height: Optional[float | str] = None,
    **kwargs,
):
    """Make a patch of the given marker shape and size.

    Parameters:
        marker: The marker shape to use. Supported markers are:
            "|>", "|/", "|\\", ">", "<", ">>", ")>", ")", "(", "]", "[", "|",
            "x", "s", "d", "p", "q". Dashes at the start of this string will
            be ignored, so "->" is equivalent to ">".
        width: The width of the marker in points. Height is by default 1.3 the
            width, unless specified separately.
        height: The height of the marker in points. If not specified, it is
            1.3 times the width. This can also be the string "width", in which
            case the height will be equal to the width.
        **kwargs: Additional keyword arguments passed to the PathPatch.

    Returns:
        A pair with the patch and the max size of the patch in points.
    """
    # Forget any leading dashes
    marker = marker.lstrip("-")

    if height is None:
        height = width * 1.3
    elif height == "width":
        height = width

    # Normalise by the max size, this is taken care of in _transforms
    # subsequently in a way that is nice to dpi scaling
    size_max = max(width, height)
    if size_max > 0:
        height /= size_max
        width /= size_max

    if marker == "|>":
        codes = ["MOVETO", "LINETO", "LINETO", "CLOSEPOLY"]
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        path = mpl.path.Path(
            np.array(
                [
                    [-height, width * 0.5],
                    [-height, -width * 0.5],
                    [0, 0],
                    [-height, width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == "|\\":
        codes = ["MOVETO", "LINETO", "LINETO", "CLOSEPOLY"]
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        path = mpl.path.Path(
            np.array(
                [
                    [-height, width * 0.5],
                    [-height, 0],
                    [0, 0],
                    [-height, width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == "|/":
        codes = ["MOVETO", "LINETO", "LINETO", "CLOSEPOLY"]
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        path = mpl.path.Path(
            np.array(
                [
                    [-height, 0],
                    [-height, -width * 0.5],
                    [0, 0],
                    [-height, 0],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == ">":
        kwargs["facecolor"] = "none"
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array([[-height, width * 0.5], [0, 0], [-height, -width * 0.5]]),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=False,
        )
    elif marker == "<":
        kwargs["facecolor"] = "none"
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array([[height, width * 0.5], [0, 0], [height, -width * 0.5]]),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=False,
        )
    elif marker == ">>":
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        overhang = kwargs.pop("overhang", 0.25)
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO", "CLOSEPOLY"]
        path = mpl.path.Path(
            np.array(
                [
                    [0, 0],
                    [-height, -width * 0.5],
                    [-height * (1.0 - overhang), 0],
                    [-height, width * 0.5],
                    [0, 0],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == ")>":
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        overhang = kwargs.pop("overhang", 0.25)
        codes = ["MOVETO", "LINETO", "CURVE3", "CURVE3", "CLOSEPOLY"]
        path = mpl.path.Path(
            np.array(
                [
                    [0, 0],
                    [-height, -width * 0.5],
                    [-height * (1.0 - overhang), 0],
                    [-height, width * 0.5],
                    [0, 0],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == ")":
        kwargs["facecolor"] = "none"
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "CURVE3", "CURVE3"]
        path = mpl.path.Path(
            np.array(
                [
                    [-height * 0.5, width * 0.5],
                    [height * 0.5, 0],
                    [-height * 0.5, -width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=False,
        )
    elif marker == "(":
        kwargs["facecolor"] = "none"
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "CURVE3", "CURVE3"]
        path = mpl.path.Path(
            np.array(
                [
                    [height * 0.5, width * 0.5],
                    [-height * 0.5, 0],
                    [height * 0.5, -width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=False,
        )
    elif marker == "]":
        kwargs["facecolor"] = "none"
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array(
                [
                    [-height, width * 0.5],
                    [0, width * 0.5],
                    [0, -width * 0.5],
                    [-height, -width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=False,
        )
    elif marker == "[":
        kwargs["facecolor"] = "none"
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array(
                [
                    [height, width * 0.5],
                    [0, width * 0.5],
                    [0, -width * 0.5],
                    [height, -width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=False,
        )
    elif marker == "|":
        kwargs["facecolor"] = "none"
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO"]
        path = mpl.path.Path(
            np.array([[0, width * 0.5], [0, -width * 0.5]]),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=False,
        )
    elif marker in ("x", "X"):
        kwargs["facecolor"] = "none"
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO", "MOVETO", "LINETO"]
        path = mpl.path.Path(
            np.array(
                [
                    [height * 0.5, width * 0.5],
                    [-height * 0.5, -width * 0.5],
                    [-height * 0.5, width * 0.5],
                    [height * 0.5, -width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=False,
        )
    elif marker == "s":
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO", "CLOSEPOLY"]
        path = mpl.path.Path(
            np.array(
                [
                    [-height, width * 0.5],
                    [-height, -width * 0.5],
                    [0, -width * 0.5],
                    [0, width * 0.5],
                    [-height, width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == "d":
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO", "CLOSEPOLY"]
        path = mpl.path.Path(
            np.array(
                [
                    [-height * 0.5, width * 0.5],
                    [-height, 0],
                    [-height * 0.5, -width * 0.5],
                    [0, 0],
                    [-height * 0.5, width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == "p":
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO", "CLOSEPOLY"]
        path = mpl.path.Path(
            np.array(
                [
                    [-height, 0],
                    [0, 0],
                    [0, -width],
                    [-height, -width],
                    [-height, 0],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == "q":
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO", "CLOSEPOLY"]
        path = mpl.path.Path(
            np.array(
                [
                    [-height, 0],
                    [0, 0],
                    [0, width],
                    [-height, width],
                    [-height, 0],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == "none":
        if "color" in kwargs:
            kwargs["facecolor"] = kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO"]
        path = mpl.path.Path(
            np.array(
                [
                    [0, 0],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    else:
        raise ValueError(f"Arrow marker not found: {marker}.")

    patch = PathPatch(
        path,
        **kwargs,
    )

    return patch, size_max
