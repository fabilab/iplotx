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
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
        *args,
        **kwargs,
    ):

        self._edge_collection = edge_collection
        self._style = get_style(".arrow")

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
        """Get vertex sizes (max of width and height), not scaled by dpi."""
        return self._sizes

    def get_sizes_dpi(self):
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

    def get_offset_transform(self):
        return self._edge_collection.get_transform()

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

    def _create_artists(self):
        style = self._style if self._style is not None else {}

        patches = []
        sizes = []
        for i, (vid1, vid2) in enumerate(self._edge_collection._vertex_ids):
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

    def set_array(self, array):
        """Set the array for cmap/norm coloring, but keep the facecolors as set (usually 'none')."""
        fcs = self.get_facecolors()
        super().set_array(array)
        self.set_facecolors(fcs)

    @property
    def stale(self):
        return super().stale

    @stale.setter
    def stale(self, val):
        mpl.collections.PatchCollection.stale.fset(self, val)
        if val and hasattr(self, "stale_callback_post"):
            self.stale_callback_post(self)

    @mpl.artist.allow_rasterization
    def draw(self, renderer):
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)
        super().draw(renderer)


def make_arrow_patch(marker: str = "|>", width: float = 8, **kwargs):
    """Make a patch of the given marker shape and size."""
    height = kwargs.pop("height", width * 1.3)

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
    elif marker == "|":
        kwargs["facecolor"] = "none"
        if "color" in kwargs:
            kwargs["edgecolor"] = kwargs.pop("color")
        codes = ["MOVETO", "LINETO"]
        path = mpl.path.Path(
            np.array([[-height, width * 0.5], [-height, -width * 0.5]]),
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
