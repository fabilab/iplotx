"""
Module containing code to manipulate vertex visualisations, especially the VertexCollection class.
"""

from typing import (
    Optional,
    Sequence,
    Any,
    Never,
)
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import (
    Patch,
    Ellipse,
    Circle,
    RegularPolygon,
    Rectangle,
)

from .style import (
    get_style,
    rotate_style,
    copy_with_deep_values,
)
from .utils.matplotlib import (
    _get_label_width_height,
    _build_cmap_fun,
    _forwarder,
)
from .label import LabelCollection


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
class VertexCollection(PatchCollection):
    """Collection of vertex patches for plotting."""

    _factor = 1.0

    def __init__(
        self,
        layout: pd.DataFrame,
        *args,
        layout_coordinate_system: str = "cartesian",
        style: Optional[dict[str, Any]] = None,
        labels: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        """Initialise the VertexCollection.

        Parameters:
            layout: The vertex layout.
            layout_coordinate_system: The coordinate system for the layout, usually "cartesian").
            style: The vertex style (subdictionary "vertex") to apply.
            labels: The vertex labels, if present.
        """

        self._index = layout.index
        self._style = style
        self._labels = labels

        # Create patches from structured data
        patches, offsets, sizes, kwargs2 = self._init_vertex_patches(
            layout,
            layout_coordinate_system=layout_coordinate_system,
        )

        kwargs.update(kwargs2)
        kwargs["offsets"] = offsets
        kwargs["match_original"] = True

        # Pass to PatchCollection constructor
        super().__init__(patches, *args, **kwargs)

        # Compute _transforms like in _CollectionWithScales for dpi issues
        self.set_sizes(sizes)

        if self._labels is not None:
            self._compute_label_collection()

    def get_children(self) -> tuple[mpl.artist.Artist]:
        """Get the children artists.

        This can include the labels as a LabelCollection.
        """
        children = []
        if hasattr(self, "_label_collection"):
            children.append(self._label_collection)
        return tuple(children)

    def set_figure(self, fig) -> Never:
        """Set the figure for this artist and all children."""
        super().set_figure(fig)
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)
        for child in self.get_children():
            child.set_figure(fig)

    def get_index(self):
        """Get the VertexCollection index."""
        return self._index

    def get_vertex_id(self, index):
        """Get the id of a single vertex at a positional index."""
        return self._index[index]

    def get_sizes(self):
        """Get vertex sizes (max of width and height), not scaled by dpi."""
        return self._sizes

    def get_sizes_dpi(self):
        """Get vertex sizes (max of width and height), scaled by dpi."""
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

    def _init_vertex_patches(
        self, vertex_layout_df, layout_coordinate_system="cartesian"
    ):
        style = self._style or {}
        if "cmap" in style:
            cmap_fun = _build_cmap_fun(
                style["facecolor"],
                style["cmap"],
            )
        else:
            cmap_fun = None

        if style.get("size", 20) == "label":
            if self._labels is None:
                warnings.warn(
                    "No labels found, cannot resize vertices based on labels."
                )
                style["size"] = get_style("default.vertex")["size"]
            else:
                vertex_labels = self._labels

        if "cmap" in style:
            colorarray = []
        patches = []
        offsets = []
        sizes = []
        for i, (vid, row) in enumerate(vertex_layout_df.iterrows()):
            # Centre of the vertex
            offset = list(row.values)

            # Transform to cartesian coordinates if needed
            if layout_coordinate_system == "polar":
                r, theta = offset
                offset = [r * np.cos(theta), r * np.sin(theta)]

            offsets.append(offset)

            if style.get("size") == "label":
                # NOTE: it's ok to overwrite the dict here
                style["size"] = _get_label_width_height(
                    str(vertex_labels[vid]), **style.get("label", {})
                )

            stylei = rotate_style(style, index=i, key=vid)
            if cmap_fun is not None:
                colorarray.append(style["facecolor"])
                stylei["facecolor"] = cmap_fun(stylei["facecolor"])

            # Shape of the vertex (Patch)
            art, size = make_patch(**stylei)
            patches.append(art)
            sizes.append(size)

        kwargs = {}
        if "cmap" in style:
            vmin = np.min(colorarray)
            vmax = np.max(colorarray)
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            kwargs["cmap"] = style["cmap"]
            kwargs["norm"] = norm

        return patches, offsets, sizes, kwargs

    def _compute_label_collection(self):
        transform = self.get_offset_transform()

        style = (
            copy_with_deep_values(self._style.get("label", None))
            if self._style is not None
            else {}
        )
        forbidden_props = ["hpadding", "vpadding"]
        for prop in forbidden_props:
            if prop in style:
                del style[prop]

        self._label_collection = LabelCollection(
            self._labels,
            style=style,
            offsets=self._offsets,
            transform=transform,
        )

    def get_labels(self):
        if hasattr(self, "_label_collection"):
            return self._label_collection
        else:
            return None

    @property
    def stale(self):
        return super().stale

    @stale.setter
    def stale(self, val):
        PatchCollection.stale.fset(self, val)
        if val and hasattr(self, "stale_callback_post"):
            self.stale_callback_post(self)

    @mpl.artist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return

        # null graph, no need to draw anything
        # NOTE: I would expect this to be already a clause in the superclass by oh well
        if len(self.get_paths()) == 0:
            return

        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)

        # NOTE: This draws the vertices first, then the labels.
        # The correct order would be vertex1->label1->vertex2->label2, etc.
        # We might fix if we manage to find a way to do it.
        super().draw(renderer)
        for child in self.get_children():
            child.draw(renderer)


def make_patch(
    marker: str, size: float | Sequence[float], **kwargs
) -> tuple[Patch, float]:
    """Make a patch of the given marker shape and size."""
    forbidden_props = ["label", "cmap", "norm"]
    for prop in forbidden_props:
        if prop in kwargs:
            kwargs.pop(prop)

    if np.isscalar(size):
        size = float(size)
        size = (size, size)

    # Size of vertices is determined in self._transforms, which scales with dpi, rather than here,
    # so normalise by the average dimension (btw x and y) to keep the ratio of the marker.
    # If you check in get_sizes, you will see that rescaling also happens with the max of width and height.
    size = np.asarray(size, dtype=float)
    size_max = size.max()
    if size_max > 0:
        size /= size_max

    art: Patch
    if marker in ("o", "c", "circle"):
        art = Circle((0, 0), size[0] / 2, **kwargs)
    elif marker in ("s", "square", "r", "rectangle"):
        art = Rectangle((-size[0] / 2, -size[1] / 2), size[0], size[1], **kwargs)
    elif marker in ("^", "triangle"):
        art = RegularPolygon((0, 0), numVertices=3, radius=size[0] / 2, **kwargs)
    elif marker in ("d", "diamond"):
        art, _ = make_patch("s", size[0], angle=45, **kwargs)
    elif marker in ("v", "triangle_down"):
        art = RegularPolygon(
            (0, 0), numVertices=3, radius=size[0] / 2, orientation=np.pi, **kwargs
        )
    elif marker in ("e", "ellipse"):
        art = Ellipse((0, 0), size[0] / 2, size[1] / 2, **kwargs)
    else:
        raise KeyError(f"Unknown marker: {marker}")

    return (art, size_max)
