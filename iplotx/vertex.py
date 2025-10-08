"""
Module containing code to manipulate vertex visualisations, especially the VertexCollection class.
"""

from typing import (
    Optional,
    Sequence,
    Any,
)
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.patches import (
    Patch,
    Polygon,
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
        labels: Optional[Sequence[str] | pd.Series] = None,
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
        self._style = style if style is not None else {}
        self._layout = layout
        self._layout_coordinate_system = layout_coordinate_system

        if (labels is not None) and (not isinstance(labels, pd.Series)):
            labels = pd.Series(labels, index=self._layout.index)
        self._labels = labels

        # Create patches from structured data
        patches, sizes, kwargs2 = self._init_vertex_patches()

        kwargs.update(kwargs2)
        kwargs["match_original"] = True

        # Pass to PatchCollection constructor
        super().__init__(patches, *args, **kwargs)

        # Set offsets in coordinate system
        self._update_offsets_from_layout()

        # Compute _transforms like in _CollectionWithScales for dpi issues
        self.set_sizes(sizes)

        if self._labels is not None:
            self._compute_label_collection()

        zorder = self._style.get("zorder", 1)
        self.set_zorder(zorder)

    def __len__(self):
        """Return the number of vertices in the collection."""
        return len(self.get_paths())

    def get_children(self) -> tuple[mpl.artist.Artist]:
        """Get the children artists.

        This can include the labels as a LabelCollection.
        """
        children = []
        if hasattr(self, "_label_collection"):
            children.append(self._label_collection)
        return tuple(children)

    def set_figure(self, fig) -> None:
        """Set the figure for this artist and all children."""
        super().set_figure(fig)
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)
        self._update_children()
        for child in self.get_children():
            child.set_figure(fig)

    @property
    def axes(self):
        return PatchCollection.axes.__get__(self)

    @axes.setter
    def axes(self, new_axes):
        PatchCollection.axes.__set__(self, new_axes)
        for child in self.get_children():
            child.axes = new_axes

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

    def set_sizes(self, sizes, dpi: float = 72.0) -> None:
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

    def get_layout(self) -> pd.DataFrame:
        """Get the vertex layout.

        Returns:
            The vertex layout as a DataFrame.
        """
        return self._layout

    def get_layout_coordinate_system(self) -> str:
        """Get the layout coordinate system.

        Returns:
            Name of the layout coordinate system, e.g. "cartesian" or "polar".
        """
        return self._layout_coordinate_system

    def get_offsets(self, ignore_layout: bool = True) -> np.ndarray:
        """Get the vertex offsets.

        Parameters:
            ignore_layout: If True, return the matplotlib Artist._offsets directly, ignoring the
                layout coordinate system. If False, it's equivalent to get_layout().values.

        Returns:
            The vertex offsets as a 2D numpy array.

        Note: It is best for users to *not* ignore the layout coordinate system, as it may lead
        to inconsistencies. However, some internal matplotlib functions require the default
        signature of this function to look at the vanilla offsets, hence the default parameters.
        """
        if not ignore_layout:
            return self.get_layout().values
        else:
            return self._offsets

    def _update_offsets_from_layout(self) -> None:
        """Update offsets in matplotlib coordinates from the layout DataFrame."""
        if self._layout_coordinate_system == "cartesian":
            # Make sure we accept 3D values and ignore the z component if present
            # This makes life upstream a little more readable
            self._offsets = self._layout.values[:, :2]
        elif self._layout_coordinate_system == "polar":
            # Convert polar coordinates (r, theta) to cartesian (x, y)
            r = self._layout.iloc[:, 0].values
            theta = self._layout.iloc[:, 1].values
            if self._offsets is None:
                self._offsets = np.zeros((len(r), 2))
            self._offsets[:, 0] = r * np.cos(theta)
            self._offsets[:, 1] = r * np.sin(theta)
        else:
            raise ValueError(
                f"Layout coordinate system not supported: {self._layout_coordinate_system}."
            )

    def set_offsets(self, offsets: np.ndarray) -> None:
        """Set the vertex positions/offsets in layout coordinates.

        Parameters:
            offsets: Array of coordinates in the layout coordinate system. For polar layouts,
                these should be in the form of (r, theta) pairs.
        """
        self._layout.values[:] = offsets
        self._update_offsets_from_layout()
        self.stale = True

    def set_offset_transform(self, transform: mpl.transforms.Transform) -> None:
        """Set the offset transform for the vertices.

        Parameters:
            transform: The matplotlib transform to use for the offsets.
        """
        super().set_offset_transform(transform)
        if hasattr(self, "_label_collection"):
            self._label_collection.set_transform(transform)

    def get_style(self) -> Optional[dict[str, Any]]:
        """Get the style dictionary for the vertices."""
        return self._style

    def _init_vertex_patches(self):
        style = self._style or {}
        if "cmap" in style:
            cmap_fun = _build_cmap_fun(
                style,
                "facecolor",
            )
        else:
            cmap_fun = None

        size_tmp = style.get("size", 20)
        if isinstance(size_tmp, str) and (size_tmp == "label"):
            if self._labels is None:
                warnings.warn("No labels found, cannot resize vertices based on labels.")
                style["size"] = get_style("default.vertex")["size"]
        del size_tmp

        if "cmap" in style:
            colorarray = []
        patches = []
        sizes = []
        for i, (vid, row) in enumerate(self._layout.iterrows()):
            # Use vertex labels if present as a fallback key for style rotation
            # This way one can be very specific via the exact object or cast
            # a looser net with the label string
            key2 = self._labels.iloc[i] if self._labels is not None else None
            stylei = rotate_style(style, index=i, key=vid, key2=key2)
            if stylei.get("size", 20) == "label":
                stylei["size"] = _get_label_width_height(
                    str(self._labels[vid]), **style.get("label", {})
                )
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

        return patches, sizes, kwargs

    def _compute_label_collection(self):
        transform = self.get_offset_transform()

        style = (
            copy_with_deep_values(self._style.get("label", None)) if self._style is not None else {}
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

    def get_ndim(self):
        """Get the number of dimensions of the layout."""
        return self._layout.shape[1]

    def get_labels(self):
        """Get the vertex labels.

        Returns:
            The artist with the LabelCollection.
        """

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

    def _update_children(self) -> None:
        """Update children before drawing and before first render."""
        self._update_labels()

    def _update_labels(self) -> None:
        """Update labels before drawing.

        NOTE: This needs to work in figure coordinates.
        """
        if not hasattr(self, "_label_collection"):
            return

        if self.get_layout_coordinate_system() != "polar":
            return

        transform = self.get_offset_transform()
        trans = transform.transform

        zero_fig = trans(np.array([0, 0]))
        offsets_fig = trans(self.get_labels().get_offsets())
        doffsets_fig = offsets_fig - zero_fig
        rotations = np.arctan2(doffsets_fig[:, 1], doffsets_fig[:, 0])
        self.get_labels().set_rotations(rotations)

    def _update_before_draw(self) -> None:
        """Update the collection before drawing."""
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)

    @mpl.artist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return

        # null graph, no need to draw anything
        # NOTE: I would expect this to be already a clause in the superclass by oh well
        if len(self.get_paths()) == 0:
            return

        self._update_before_draw()
        super().draw(renderer)

        # Set the label rotations already, hopefully this is not too early
        self._update_children()

        # NOTE: This draws the vertices first, then the labels.
        # The correct order would be vertex1->label1->vertex2->label2, etc.
        # We might fix if we manage to find a way to do it.
        for child in self.get_children():
            child.draw(renderer)


def make_patch(
    marker: str = "o",
    size: float | Sequence[float] = 20,
    **kwargs,
) -> tuple[Patch, float]:
    """Make a patch of the given marker shape and size."""
    forbidden_props = ["label", "cmap", "norm", "cascade", "deep", "depthshade"]
    for prop in forbidden_props:
        if prop in kwargs:
            kwargs.pop(prop)

    if np.isscalar(size):
        size = float(size)
        size = (size, size)
    size = np.asarray(size, dtype=float)

    # Size of vertices is determined in self._transforms, which scales with dpi, rather than here,
    # so normalise by the average dimension (btw x and y) to keep the ratio of the marker.
    # If you check in get_sizes, you will see that rescaling also happens with the max of width
    # and height.
    size_max = size.max()
    if size_max > 0:
        size /= size_max

    art: Patch
    if marker in ("o", "c", "circle"):
        art = Circle((0, 0), size[0] / 2, **kwargs)
    elif marker in ("s", "square", "r", "rectangle"):
        art = Rectangle((-size[0] / 2, -size[1] / 2), size[0], size[1], **kwargs)
    elif marker in ("^", "triangle"):
        art = RegularPolygon((0, 0), numVertices=3, radius=size[0] / np.sqrt(2), **kwargs)
    elif marker in ("v", "triangle_down"):
        art = RegularPolygon(
            (0, 0),
            numVertices=3,
            radius=size[0] / np.sqrt(2),
            orientation=np.pi,
            **kwargs,
        )
    elif marker in ("<", "triangle_left"):
        art = RegularPolygon(
            (0, 0),
            numVertices=3,
            radius=size[0] / np.sqrt(2),
            orientation=np.pi / 2,
            **kwargs,
        )
    elif marker in (">", "triangle_right"):
        art = RegularPolygon(
            (0, 0),
            numVertices=3,
            radius=size[0] / np.sqrt(2),
            orientation=-np.pi / 2,
            **kwargs,
        )
    elif marker in ("d", "diamond"):
        art = RegularPolygon((0, 0), numVertices=4, radius=size[0] / np.sqrt(2), **kwargs)
    elif marker in ("p", "pentagon"):
        art = RegularPolygon((0, 0), numVertices=5, radius=size[0] / np.sqrt(2), **kwargs)
    elif marker in ("h", "hexagon"):
        art = RegularPolygon((0, 0), numVertices=6, radius=size[0] / np.sqrt(2), **kwargs)
    elif marker in ("8", "octagon"):
        art = RegularPolygon((0, 0), numVertices=8, radius=size[0] / np.sqrt(2), **kwargs)
    elif marker in ("e", "ellipse"):
        art = Ellipse((0, 0), size[0], size[1], **kwargs)
    elif marker in ("*", "star"):
        size *= np.sqrt(2)
        art = Polygon(
            [
                (0, size[1] / 2),
                (size[0] / 7, size[1] / 7),
                (size[0] / 2, size[1] / 7),
                (size[0] / 4, -size[1] / 8),
                (size[0] / 3, -size[1] / 2),
                (0, -0.27 * size[1]),
                (-size[0] / 3, -size[1] / 2),
                (-size[0] / 4, -size[1] / 8),
                (-size[0] / 2, size[1] / 7),
                (-size[0] / 7, size[1] / 7),
                (0, size[1] / 2),
            ][::-1],
            **kwargs,
        )
    else:
        raise KeyError(f"Unknown marker: {marker}")

    return (art, size_max)
