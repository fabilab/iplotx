from copy import deepcopy
import numpy as np
from matplotlib import artist
from matplotlib.transforms import IdentityTransform
from matplotlib.collections import PatchCollection
from matplotlib.patches import (
    Ellipse,
    Circle,
    RegularPolygon,
    Rectangle,
)

from .style import (
    get_style,
    rotate_style,
)
from .utils.matplotlib import (
    _get_label_width_height,
    _build_cmap_fun,
)
from .label import LabelCollection


class VertexCollection(PatchCollection):
    """Collection of vertex patches for plotting.

    This class takes additional keyword arguments compared to PatchCollection:

    @param vertex_builder: A list of vertex builders to construct the visual
        vertices. This is updated if the size of the vertices is changed.
    @param size_callback: A function to be triggered after vertex sizes are
        changed. Typically this redraws the edges.
    """

    _factor = 1.0

    def __init__(self, *args, **kwargs):

        layout = kwargs.pop("layout")
        self._style = kwargs.pop("style", None)
        self._labels = kwargs.pop("labels", None)

        # Create patches from structured data
        patches, offsets, sizes, kwargs2 = self._create_artists(
            layout,
        )
        kwargs.update(kwargs2)
        kwargs["offsets"] = offsets
        kwargs["match_original"] = True

        # Pass to PatchCollection constructor
        super().__init__(patches, *args, **kwargs)

        # Compute _transforms like in _CollectionWithScales for dpi issues
        self.set_sizes(sizes)

    def get_sizes(self):
        return self._sizes

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
            # This is kinda funny: even though the scale is apparently
            # applied linearly to each axis (dpi means "dots per inch",
            # which is also a linear measurement), we still need to take
            # the square root of the sizes for the **visual** size on screen
            # (or on PNG) to scale **linearly**. It is more funny still
            # because the scaling is not exact, there is about a factor
            # ~1.2-1.5 difference between the scale one sets and the pixels on
            # screen...
            # But this does fix #5 in terms of storing the PNG with different
            # dpi resolutions: they look the same. But relative scaling is
            # all off LOL
            scale = self._sizes * dpi / 72.0 * self._factor
            self._transforms[:, 0, 0] = scale
            self._transforms[:, 1, 1] = scale
            self._transforms[:, 2, 2] = 1.0
        self.stale = True

    get_size = get_sizes
    set_size = set_sizes

    def _create_artists(self, vertex_layout_df):
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
            offsets.append(list(row.values))

            if style.get("size") == "label":
                # NOTE: it's ok to overwrite the dict here
                style["size"] = _get_label_width_height(
                    str(vertex_labels[vid]), **style.get("label", {})
                )

            stylei = rotate_style(style, index=i, id=vid)
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

    def _compute_labels(self):
        transform = self.get_offset_transform()
        trans = transform.transform

        style = (
            deepcopy(self._style.get("label", None)) if self._style is not None else {}
        )
        forbidden_props = ["hpadding", "vpadding"]
        for prop in forbidden_props:
            if prop in style:
                del style[prop]

        if not hasattr(self, "_label_collection"):
            self._label_collection = LabelCollection(
                self._labels,
                style=style,
                offsets=self._offsets,
                transform=transform,
            )

            # Forward a bunch of mpl settings that are needed
            self._label_collection.set_figure(self.figure)
            self._label_collection.axes = self.axes
            # forward the clippath/box to the children need this logic
            # because mpl exposes some fast-path logic
            clip_path = self.get_clip_path()
            if clip_path is None:
                clip_box = self.get_clip_box()
                self._label_collection.set_clip_box(clip_box)
            else:
                self._label_collection.set_clip_path(clip_path)

            # Finally make the patches
            self._label_collection._create_artists()

    def get_labels(self):
        if hasattr(self, "_label_collection"):
            return self._label_collection
        else:
            return None

    def get_children(self):
        children = []
        if hasattr(self, "_label_collection"):
            children.append(self._label_collection)
        return children

    @property
    def stale(self):
        return super().stale

    @stale.setter
    def stale(self, val):
        PatchCollection.stale.fset(self, val)
        if val and hasattr(self, "stale_callback_post"):
            self.stale_callback_post(self)

    @artist.allow_rasterization
    def draw(self, renderer):
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)
        if self._labels is not None:
            self._compute_labels()
        super().draw(renderer)

        for child in self.get_children():
            child.draw(renderer)


def make_patch(marker: str, size, **kwargs):
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
    size /= size_max

    if marker in ("o", "circle"):
        art = Circle((0, 0), size[0] / 2, **kwargs)
    elif marker in ("s", "square", "r", "rectangle"):
        art = Rectangle((-size[0] / 2, -size[1] / 2), size[0], size[1], **kwargs)
    elif marker in ("^", "triangle"):
        art = RegularPolygon((0, 0), numVertices=3, radius=size[0] / 2, **kwargs)
    elif marker in ("d", "diamond"):
        art = make_patch("s", size[0], angle=45, **kwargs)
    elif marker in ("v", "triangle_down"):
        art = RegularPolygon(
            (0, 0), numVertices=3, radius=size[0] / 2, orientation=np.pi, **kwargs
        )
    elif marker in ("e", "ellipse"):
        art = Ellipse((0, 0), size[0] / 2, size[1] / 2, **kwargs)
    else:
        raise KeyError(f"Unknown marker: {marker}")

    return art, size_max
