from copy import deepcopy
import numpy as np
import matplotlib as mpl
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
    """Collection of vertex patches for plotting.

    This class takes additional keyword arguments compared to PatchCollection:

    @param vertex_builder: A list of vertex builders to construct the visual
        vertices. This is updated if the size of the vertices is changed.
    @param size_callback: A function to be triggered after vertex sizes are
        changed. Typically this redraws the edges.
    """

    _factor = 1.0

    def __init__(self, *args, **kwargs):

        layout_df = kwargs.pop("layout")
        self._index = layout_df.index
        self._style = kwargs.pop("style", None)
        self._labels = kwargs.pop("labels", None)

        # Create patches from structured data
        patches, offsets, sizes, kwargs2 = self._init_vertex_patches(
            layout_df,
        )
        kwargs.update(kwargs2)
        kwargs["offsets"] = offsets
        kwargs["match_original"] = True

        # Pass to PatchCollection constructor
        super().__init__(patches, *args, **kwargs)

        # Compute _transforms like in _CollectionWithScales for dpi issues
        self.set_sizes(sizes)

    def get_children(self):
        children = []
        if hasattr(self, "_label_collection"):
            children.append(self._label_collection)
        return tuple(children)

    def set_figure(self, figure):
        ret = super().set_figure(figure)
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)
        for child in self.get_children():
            child.set_figure(figure)
        return ret

    def get_index(self):
        """Get the vertex index."""
        return self._index

    def get_vertex_id(self, index):
        return self._index[index]

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

    def _init_vertex_patches(self, vertex_layout_df):
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

    def _compute_label_collection(self):
        transform = self.get_offset_transform()
        trans = transform.transform

        style = (
            deepcopy(self._style.get("label", None)) if self._style is not None else {}
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
        self.set_sizes(self._sizes, self.get_figure(root=True).dpi)
        if (self._labels is not None) and (not hasattr(self, "_label_collection")):
            self._compute_label_collection()

        # NOTE: This draws the vertices first, then the labels.
        # The correct order would be vertex1->label1->vertex2->label2, etc.
        # We might fix if we manage to find a way to do it.
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
    if size_max > 0:
        size /= size_max

    if marker in ("o", "c", "circle"):
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
