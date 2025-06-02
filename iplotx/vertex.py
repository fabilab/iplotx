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
        super().__init__(*args, **kwargs)
        self.set_sizes(self._get_sizes_from_paths())

    def _get_sizes_from_paths(self):
        """Get vertex sizes.

        If width and height are unequal, get the largest of the two.

        @return: An array of vertex sizes.
        """
        import numpy as np

        sizes = []
        for path in self.get_paths():
            bbox = path.get_extents()
            mins, maxs = bbox.min, bbox.max
            width, height = maxs - mins
            size = max(width, height)
            sizes.append(size)
        return np.array(sizes)

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
            scale = self._sizes**0.5 * dpi / 72.0 * self._factor
            self._transforms[:, 0, 0] = scale
            self._transforms[:, 1, 1] = scale
            self._transforms[:, 2, 2] = 1.0
        self.stale = True

    get_size = get_sizes
    set_size = set_sizes

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
        super().draw(renderer)


def make_patch(marker: str, size, **kwargs):
    """Make a patch of the given marker shape and size."""
    forbidden_props = ["label", "cmap", "norm"]
    for prop in forbidden_props:
        if prop in kwargs:
            kwargs.pop(prop)

    if np.isscalar(size):
        size = float(size)
        size = (size, size)

    if marker in ("o", "circle"):
        return Circle((0, 0), size[0] / 2, **kwargs)
    elif marker in ("s", "square", "r", "rectangle"):
        return Rectangle((-size[0] / 2, -size[1] / 2), size[0], size[1], **kwargs)
    elif marker in ("^", "triangle"):
        return RegularPolygon((0, 0), numVertices=3, radius=size[0] / 2, **kwargs)
    elif marker in ("d", "diamond"):
        return make_patch("s", size[0], angle=45, **kwargs)
    elif marker in ("v", "triangle_down"):
        return RegularPolygon(
            (0, 0), numVertices=3, radius=size[0] / 2, orientation=np.pi, **kwargs
        )
    elif marker in ("e", "ellipse"):
        return Ellipse((0, 0), size[0] / 2, size[1] / 2, **kwargs)
    raise KeyError(f"Unknown marker: {marker}")
