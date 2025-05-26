import numpy as np
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_sizes(self):
        """Same as get_size."""
        return self.get_size()

    def get_size(self):
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

    def set_size(self, sizes):
        """Set vertex sizes.

        This rescales the current vertex symbol/path linearly, using this
        value as the largest of width and height.

        @param sizes: A sequence of vertex sizes or a single size.
        """
        paths = self._paths
        try:
            iter(sizes)
        except TypeError:
            sizes = [sizes] * len(paths)

        sizes = list(sizes)
        current_sizes = self.get_sizes()
        for path, cursize in zip(paths, current_sizes):
            # Circular use of sizes
            size = sizes.pop(0)
            sizes.append(size)
            # Rescale the path for this vertex
            path.vertices *= size / cursize

        self.stale = True

    def set_sizes(self, sizes):
        """Same as set_size."""
        self.set_size(sizes)

    @property
    def stale(self):
        return super().stale

    @stale.setter
    def stale(self, val):
        PatchCollection.stale.fset(self, val)
        if val and hasattr(self, "stale_callback_post"):
            self.stale_callback_post(self)


def make_patch(marker: str, size, **kwargs):
    """Make a patch of the given marker shape and size."""
    forbidden_props = ["label"]
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
