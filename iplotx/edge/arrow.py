import numpy as np
import matplotlib as mpl
from matplotlib.patches import PathPatch


class EdgeArrowCollection(mpl.collections.PatchCollection):
    """Collection of arrow patches for plotting directed edgs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._angles = np.zeros(len(self._paths))

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


def make_arrow_patch(marker: str = "|>", width: float = 8, **kwargs):
    """Make a patch of the given marker shape and size."""
    height = kwargs.pop("height", width * 1.3)

    if marker == "|>":
        codes = ["MOVETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array([[-height, width * 0.5], [-height, -width * 0.5], [0, 0]]),
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
        overhang = kwargs.pop("overhang", 0.25)
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array(
                [
                    [0, 0],
                    [-height, -width * 0.5],
                    [-height * (1.0 - overhang), 0],
                    [-height, width * 0.5],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == ")>":
        overhang = kwargs.pop("overhang", 0.25)
        codes = ["MOVETO", "LINETO", "CURVE3", "CURVE3"]
        path = mpl.path.Path(
            np.array(
                [
                    [0, 0],
                    [-height, -width * 0.5],
                    [-height * (1.0 - overhang), 0],
                    [-height, width * 0.5],
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
    elif marker == "d":
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array(
                [
                    [-height * 0.5, width * 0.5],
                    [-height, 0],
                    [-height * 0.5, -width * 0.5],
                    [0, 0],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == "p":
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array(
                [
                    [-height, 0],
                    [0, 0],
                    [0, -width],
                    [-height, -width],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
    elif marker == "q":
        codes = ["MOVETO", "LINETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array(
                [
                    [-height, 0],
                    [0, 0],
                    [0, width],
                    [-height, width],
                ]
            ),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )

    patch = PathPatch(
        path,
        **kwargs,
    )
    return patch
