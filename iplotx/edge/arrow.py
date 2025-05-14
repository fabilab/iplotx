import numpy as np
import matplotlib as mpl
from matplotlib.patches import PathPatch


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

    patch = PathPatch(
        path,
        **kwargs,
    )
    return patch

    raise KeyError(f"Unknown marker: {marker}")
