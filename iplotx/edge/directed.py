from copy import deepcopy
from math import atan2, tan, cos, pi, sin
import numpy as np
import matplotlib as mpl
from matplotlib.patches import PathPatch

from .common import _compute_loops_per_angle
from .tools.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _additional_set_methods,
)


@_forwarder(
    (
        "set_clip_path",
        "set_clip_box",
        "set_transform",
        "set_snap",
        "set_sketch_params",
        "set_figure",
        "set_animated",
        "set_picker",
    )
)
class DirectedEdgeCollection(mpl.artist.Artist):
    def __init__(self, edges, arrows, **kwargs):
        super().__init__()

        style = deepcopy(kwargs.pop("style"))
        kwargs_arrows = {
            "style": style.pop("arrow"),
        }
        if "color" in style:
            kwargs_arrows["color"] = style["color"]

        kwargs["style"] = style

        # FIXME: do we need a separate _clear_state and _process like in the network
        self._edges = UndirectedEdgeCollection(*edges, **kwargs)
        self._arrows = EdgeArrowCollection(*arrows, **kwargs_arrows)
        self._processed = False

    def get_children(self):
        artists = []
        # Collect edges first. This way vertices are on top of edges,
        # since vertices are drawn later. That is what most people expect.
        if self._edges is not None:
            artists.append(self._edges)
        if self._arrows is not None:
            artists.append(self._arrows)
        return tuple(artists)

    def get_edges(self):
        """Get UndirectedEdgeCollection artist."""
        return self._edges

    def get_arrows(self):
        """Get EdgeArrowCollection artist."""
        return self._arrows

    def _process(self):
        # Forward mpl properties to children
        # TODO sort out all of the things that need to be forwarded
        for child in self.get_children():
            # set the figure & axes on child, this ensures each artist
            # down the hierarchy knows where to draw
            if hasattr(child, "set_figure"):
                child.set_figure(self.figure)
            child.axes = self.axes

            # forward the clippath/box to the children need this logic
            # because mpl exposes some fast-path logic
            clip_path = self.get_clip_path()
            if clip_path is None:
                clip_box = self.get_clip_box()
                child.set_clip_box(clip_box)
            else:
                child.set_clip_path(clip_path)

        self._processed = True

    @_stale_wrapper
    def draw(self, renderer, *args, **kwds):
        """Draw each of the children, with some buffering mechanism."""
        if not self.get_visible():
            return

        if not self._processed:
            self._process()

        # NOTE: looks like we have to manage the zorder ourselves
        # this is kind of funny actually
        children = list(self.get_children())
        children.sort(key=lambda x: x.zorder)
        for art in children:
            art.draw(renderer, *args, **kwds)


class EdgeArrowCollection(PatchCollection):
    """Collection of arrow patches for plotting directed edgs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def stale(self):
        return super().stale

    @stale.setter
    def stale(self, val):
        PatchCollection.stale.fset(self, val)
        if val and hasattr(self, "stale_callback_post"):
            self.stale_callback_post(self)


def make_arrow_patch(marker: str = "|>", width: float = 3, **kwargs):
    """Make a patch of the given marker shape and size."""
    height = kwargs.pop("height", width * 1.5)

    if marker == "|>":
        codes = ["MOVETO", "LINETO", "LINETO"]
        path = mpl.path.Path(
            np.array([[-height, width * 0.5], [-height, -width * 0.5], [0, 0]]),
            codes=[getattr(mpl.path.Path, x) for x in codes],
            closed=True,
        )
        patch = PathPatch(
            path,
            kwargs,
        )
        return patch

    raise KeyError(f"Unknown marker: {marker}")
