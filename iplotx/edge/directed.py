from copy import deepcopy
from math import atan2, tan, cos, pi, sin
import numpy as np
import matplotlib as mpl
from matplotlib.transforms import Affine2D

from .common import _compute_loops_per_angle
from .undirected import UndirectedEdgeCollection
from .arrow import (
    EdgeArrowCollection,
    make_arrow_patch,
)
from ..utils.matplotlib import (
    _stale_wrapper,
    _forwarder,
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
    def __init__(self, edges, arrows, labels=None, **kwargs):
        super().__init__()

        self._edges = UndirectedEdgeCollection(edges, labels=labels, **kwargs)

        # NOTE: offsets are a placeholder for later
        self._arrows = EdgeArrowCollection(
            arrows,
            offsets=np.zeros((len(arrows), 2)),
            offset_transform=kwargs["transform"],
            transform=Affine2D(),
            match_original=True,
            cmap=self._edges.get_cmap(),
            norm=self._edges.norm,
        )
        self._processed = False

    def set_array(self, array):
        self._edges.set_array(array)
        self._arrows.set_array(array)

    def get_array(self):
        return self._edges.get_array()

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

    def get_paths(self):
        """Get the edge paths."""
        return self._edges.get_paths()

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

    def _set_edge_info_for_arrows(
        self,
        which="end",
        transform=None,
    ):
        """Extract the start and/or end angles of the paths to compute arrows."""
        if transform is None:
            transform = self.get_edges().get_transform()
        trans = transform.transform
        trans_inv = transform.inverted().transform

        arrow_offsets = self._arrows._offsets
        for i, epath in enumerate(self._edges._paths):
            # Offset the arrow to point to the end of the edge
            self._arrows._offsets[i] = epath.vertices[-1]

            # Rotate the arrow to point in the direction of the edge
            apath = self._arrows._paths[i]
            # NOTE: because the tip of the arrow is at (0, 0) in patch space,
            # in theory it will rotate around that point already
            v2 = trans(epath.vertices[-1])
            v1 = trans(epath.vertices[-2])
            dv = v2 - v1
            theta = atan2(*(dv[::-1]))
            theta_old = self._arrows._angles[i]
            dtheta = theta - theta_old
            mrot = np.array([[cos(dtheta), sin(dtheta)], [-sin(dtheta), cos(dtheta)]])
            apath.vertices = apath.vertices @ mrot
            self._arrows._angles[i] = theta

    @_stale_wrapper
    def draw(self, renderer, *args, **kwds):
        """Draw each of the children, with some buffering mechanism."""
        if not self.get_visible():
            return

        if not self._processed:
            self._process()

        # We should manage zorder ourselves, but we need to compute
        # the new offsets and angles of arrows from the edges before drawing them
        self._edges.draw(renderer, *args, **kwds)
        self._set_edge_info_for_arrows(which="end")
        self._arrows.draw(renderer, *args, **kwds)
