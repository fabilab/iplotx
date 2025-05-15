import numpy as np
import matplotlib as mpl

from ..tools.matplotlib import (
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
class LabelCollection(mpl.artist.Artist):
    def __init__(self, labels, offsets=None, style=None):
        self._create_labels(labels, offsets, style)

    def _create_labels(self, labels, offsets, style):
        if offsets is None:
            offsets = np.zeros((len(labels), 2))
        if style is None:
            style = {}

        arts = []
        for label, offset in zip(labels, offsets):
            art = mpl.text.Text(
                offset[0],
                offset[1],
                label,
                transform=self.axes.transData,
                **style,
            )
            arts.append(art)
        self._labels = arts

    def get_children(self):
        return self._labels

    def set_offsets(self, offsets):
        for art, offset in zip(self._labels, offsets):
            art.set_position((offset[0], offset[1]))

    @property
    def stale(self):
        return super().stale

    @_stale_wrapper
    def draw(self, renderer, *args, **kwds):
        """Draw each of the children, with some buffering mechanism."""
        if not self.get_visible():
            return

        # We should manage zorder ourselves, but we need to compute
        # the new offsets and angles of arrows from the edges before drawing them
        for art in self.get_children():
            art.draw(renderer, *args, **kwargs)
