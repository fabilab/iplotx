import numpy as np
import matplotlib as mpl

from ..utils.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _additional_set_methods,
)


class LabelCollection(mpl.artist.Artist):
    def __init__(self, labels, style=None):
        self._labels = labels
        self._style = style
        super().__init__()

    def _create_labels(self):
        style = self._style if self._style is not None else {}

        arts = []
        for label in self._labels:
            art = mpl.text.Text(
                0,
                0,
                label,
                transform=self.axes.transData,
                **style,
            )
            art.set_figure(self.figure)
            art.axes = self.axes
            arts.append(art)
        self._labels = arts

    def get_children(self):
        return self._labels

    def set_offsets(self, offsets):
        for art, offset in zip(self._labels, offsets):
            art.set_position((offset[0], offset[1]))

    @_stale_wrapper
    def draw(self, renderer, *args, **kwds):
        """Draw each of the children, with some buffering mechanism."""
        if not self.get_visible():
            return

        # We should manage zorder ourselves, but we need to compute
        # the new offsets and angles of arrows from the edges before drawing them
        for art in self.get_children():
            art.draw(renderer, *args, **kwds)
