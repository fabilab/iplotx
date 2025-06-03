from copy import deepcopy
import numpy as np
import matplotlib as mpl

from .utils.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _additional_set_methods,
)


class LabelCollection(mpl.artist.Artist):
    def __init__(self, labels, style=None, offsets=None, transform=None):
        self._labels = labels
        self._offsets = offsets if offsets is not None else np.zeros((len(labels), 2))
        self._style = style
        super().__init__()
        if transform is not None:
            self.set_transform(transform)

    def _create_artists(self):
        style = deepcopy(self._style) if self._style is not None else {}

        forbidden_props = ["rotate"]
        for prop in forbidden_props:
            if prop in style:
                del style[prop]

        arts = []
        for i, label in enumerate(self._labels):
            art = mpl.text.Text(
                self._offsets[i][0],
                self._offsets[i][1],
                label,
                transform=self.get_transform(),
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
        stale = True

    def set_rotations(self, rotations):
        for art, rotation in zip(self._labels, rotations):
            rot_deg = 180.0 / np.pi * rotation
            # Force the font size to be upwards
            rot_deg = ((rot_deg + 90) % 180) - 90
            art.set_rotation(rot_deg)
        stale = True

    @_stale_wrapper
    def draw(self, renderer, *args, **kwds):
        """Draw each of the children, with some buffering mechanism."""
        if not self.get_visible():
            return

        # We should manage zorder ourselves, but we need to compute
        # the new offsets and angles of arrows from the edges before drawing them
        for art in self.get_children():
            art.draw(renderer, *args, **kwds)
