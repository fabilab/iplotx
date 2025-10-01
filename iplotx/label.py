"""
Module for label collection in iplotx.
"""

from typing import (
    Optional,
    Sequence,
)
import numpy as np
import pandas as pd
import matplotlib as mpl

from .style import (
    rotate_style,
    copy_with_deep_values,
)
from .utils.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _get_label_width_height,
)


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
class LabelCollection(mpl.artist.Artist):
    """Collection of labels for iplotx with styles.

    NOTE: This class is not a subclass of `mpl.collections.Collection`, although in some ways items
    behaves like one. It is named LabelCollection quite literally to indicate it contains a list of
    labels for vertices, edges, etc.
    """

    def __init__(
        self,
        labels: pd.Series,
        style: Optional[dict[str, dict]] = None,
        offsets: Optional[np.ndarray] = None,
        transform: mpl.transforms.Transform = mpl.transforms.IdentityTransform(),
    ) -> None:
        """Initialize a collection of labels.

        Parameters:
            labels: A sequence of labels to be displayed.
            style: A dictionary of styles to apply to the labels. The keys are style properties.
            offsets: A sequence of offsets for each label, specifying the position of the label.
            transform: A transform to apply to the labels. This is usually ax.transData.
        """
        self._labels = labels
        self._offsets = offsets if offsets is not None else np.zeros((len(labels), 2))
        self._style = style
        super().__init__()

        self.set_transform(transform)
        self._create_artists()

    def get_children(self) -> tuple[mpl.artist.Artist]:
        """Get the children of this artist, which are the label artists."""
        return tuple(self._labelartists)

    def set_figure(self, fig) -> None:
        """Set the figure of this artist.

        Parameters:
            fig: The figure to set.
        """
        super().set_figure(fig)
        for child in self.get_children():
            child.set_figure(fig)
        self._update_offsets(dpi=fig.dpi)

    @property
    def axes(self):
        return mpl.artist.Artist.axes.__get__(self)

    @axes.setter
    def axes(self, new_axes):
        mpl.artist.Artist.axes.__set__(self, new_axes)
        for child in self.get_children():
            child.axes = new_axes

    def set_transform(self, transform: mpl.transforms.Transform) -> None:
        """Set the transform for this artist and children.

        Parameters:
            transform: The transform to set.
        """
        super().set_transform(transform)
        if hasattr(self, "_labelartists"):
            for art in self._labelartists:
                art.set_transform(transform)

    def get_texts(self):
        """Get the texts of the labels."""
        return [child.get_text() for child in self.get_children()]

    get_text = get_texts

    def _get_margins_with_dpi(self, dpi: float = 72.0) -> np.ndarray:
        return self._margins * dpi / 72.0

    def _create_artists(self) -> None:
        style = copy_with_deep_values(self._style) if self._style is not None else {}
        transform = self.get_transform()

        margins = []

        forbidden_props = ["rotate"]
        for prop in forbidden_props:
            if prop in style:
                del style[prop]

        arts = []
        for i, (anchor_id, label) in enumerate(self._labels.items()):
            stylei = rotate_style(style, index=i, key=anchor_id)
            # Margins are handled separately
            hmargin = stylei.pop("hmargin", 0.0)
            vmargin = stylei.pop("vmargin", 0.0)
            margins.append((hmargin, vmargin))

            # Initially, ignore autoalignment since we do not know the
            # rotations
            if stylei.get("horizontalalignment") == "auto":
                stylei["horizontalalignment"] = "center"

            art = mpl.text.Text(
                self._offsets[i][0],
                self._offsets[i][1],
                label,
                transform=transform,
                rotation_mode="anchor",
                **stylei,
            )
            arts.append(art)
        self._labelartists = arts
        self._margins = np.array(margins)
        self._rotations = np.array([np.pi / 180 * art.get_rotation() for art in arts])

    def _update_offsets(self, dpi: float = 72.0) -> None:
        """Update offsets including margins."""
        self.set_offsets(self._offsets, dpi=dpi)

    def get_offsets(self, with_margins: bool = False) -> np.ndarray:
        """Get the positions (offsets) of the labels."""
        if not with_margins:
            return self._offsets
        else:
            return np.array(
                [art.get_position() for art in self._labelartists],
            )

    def _adjust_offsets_for_margins(self, offsets, dpi=72.0):
        margins = self._get_margins_with_dpi(dpi=dpi)
        if (margins != 0).any():
            transform = self.get_transform()
            trans = transform.transform
            trans_inv = transform.inverted().transform

            # Add margins *before* applying the rotation
            margins_rot = np.empty_like(margins)
            rotations = self.get_rotations()
            vrot = [np.cos(rotations), np.sin(rotations)]
            margins_rot[:, 0] = margins[:, 0] * vrot[0] - margins[:, 1] * vrot[1]
            margins_rot[:, 1] = margins[:, 0] * vrot[1] + margins[:, 1] * vrot[0]
            offsets = trans_inv(trans(offsets) + margins_rot)
        return offsets

    def set_offsets(self, offsets, dpi: float = 72.0) -> None:
        """Set positions (offsets) of the labels.

        Parameters:
            offsets: A sequence of offsets for each label, specifying the position of the label.
        """
        self._offsets = np.asarray(offsets)
        offsets_with_margins = self._adjust_offsets_for_margins(offsets, dpi=dpi)
        for art, offset in zip(self._labelartists, offsets_with_margins):
            art.set_position((offset[0], offset[1]))

    def get_rotations(self) -> np.ndarray:
        """Get the rotations of the labels in radians."""
        return self._rotations

    def set_rotations(self, rotations: Sequence[float]) -> None:
        """Set the rotations of the labels.

        Parameters:
            rotations: A sequence of rotations in radians for each label.
        """
        self._rotations = np.asarray(rotations)
        ha = self._style.get("horizontalalignment", "center")
        for art, rotation in zip(self._labelartists, rotations):
            rot_deg = 180.0 / np.pi * rotation
            # Force the font size to be upwards
            if ha == "auto":
                if -90 <= np.round(rot_deg, 3) < 90:
                    art.set_horizontalalignment("left")
                else:
                    art.set_horizontalalignment("right")
            rot_deg_new = ((rot_deg + 90) % 180) - 90
            art.set_rotation(rot_deg_new)

    def get_datalim(self, transData=None) -> mpl.transforms.Bbox:
        """Get the data limits of the labels."""
        bboxes = self.get_datalims_children(transData=transData)
        bbox = mpl.transforms.Bbox.union(bboxes)
        return bbox

    def get_datalims_children(self, transData=None) -> Sequence[mpl.transforms.Bbox]:
        """Get the data limits  of the children of this artist."""
        dpi = self.figure.dpi if self.figure is not None else 72.0
        if transData is None:
            transData = self.get_transform()
        trans = transData.transform
        trans_inv = transData.inverted().transform
        bboxes = []
        for art, rot in zip(self._labelartists, self.get_rotations()):
            # These are in figure points
            textprops = ("text", "fontsize")
            props = art.properties()
            props = {key: props[key] for key in textprops}
            props["dpi"] = dpi
            props["hpadding"] = 12
            props["vpadding"] = 12
            width, height = _get_label_width_height(**props)

            # Positions are in data coordinates
            # and include the margins (it's the actual position of the
            # text anchor)
            pos_data = art.get_position()

            # Four corners
            dh = width / 2 * np.array([-np.sin(rot), np.cos(rot)])
            dw = height * np.array([np.cos(rot), np.sin(rot)])
            c1 = trans_inv(trans(pos_data) + dh)
            c2 = trans_inv(trans(pos_data) - dh)
            if art.get_horizontalalignment() == "right":
                c3 = trans_inv(trans(pos_data) - dw + dh)
                c4 = trans_inv(trans(pos_data) - dw - dh)
            else:
                c3 = trans_inv(trans(pos_data) + dw + dh)
                c4 = trans_inv(trans(pos_data) + dw - dh)
            bbox = mpl.transforms.Bbox.null()
            bbox.update_from_data_xy([c1, c2, c3, c4], ignore=True)
            bboxes.append(bbox)
        return bboxes

    @_stale_wrapper
    def draw(self, renderer) -> None:
        """Draw each of the children, with some buffering mechanism."""
        if not self.get_visible():
            return

        self._update_offsets(dpi=renderer.dpi)

        # We should manage zorder ourselves, but we need to compute
        # the new offsets and angles of arrows from the edges before drawing them
        for art in self.get_children():
            art.draw(renderer)
