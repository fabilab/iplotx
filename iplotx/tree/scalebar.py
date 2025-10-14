from typing import (
    Optional,
    Any,
)
import numbers
import numpy as np
import matplotlib as mpl
from matplotlib import (
    _api,
    offsetbox,
)
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
from matplotlib.transforms import BboxTransformTo
from matplotlib.offsetbox import (
    VPacker,
    DrawingArea,
    TextArea,
)


class TreeScalebarArtist(mpl.artist.Artist):
    codes = Legend.codes
    zorder = 5

    def __init__(
        self,
        tree_artist,
        length: float,
        loc: str = "lower right",
        pad: float = 0.1,
        bbox_to_anchor: Optional[Any] = None,  # bbox to which the legend will be anchored
        bbox_transform: Optional[mpl.transforms.Transform] = None,  # transform for the bbox
        frameon: bool = True,
        facecolor=None,
        edgecolor="black",
        framealpha: Optional[float] = None,
        fancybox: bool = True,
        fontsize: Optional[float] = None,
        labelspacing: Optional[float] = None,
        handleheight: Optional[float] = None,
        handlelength: Optional[float] = None,
        botderaxespad: Optional[float] = None,
        mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        ax = tree_artist.axes
        if ax is None:
            raise ValueError("TreeScalebarArtist requires the tree to be in an axes.")
        self.axes = ax
        self.parent = ax

        # Choose the artist handle to use for the scale bar. Since it refers to edge
        # length, we use the edge collection.
        handle = tree_artist.get_edges()

        self.prop = FontProperties(size=mpl._val_or_rc(fontsize, "legend.fontsize"))
        self._fontsize = self.prop.get_size_in_points()

        self.length = length
        self.loc = loc
        self.pad = pad
        self.kwargs = kwargs

        self.labelspacing = mpl._val_or_rc(labelspacing, "legend.labelspacing")
        self.handleheight = mpl._val_or_rc(handleheight, "legend.handleheight")
        self.handlelength = mpl._val_or_rc(handlelength, "legend.handlelength")
        self.borderaxespad = mpl._val_or_rc(botderaxespad, "legend.borderaxespad")

        # _scale_box is a VPacker instance that contains the bar and its text
        self._scale_box = None

        ax.add_artist(self)
        self.set_figure(ax.get_figure(root=False))

        self._mode = mode
        self.set_bbox_to_anchor(bbox_to_anchor, bbox_transform)

        self.scalebarPatch = FancyBboxPatch(
            xy=(0, 0),
            width=1,
            height=1,
            facecolor=facecolor,
            edgecolor=edgecolor,
            # If shadow is used, default to alpha=1 (#8943).
            alpha=(framealpha if framealpha is not None else 1),
            # The width and height of the scalebarPatch will be set (in draw())
            # to the length that includes the padding. Thus we set pad=0 here.
            boxstyle=("round,pad=0,rounding_size=0.2" if fancybox else "square,pad=0"),
            mutation_scale=self._fontsize,
            snap=True,
            visible=frameon,
        )
        self._set_artist_props(self.scalebarPatch)

        # init with null renderer
        self._init_scale_box(handle, self.length)

        # Set legend location
        self.set_loc(loc)

    def _set_artist_props(self, a):
        """
        Set the boilerplate props for artists added to Axes.
        """
        a.set_figure(self.get_figure(root=False))
        a.axes = self.axes

        a.set_transform(self.get_transform())

    def set_bbox_to_anchor(self, bbox, transform=None):
        """
        Set the bbox that the legend will be anchored to.

        Parameters
        ----------
        bbox : `~matplotlib.transforms.BboxBase` or tuple
            The bounding box can be specified in the following ways:

            - A `.BboxBase` instance
            - A tuple of ``(left, bottom, width, height)`` in the given
              transform (normalized axes coordinate if None)
            - A tuple of ``(left, bottom)`` where the width and height will be
              assumed to be zero.
            - *None*, to remove the bbox anchoring, and use the parent bbox.

        transform : `~matplotlib.transforms.Transform`, optional
            A transform to apply to the bounding box. If not specified, this
            will use a transform to the bounding box of the parent.
        """
        if bbox is None:
            self._bbox_to_anchor = None
            return
        elif isinstance(bbox, BboxBase):
            self._bbox_to_anchor = bbox
        else:
            try:
                lenb = len(bbox)
            except TypeError as err:
                raise ValueError(f"Invalid bbox: {bbox}") from err

            if lenb == 2:
                bbox = [bbox[0], bbox[1], 0, 0]

            self._bbox_to_anchor = Bbox.from_bounds(*bbox)

        if transform is None:
            transform = BboxTransformTo(self.parent.bbox)

        self._bbox_to_anchor = TransformedBbox(self._bbox_to_anchor, transform)
        self.stale = True

    def set_loc(self, loc=None):
        """
        Set the location of the legend.

        .. versionadded:: 3.8

        Parameters
        ----------
        %(_legend_kw_set_loc_doc)s
        """
        loc0 = loc
        self._loc_used_default = loc is None
        if loc is None:
            loc = mpl.rcParams["legend.loc"]

        type_err_message = f"loc must be string, coordinate tuple, or an integer 0-10, not {loc!r}"

        # handle outside legends:
        self._outside_loc = None
        if isinstance(loc, str):
            if loc.split()[0] == "outside":
                # strip outside:
                loc = loc.split("outside ")[1]
                # strip "center" at the beginning
                self._outside_loc = loc.replace("center ", "")
                # strip first
                self._outside_loc = self._outside_loc.split()[0]
                locs = loc.split()
                if len(locs) > 1 and locs[0] in ("right", "left"):
                    # locs doesn't accept "left upper", etc, so swap
                    if locs[0] != "center":
                        locs = locs[::-1]
                    loc = locs[0] + " " + locs[1]
            # check that loc is in acceptable strings
            loc = _api.check_getitem(self.codes, loc=loc)
        elif np.iterable(loc):
            # coerce iterable into tuple
            loc = tuple(loc)
            # validate the tuple represents Real coordinates
            if len(loc) != 2 or not all(isinstance(e, numbers.Real) for e in loc):
                raise ValueError(type_err_message)
        elif isinstance(loc, int):
            # validate the integer represents a string numeric value
            if loc < 0 or loc > 10:
                raise ValueError(type_err_message)
        else:
            # all other cases are invalid values of loc
            raise ValueError(type_err_message)

        if self._outside_loc:
            raise ValueError(
                f"'outside' option for loc='{loc0}' keyword argument only works for figure legends"
            )

        tmp = self._loc_used_default
        self._set_loc(loc)
        self._loc_used_default = tmp  # ignore changes done by _set_loc

    def _set_loc(self, loc):
        # find_offset function will be provided to _scale_box and
        # _scale_box will draw itself at the location of the return
        # value of the find_offset.
        self._loc_used_default = False
        self._loc_real = loc
        self.stale = True
        self._scale_box.set_offset(self._findoffset)

    def _get_loc(self):
        return self._loc_real

    _loc = property(_get_loc, _set_loc)

    def _findoffset(self, width, height, xdescent, ydescent, renderer):
        """Helper function to locate the legend."""

        if self._loc == 0:  # "best".
            x, y = self._find_best_position(width, height, renderer)
        elif self._loc in Legend.codes.values():  # Fixed location.
            bbox = Bbox.from_bounds(0, 0, width, height)
            x, y = self._get_anchored_bbox(self._loc, bbox, self.get_bbox_to_anchor(), renderer)
        else:  # Axes or figure coordinates.
            fx, fy = self._loc
            bbox = self.get_bbox_to_anchor()
            x, y = bbox.x0 + bbox.width * fx, bbox.y0 + bbox.height * fy

        return x + xdescent, y + ydescent

    def get_bbox_to_anchor(self):
        """Return the bbox that the legend will be anchored to."""
        if self._bbox_to_anchor is None:
            return self.parent.bbox
        else:
            return self._bbox_to_anchor

    def _get_anchored_bbox(self, loc, bbox, parentbbox, renderer):
        """
        Place the *bbox* inside the *parentbbox* according to a given
        location code. Return the (x, y) coordinate of the bbox.

        Parameters
        ----------
        loc : int
            A location code in range(1, 11). This corresponds to the possible
            values for ``self._loc``, excluding "best".
        bbox : `~matplotlib.transforms.Bbox`
            bbox to be placed, in display coordinates.
        parentbbox : `~matplotlib.transforms.Bbox`
            A parent box which will contain the bbox, in display coordinates.
        """
        pad = self.borderaxespad * renderer.points_to_pixels(self._fontsize)
        return offsetbox._get_anchored_bbox(loc, bbox, parentbbox, pad, pad)

    @staticmethod
    def _make_scale_handle(handle, handlebox):
        """Make a proxy handle for the scale bar."""
        kwargs = {
            "edgecolor": "black",
            "linestyle": "solid",
            "linewidth": 2,
        }
        for prop in kwargs:
            coll_res = getattr(handle, f"get_{prop}")()
            if len(coll_res) > 0:
                kwargs[prop] = coll_res[0]

        kwargs["color"] = kwargs.pop("edgecolor")

        line = Line2D(
            [0, 1],
            [0, 0],
            **kwargs,
        )
        artists = [line]

        # TODO: add more artists for the caplines

        for a in artists:
            handlebox.add_artist(a)

        return artists[0]

    def _init_scale_box(self, handle, length):
        """
        Initialize the scale_box. The scale_box is an instance of
        the OffsetBox, which is packed with legend handles and
        texts. Once packed, their location is calculated during the
        drawing time.
        """

        fontsize = self._fontsize
        label = f"{length:.2f}"

        orig_handle = handle

        # scale_box is a VPacker, vertically packed with:
        # - handlebox: a DrawingArea which contains the scalebar handle.
        # - labelbox: a TextArea which contains the scalebar text.

        # The approximate height and descent of text. These values are
        # only used for plotting the legend handle.
        descent = 0.35 * fontsize * (self.handleheight - 0.7)  # heuristic.
        height = fontsize * self.handleheight - descent
        # each handle needs to be drawn inside a box of (x, y, w, h) =
        # (0, -descent, width, height).  And their coordinates should
        # be given in the display coordinates.

        # The transformation of each handle will be automatically set
        # to self.get_transform(). If the artist does not use its
        # default transform (e.g., Collections), you need to
        # manually set their transform to the self.get_transform().
        # legend_handler_map = self.get_legend_handler_map()

        textbox = TextArea(
            label,
            multilinebaseline=True,
            textprops=dict(
                verticalalignment="baseline", horizontalalignment="left", fontproperties=self.prop
            ),
        )
        handlebox = DrawingArea(
            width=self.handlelength * fontsize, height=height, xdescent=0.0, ydescent=descent
        )

        handle = self._make_scale_handle(orig_handle, handlebox)

        # Create the artist for the legend which represents the
        # original artist/handle.
        # handle_list.append(handler.legend_artist(self, orig_handle,
        #                                         fontsize, handlebox))

        itembox = VPacker(
            pad=0,
            sep=self.labelspacing * fontsize,
            children=[handlebox, textbox],
            align="baseline",
        )

        # mode = "expand" if self._mode == "expand" else "fixed"
        self._scale_box = itembox
        self._scale_box.set_figure(self.get_figure(root=False))
        self._scale_box.axes = self.axes
        self.text = label
        self.scale_handle = handle

    @mpl.artist.allow_rasterization
    def draw(self, renderer):
        if not self.get_visible():
            return

        renderer.open_group("scalebar", gid=self.get_gid())

        fontsize = renderer.points_to_pixels(self._fontsize)

        # update the location and size of the legend. This needs to
        # be done in any case to clip the figure right.
        bbox = self._scale_box.get_window_extent(renderer)
        self.scalebarPatch.set_bounds(bbox.bounds)
        self.scalebarPatch.set_mutation_scale(fontsize)

        # label = f"{self.length:.2f}"

        self.scalebarPatch.draw(renderer)
        self._scale_box.draw(renderer)

        renderer.close_group("legend")
        self.stale = False
