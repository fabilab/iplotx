from typing import (
    Any,
)
import numpy as np
from matplotlib import (
    _api,
)
from matplotlib import collections as mcoll
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerErrorbar
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (
    HPacker,
    VPacker,
    DrawingArea,
    TextArea,
)


def _update_prop(legend_artist, orig_handle):
    # NOTE: This is de facto a bug in mpl, because Line2D.set_linestyle()
    # does two things: it reformats tuple-style dashing, and it sets the
    # artist as stale. We want to do the former only here, so we reset
    # the artist as the original stale state after calling it.
    stale_orig = legend_artist.stale
    legend_artist.set_linestyle(orig_handle.get_linestyle()[0])
    legend_artist.stale = stale_orig

    # These other properties can be set directly.
    legend_artist._linewidth = orig_handle.get_linewidth()[0]
    legend_artist._color = orig_handle.get_edgecolor()[0]
    legend_artist._gapcolor = orig_handle._gapcolor


class TreeScalebarArtist(Legend):
    def __init__(
        self,
        treeartist,
        layout: str = "horizontal",
        frameon: bool = False,
        label_format: str = ".2f",
        **kwargs,
    ):
        handles = [treeartist.get_edges()]
        labels = [""]
        self._layout = layout
        self._treeartist = treeartist
        self._label_format = label_format

        if layout == "vertical":
            handler_kwargs = dict(xerr_size=0, yerr_size=1)
        else:
            handler_kwargs = dict(xerr_size=1)
        handler = TreeLegendHandler(
            update_func=_update_prop,
            **handler_kwargs,
        )

        super().__init__(
            treeartist.axes,
            handles,
            labels,
            handler_map={handles[0]: handler},
            frameon=frameon,
            **kwargs,
        )

    def _init_legend_box(self, handles, labels, markerfirst=True):
        """
        Create the legend box.

        This is a modified version of the original Legend._init_legend_box
        method to accommodate a scale bar.
        """

        fontsize = self._fontsize

        # legend_box is a HPacker, horizontally packed with columns.
        # Each column is a VPacker, vertically packed with legend items.
        # Each legend item is a HPacker packed with:
        # - handlebox: a DrawingArea which contains the legend handle.
        # - labelbox: a TextArea which contains the legend text.

        text_list = []  # the list of text instances
        handle_list = []  # the list of handle instances
        handles_and_labels = []

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
        legend_handler_map = self.get_legend_handler_map()

        for orig_handle, label in zip(handles, labels):
            handler = self.get_legend_handler(legend_handler_map, orig_handle)
            if handler is None:
                _api.warn_external(
                    "Legend does not support handles for "
                    f"{type(orig_handle).__name__} "
                    "instances.\nA proxy artist may be used "
                    "instead.\nSee: https://matplotlib.org/"
                    "stable/users/explain/axes/legend_guide.html"
                    "#controlling-the-legend-entries"
                )
                # No handle for this artist, so we just defer to None.
                handle_list.append(None)
            else:
                handlebox = DrawingArea(
                    width=self.handlelength * fontsize,
                    height=height,
                    xdescent=0.0,
                    ydescent=descent,
                )
                # Create the artist for the legend which represents the
                # original artist/handle.
                handle_list.append(handler.legend_artist(self, orig_handle, fontsize, handlebox))

                # The scale bar line is in this handle
                bar_handle = handle_list[-1]
                label = self._get_label_from_bar_handle(bar_handle)

                textbox = TextArea(
                    label,
                    multilinebaseline=True,
                    textprops=dict(
                        verticalalignment="baseline",
                        horizontalalignment="left",
                        fontproperties=self.prop,
                    ),
                )
                text_list.append(textbox._text)

                handles_and_labels.append((handlebox, textbox))

        columnbox = []
        # array_split splits n handles_and_labels into ncols columns, with the
        # first n%ncols columns having an extra entry.  filter(len, ...)
        # handles the case where n < ncols: the last ncols-n columns are empty
        # and get filtered out.
        for handles_and_labels_column in filter(
            len, np.array_split(handles_and_labels, self._ncols)
        ):
            # pack handlebox and labelbox into itembox
            if self._layout == "vertical":
                itempacker = HPacker
            else:
                itempacker = VPacker
            itemboxes = [
                itempacker(
                    pad=0,
                    sep=self.handletextpad * fontsize,
                    children=[h, t] if markerfirst else [t, h],
                    align="center",
                )
                for h, t in handles_and_labels_column
            ]
            # pack columnbox
            alignment = "baseline" if markerfirst else "right"
            columnbox.append(
                VPacker(
                    pad=0, sep=self.labelspacing * fontsize, align=alignment, children=itemboxes
                )
            )

        mode = "expand" if self._mode == "expand" else "fixed"
        sep = self.columnspacing * fontsize
        self._legend_handle_box = HPacker(
            pad=0, sep=sep, align="baseline", mode=mode, children=columnbox
        )
        self._legend_title_box = TextArea("")
        self._legend_box = VPacker(
            pad=self.borderpad * fontsize,
            sep=self.labelspacing * fontsize,
            align=self._alignment,
            children=[self._legend_title_box, self._legend_handle_box],
        )
        self._legend_box.set_figure(self.get_figure(root=False))
        self._legend_box.axes = self.axes
        self.texts = text_list
        self.legend_handles = handle_list

    def _get_label_from_bar_handle(self, bar_handle: Any) -> str:
        # Extract the x coordinates of the scale bar
        p0, p1 = bar_handle.get_segments()[0]

        bar_trans = bar_handle.get_transform()
        data_trans = self.parent.transData
        composite_trans = data_trans.inverted() + bar_trans

        p0_data = composite_trans.transform(p0)
        p1_data = composite_trans.transform(p1)
        distance = np.linalg.norm(p1_data - p0_data)
        label = format(distance, self._label_format)
        return label

    def draw(self, renderer):
        bar_handle = self.legend_handles[0]
        label = self._get_label_from_bar_handle(bar_handle)

        text_handle = (
            self._legend_box.get_children()[1].get_children()[0].get_children()[0].get_children()[1]
        )
        # Bypass stale=True (we are already redrawing)
        text_handle.set_text(label)

        super().draw(renderer)


class TreeLegendHandler(HandlerErrorbar):
    def __init__(self, marker_size=6, **kw):
        self.marker_size = marker_size
        super().__init__(**kw)

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        # docstring inherited
        plotline = orig_handle

        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent, width, height, fontsize)
        ydata = np.full_like(xdata, (height - ydescent) / 2)

        xdata_marker = np.asarray(xdata_marker)
        ydata_marker = np.asarray(ydata[: len(xdata_marker)])

        xerr_size, yerr_size = self.get_err_size(
            legend, xdescent, ydescent, width, height, fontsize
        )

        if legend._layout == "vertical":
            xdata, ydata = np.array(
                [
                    ((x, y - yerr_size), (x, y + yerr_size))
                    for x, y in zip(xdata_marker, ydata_marker)
                ]
            ).T

        legline = Line2D(xdata, ydata)

        legline_marker = Line2D(xdata_marker, ydata_marker)

        # when plotlines are None (only errorbars are drawn), we just
        # make legline invisible.
        if plotline is None:
            legline.set_visible(False)
            legline_marker.set_visible(False)
        else:
            self.update_prop(legline, plotline, legend)

            legline.set_drawstyle("default")
            legline.set_marker("none")

            self.update_prop(legline_marker, plotline, legend)
            legline_marker.set_linestyle("None")

            if legend.markerscale != 1:
                newsz = legline_marker.get_markersize() * legend.markerscale
                legline_marker.set_markersize(newsz)

        handle_barlinecols = []
        handle_caplines = []

        if legend._layout != "vertical":
            verts = [
                ((x - xerr_size, y), (x + xerr_size, y)) for x, y in zip(xdata_marker, ydata_marker)
            ]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, plotline, legend)
            handle_barlinecols.append(coll)

            # Always show the cap lines
            if True:
                capline_left = Line2D(xdata_marker - xerr_size, ydata_marker)
                capline_right = Line2D(xdata_marker + xerr_size, ydata_marker)
                self.update_prop(capline_left, plotline, legend)
                self.update_prop(capline_right, plotline, legend)
                capline_left.set_marker("|")
                capline_right.set_marker("|")

                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)

        else:
            verts = [
                ((x, y - yerr_size), (x, y + yerr_size)) for x, y in zip(xdata_marker, ydata_marker)
            ]
            coll = mcoll.LineCollection(verts)
            self.update_prop(coll, plotline, legend)
            handle_barlinecols.append(coll)

            # Always show the cap lines
            if True:
                capline_left = Line2D(xdata_marker, ydata_marker - yerr_size)
                capline_right = Line2D(xdata_marker, ydata_marker + yerr_size)
                self.update_prop(capline_left, plotline, legend)
                self.update_prop(capline_right, plotline, legend)
                capline_left.set_marker("_")
                capline_right.set_marker("_")

                handle_caplines.append(capline_left)
                handle_caplines.append(capline_right)

        artists = [
            *handle_barlinecols,
            *handle_caplines,
            legline,
            legline_marker,
        ]
        for artist in artists:
            artist.set_transform(trans)
        return artists
