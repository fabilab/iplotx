from typing import Optional, Sequence
from contextlib import nullcontext
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from .typing import (
    GraphType,
    LayoutType,
    GroupingType,
)
from .network import NetworkArtist
from .groups import GroupingArtist
from .style import context


def plot(
    network: Optional[GraphType] = None,
    layout: Optional[LayoutType] = None,
    grouping: Optional[GroupingType] = None,
    vertex_labels: Optional[list | dict | pd.Series] = None,
    edge_labels: Optional[Sequence] = None,
    ax: Optional[mpl.axes.Axes] = None,
    style: str | dict | Sequence[str | dict] = (),
    title: Optional[str] = None,
    **kwargs,
):
    """Plot this network using the specified layout.

    Args:
        network: The network to plot. Can be a networkx or igraph graph.
        layout: The layout to use for plotting. If None, a layout will be looked for in the network object and, if none is found, an exception is raised. Defaults to None.
        vertex_labels: The labels for the vertices. If None or False, no vertex labels
            will be drawn. If a list, the labels are taken from the list. If a dict, the keys
            should be the vertex IDs and the values should be the labels. If True (a single bool value), the vertex IDs will be used as labels.
        edge_labels: The labels for the edges. If None, no edge labels will be drawn. Defaults to None.
        ax: The axis to plot on. If None, a new figure and axis will be created. Defaults to None.
        style: Apply this style for the objects to plot. This can be a sequence (e.g. list) of styles and they will be applied in order.
        title: If not None, set the axes title to this value.
        **kwargs: Additional arguments are treated as an alternate way to specify style. If both "style" and additional **kwargs
            are provided, they are both applied in that order (style, then **kwargs).

    Returns:
        A NetworkArtist object.
    """
    stylecontext = context(style, **kwargs) if style or kwargs else nullcontext()

    with stylecontext:
        if (network is None) and (grouping is None):
            raise ValueError("At least one of network or grouping must be provided.")

        if ax is None:
            fig, ax = plt.subplots()

        artists = []
        if network is not None:
            nwkart = NetworkArtist(
                network,
                layout,
                vertex_labels=vertex_labels,
                edge_labels=edge_labels,
            )
            ax.add_artist(nwkart)
            # Postprocess for things that require an axis (transform, etc.)
            nwkart._process()
            artists.append(nwkart)

            # Set normailsed layout since we have it by now
            layout = nwkart.get_layout()

        if grouping is not None:
            grpart = GroupingArtist(
                grouping,
                layout,
                network=network,
            )
            ax.add_artist(grpart)
            # Postprocess for things that require an axis (transform, etc.)
            grpart._process()
            artists.append(grpart)

        if title is not None:
            ax.set_title(title)

        _postprocess_axis(ax, artists)

        return artists


# INTERNAL ROUTINES
def _postprocess_axis(ax, artists):
    """Postprocess axis after plotting."""

    # Despine
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set new data limits
    bboxes = []
    for art in artists:
        bboxes.append(art.get_datalim(ax.transData))
    ax.update_datalim(mpl.transforms.Bbox.union(bboxes))

    # Autoscale for x/y axis limits
    ax.autoscale_view()
