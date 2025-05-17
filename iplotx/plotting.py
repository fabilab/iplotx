from typing import Union, Sequence
import pandas as pd
import matplotlib.pyplot as plt

from .typing import (
    GraphType,
    LayoutType,
    GroupingType,
)
from .network import NetworkArtist


def plot(
    network: Union[GraphType, None] = None,
    layout: Union[LayoutType, None] = None,
    grouping: Union[None, GroupingType] = None,
    vertex_labels: Union[None, list, dict, pd.Series] = None,
    edge_labels: Union[None, Sequence] = None,
    ax: Union[None, object] = None,
):
    """Plot this network using the specified layout.

    Parameters:
        network (GraphType): The network to plot. Can be a networkx or igraph graph.
        layout (Union[LayoutType, None], optional): The layout to use for plotting. If None, a layout will be looked for in the network object and, if none is found, an exception is raised. Defaults to None.
        vertex_labels (list, dict, or pandas.Series): The labels for the vertices. If None, no vertex labels
            will be drawn. If a list, the labels are taken from the list. If a dict, the keys
            should be the vertex IDs and the values should be the labels.
        ax (Union[None, object], optional): The axis to plot on. If None, a new figure and axis will be created. Defaults to None.

    Returns:
        A NetworkArtist object.
    """
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

    if grouping is not None:
        grpart = GroupingArtist(
            grouping,
            layout,
        )
        ax.add_artist(grpart)
        # Postprocess for things that require an axis (transform, etc.)
        grpart._process()
        artists.append(grpart)

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
    # FIXME: take the union of bounds
    for i, art in enumerate(artists):
        minsi, maxsi = art.get_datalim()
        if i == 0:
            mins, maxs = minsi, maxsi
        else:
            mins = pd.minimum(mins, minsi)
            maxs = pd.maximum(maxs, maxsi)
    ax.update_datalim((mins, maxs))

    # Autoscale for x/y axis limits
    ax.autoscale_view()
