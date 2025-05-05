import matplotlib.pyplot as plt

from .typing import (
    GraphType,
    LayoutType,
)
from .network import NetworkArtist


def plot(
    network : GraphType,
    layout : Union[LayoutType, None] = None,
    ax : Union[None, object] = None,
    ):
    """Plot this network using the specified layout.

    Parameters:
        network (GraphType): The network to plot. Can be a networkx or igraph graph.
        layout (Union[LayoutType, None], optional): The layout to use for plotting. If None, a layout will be looked for in the network object and, if none is found, an exception is raised. Defaults to None.
        ax (Union[None, object], optional): The axis to plot on. If None, a new figure and axis will be created. Defaults to None.

    Returns:
        A NetworkArtist object.
    """
    if ax is None:
        fig, ax = plt.subplots()

    nwkart = NetworkArtist(network, layout, ax)

    ax.add_artist(nwkart)

    nwkart._process()

    _postprocess_axis(ax, nwkart)

    return nwkart


# INTERNAL ROUTINES
def _postprocess_axis(ax, art):
    """Postprocess axis after plotting."""

    # Set new data limits
    ax.update_datalim(art.get_datalim())

    # Despine
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Autoscale for x/y axis limits
    ax.autoscale_view()
