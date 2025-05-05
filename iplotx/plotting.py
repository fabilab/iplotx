from .typing import (
    GraphType,
    LayoutType,
)
from .network import NetworkPlot


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
        A NetworkPlot object.
    """
    if ax is None:
        fig, ax = plt.subplots()

    nwkp = NetworkPlot(network, layout, ax)

    return nwkp
