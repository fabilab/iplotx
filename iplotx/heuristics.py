import numpy as np
import pandas as pd

from .importing import igraph, networkx
from .typing import GraphType


def network_library(
    network: GraphType,
) -> str:
    if igraph is not None and isinstance(network, igraph.Graph):
        return "igraph"
    if networkx is not None:
        if isinstance(network, networkx.Graph):
            return "networkx"
        if isinstance(network, networkx.DiGraph):
            return "networkx"
        if isinstance(network, networkx.MultiGraph):
            return "networkx"
        if isinstance(network, networkx.MultiDiGraph):
            return "networkx"
    raise TypeError("Unsupported graph type. Supported types are igraph and networkx.")


def detect_directedness(
    network: GraphType,
) -> bool:
    """Detect if the network is directed or not."""
    if network_library(network) == "igraph":
        return network.is_directed()
    if isinstance(network, (networkx.DiGraph, networkx.MultiDiGraph)):
        return True
    return False


def normalise_layout(layout, vertex_ids=None):
    """Normalise the layout to a 2D numpy array."""
    if layout is None:
        return None
    if isinstance(layout, dict):
        if vertex_ids is None:
            raise ValueError("vertex_ids must be provided when layout is a dict.")
        return np.asarray([layout[vid] for vid in vertex_ids])
    if isinstance(layout, str):
        raise NotImplementedError("Layout as a string is not supported yet.")
    if isinstance(layout, (list, tuple)):
        return np.array(layout)
    if isinstance(layout, pd.DataFrame):
        return layout.values
    if isinstance(layout, np.ndarray):
        return layout
    raise TypeError(
        "Layout must be a string, list, tuple, numpy array or pandas DataFrame."
    )
