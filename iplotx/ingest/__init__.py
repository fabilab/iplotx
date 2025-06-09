"""
This module focuses on how to ingest network data into standard data structures no matter what library they come from.
"""

from importlib.metadata import entry_points
import warnings
from typing import (
    Optional,
    Sequence,
    Hashable,
)
import numpy as np
import pandas as pd

from ..importing import igraph, networkx
from .heuristics import (
    network_library,
)
from ..typing import (
    GraphType,
    LayoutType,
)
from .common import (
    NetworkDataProvider,
    NetworkData,
    TreeDataProvider,
)


network_data_providers: dict[NetworkDataProvider] = {}
tree_data_providers: dict[TreeDataProvider] = {}
if networkx is not None:
    from .providers.networkx import NetworkXDataProvider

    network_data_providers["networkx"] = NetworkXDataProvider()
if igraph is not None:
    from .providers.igraph import IGraphDataProvider

    network_data_providers["igraph"] = IGraphDataProvider()

try:
    from .providers.biopython import BiopythonDataProvider

    tree_data_providers["biopython"] = BiopythonDataProvider
except ImportError:
    pass


def ingest_network_data(
    network: GraphType,
    layout: Optional[LayoutType] = None,
    vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
    edge_labels: Optional[Sequence[str] | dict[str]] = None,
) -> NetworkData:
    """Create internal data for the network."""
    _update_data_providers()

    nl = network_library(network, data_providers=network_data_providers)

    if nl in network_data_providers:
        provider: NetworkDataProvider = network_data_providers[nl]
    else:
        sup = ", ".join(network_data_providers.keys())
        raise ValueError(
            f"Network library '{nl}' is not installed. "
            "Currently installed supported libraries: {sup}."
        )

    result = provider(
        network=network,
        layout=layout,
        vertex_labels=vertex_labels,
        edge_labels=edge_labels,
    )
    result["network_library"] = nl
    return result


def ingest_tree_data(
    tree: TreeType,
    layout: Optional[str] = None,
) -> TreeData:
    """Create internal data for the tree."""
    _update_data_providers()

    tl = tree_library(tree, data_providers=tree_data_providers)

    if tl in tree_data_providers:
        provider: TreeDataProvider = tree_data_providers[tl]
    else:
        sup = ", ".join(tree_data_providers.keys())
        raise ValueError(
            f"Tree library '{tl}' is not installed. "
            "Currently installed supported libraries: {sup}."
        )

    result = provider(
        tree=tree,
        layout=layout,
    )
    result["tree_library"] = tl
    return result


def _update_data_providers():
    """Update data provieders dynamically from external packages."""
    global network_data_providers
    global tree_data_providers

    discovered_providers = entry_points(group="iplotx.network_data_providers")
    for entry_point in discovered_providers:
        if entry_point.name not in network_data_providers:
            try:
                provider: NetworkDataProvider = entry_point.load()
                network_data_providers[entry_point.name] = provider
            except Exception as e:
                warnings.warn(
                    f"Failed to load network data provider '{entry_point.name}': {e}"
                )

    discovered_providers = entry_points(group="iplotx.tree_data_providers")
    for entry_point in discovered_providers:
        if entry_point.name not in tree_data_providers:
            try:
                provider: TreeDataProvider = entry_point.load()
                tree_data_providers[entry_point.name] = provider
            except Exception as e:
                warnings.warn(
                    f"Failed to load tree data provider '{entry_point.name}': {e}"
                )
