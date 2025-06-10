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
    tree_library,
)
from ..typing import (
    GraphType,
    LayoutType,
    TreeType,
)
from .common import (
    NetworkDataProvider,
    NetworkData,
    TreeDataProvider,
    TreeData,
)
from .providers.networkx import NetworkXDataProvider
from .providers.igraph import IGraphDataProvider
from .providers.biopython import BiopythonDataProvider


# Internally supported data providers
network_data_providers: dict[str, NetworkDataProvider] = {}
tree_data_providers: dict[str, TreeDataProvider] = {}

provider = NetworkXDataProvider()
if provider.check_dependencies():
    network_data_providers["networkx"] = provider

provider = IGraphDataProvider()
if provider.check_dependencies():
    network_data_providers["igraph"] = provider

provider = BiopythonDataProvider()
if provider.check_dependencies():
    tree_data_providers["biopython"] = provider


# Functions to ingest data from various libraries
def ingest_network_data(
    network: GraphType,
    layout: Optional[LayoutType] = None,
    vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
    edge_labels: Optional[Sequence[str] | dict[str,]] = None,
) -> NetworkData:
    """Create internal data for the network."""
    _update_data_providers("network")

    nl = network_library(network, data_providers=network_data_providers)

    if nl in network_data_providers:
        provider: NetworkDataProvider = network_data_providers[nl]
    else:
        sup = ", ".join(network_data_providers.keys())
        raise ValueError(
            f"Network library '{nl}' is not installed. "
            f"Currently installed supported libraries: {sup}."
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
    layout: Optional[str] = "horizontal",
    orientation: Optional[str] = "right",
    directed: bool | str = False,
    vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
    edge_labels: Optional[Sequence[str] | dict[str,]] = None,
) -> TreeData:
    """Create internal data for the tree."""
    _update_data_providers("tree")

    tl = tree_library(tree, data_providers=tree_data_providers)

    if tl in tree_data_providers:
        provider: TreeDataProvider = tree_data_providers[tl]
    else:
        sup = ", ".join(tree_data_providers.keys())
        raise ValueError(
            f"Tree library '{tl}' is not installed. "
            f"Currently installed supported libraries: {sup}."
        )

    result = provider(
        tree=tree,
        layout=layout,
        orientation=orientation,
        directed=directed,
        vertex_labels=vertex_labels,
        edge_labels=edge_labels,
    )
    result["tree_library"] = tl
    return result


# INTERNAL FUNCTIONS
def _update_data_providers(kind):
    """Update data provieders dynamically from external packages."""
    global network_data_providers
    global tree_data_providers
    prov_dict = {
        "network": network_data_providers,
        "tree": tree_data_providers,
    }
    type_dict = {
        "network": NetworkDataProvider,
        "tree": TreeDataProvider,
    }

    discovered_providers = entry_points(group=f"iplotx.{kind}_data_providers")
    for entry_point in discovered_providers:
        if entry_point.name not in network_data_providers:
            try:
                provider: type_dict[kind] = entry_point.load()
                prov_dict[kind][entry_point.name] = provider
            except Exception as e:
                warnings.warn(
                    f"Failed to load {kind} data provider '{entry_point.name}': {e}"
                )
