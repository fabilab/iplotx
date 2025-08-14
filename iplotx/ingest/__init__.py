"""
This module focuses on how to ingest network/tree data into standard data structures no matter what library they come from.
"""

import pathlib
import pkgutil
import importlib
import warnings
from typing import (
    Optional,
    Sequence,
    Protocol,
)
from collections.abc import Hashable
import pandas as pd

from ..typing import (
    GraphType,
    LayoutType,
    TreeType,
)
from .typing import (
    NetworkDataProvider,
    NetworkData,
    TreeDataProvider,
    TreeData,
)

provider_protocols = {
    "network": NetworkDataProvider,
    "tree": TreeDataProvider,
}

# Internally supported data providers
data_providers: dict[str, dict[str, Protocol]] = {kind: {} for kind in provider_protocols}
for kind in data_providers:
    providers_path = pathlib.Path(__file__).parent.joinpath("providers").joinpath(kind)
    for importer, module_name, _ in pkgutil.iter_modules([providers_path]):
        module = importlib.import_module(f"iplotx.ingest.providers.{kind}.{module_name}")
        for key, val in module.__dict__.items():
            if key == provider_protocols[kind].__name__:
                continue
            if key.endswith("DataProvider"):
                data_providers[kind][module_name] = val
                break
    del providers_path


def network_library(network) -> str:
    """Guess the network library used to create the network."""
    for name, provider in data_providers["network"].items():
        if provider.check_dependencies():
            graph_type = provider.graph_type()
            if isinstance(network, graph_type):
                return name
    raise ValueError(
        f"Network {network} did not match any available network library.",
    )


def tree_library(tree) -> str:
    """Guess the tree library used to create the tree."""
    for name, provider in data_providers["tree"].items():
        if provider.check_dependencies():
            tree_type = provider.tree_type()
            if isinstance(tree, tree_type):
                return name
    raise ValueError(
        f"Tree {tree} did not match any available tree library.",
    )


# Functions to ingest data from various libraries
def ingest_network_data(
    network: GraphType,
    layout: Optional[LayoutType] = None,
    vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
    edge_labels: Optional[Sequence[str] | dict[str,]] = None,
) -> NetworkData:
    """Create internal data for the network."""
    _update_data_providers("network")

    nl = network_library(network)

    if nl in data_providers["network"]:
        provider: NetworkDataProvider = data_providers["network"][nl]
    else:
        sup = ", ".join(data_providers["network"].keys())
        raise ValueError(
            f"Network library '{nl}' is not installed. "
            f"Currently installed supported libraries: {sup}."
        )

    result = provider(network)(
        layout=layout,
        vertex_labels=vertex_labels,
        edge_labels=edge_labels,
    )
    result["network_library"] = nl
    return result


def ingest_tree_data(
    tree: TreeType,
    layout: Optional[str] = "horizontal",
    directed: bool | str = False,
    layout_style: Optional[dict[str, str | int | float]] = None,
    vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
    edge_labels: Optional[Sequence[str] | dict[Hashable, str]] = None,
    leaf_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
) -> TreeData:
    """Create internal data for the tree."""
    _update_data_providers("tree")

    tl = tree_library(tree)

    if tl in data_providers["tree"]:
        provider: TreeDataProvider = data_providers["tree"][tl]
    else:
        sup = ", ".join(data_providers["tree"].keys())
        raise ValueError(
            f"Tree library '{tl}' is not installed. Currently installed supported libraries: {sup}."
        )

    result = provider(
        tree=tree,
    )(
        layout=layout,
        directed=directed,
        layout_style=layout_style,
        vertex_labels=vertex_labels,
        edge_labels=edge_labels,
        leaf_labels=leaf_labels,
    )
    result["tree_library"] = tl

    return result


# INTERNAL FUNCTIONS
def _update_data_providers(kind):
    """Update data provieders dynamically from external packages."""
    discovered_providers = importlib.metadata.entry_points(group=f"iplotx.{kind}_data_providers")
    for entry_point in discovered_providers:
        if entry_point.name not in data_providers["network"]:
            try:
                data_providers[kind][entry_point.name] = entry_point.load()
            except Exception as e:
                warnings.warn(f"Failed to load {kind} data provider '{entry_point.name}': {e}")
