"""
This module focuses on how to ingest network data into standard data structures no matter what library they come from.
"""

from importlib.metadata import entry_points
import warnings
from typing import (
    Optional,
    Sequence,
    Protocol,
    Hashable,
)
import numpy as np
import pandas as pd

from ..importing import igraph, networkx
from ..network import network_library, detect_directedness
from ..heuristics import normalise_layout
from ..typing import (
    GraphType,
    LayoutType,
)
from .common import (
    NetworkDataProvider,
    NetworkData,
)


data_providers: dict[NetworkDataProvider] = {}
if networkx is not None:
    from .providers.networkx import NetworkXDataProvider

    data_providers["networkx"] = NetworkXDataProvider()
if igraph is not None:
    from .providers.igraph import IGraphDataProvider

    data_providers["igraph"] = IGraphDataProvider()


def ingest_network_data(
    network: GraphType,
    layout: Optional[LayoutType] = None,
    vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
    edge_labels: Optional[Sequence[str] | dict[str]] = None,
) -> NetworkData:
    """Create internal data for the network."""
    _update_data_providers()

    nl = network_library(network, data_providers=data_providers)

    if nl in data_providers:
        provider: NetworkDataProvider = data_providers[nl]
    else:
        sup = ", ".join(data_providers.keys())
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


def _update_data_providers():
    """Update data provieders dynamically from external packages."""
    global data_providers

    discovered_providers = entry_points(group="iplotx.network_data_providers")
    for entry_point in discovered_providers:
        if entry_point.name not in data_providers:
            try:
                provider: NetworkDataProvider = entry_point.load()
                data_providers[entry_point.name] = provider
            except Exception as e:
                warnings.warn(
                    f"Failed to load network data provider '{entry_point.name}': {e}"
                )
