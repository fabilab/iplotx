"""
Typing module for data/object ingestion. This module described the abstract data types that providers need to comply with to be compatible with iplotx.

Networkx and trees are treated separately for practical reasons: many tree analysis libraries rely heavily on recursive data structures, which do not
work as well on general networks.
"""

from typing import (
    NotRequired,
    TypedDict,
    Protocol,
    Optional,
    Sequence,
)
from collections.abc import Hashable
import pandas as pd
from ..typing import (
    GraphType,
    LayoutType,
    TreeType,
)


class NetworkData(TypedDict):
    """Network data structure for iplotx."""

    directed: bool
    vertex_df: pd.DataFrame
    edge_df: pd.DataFrame
    ndim: int
    network_library: NotRequired[str]


class NetworkDataProvider(Protocol):
    """Protocol for network data ingestion provider for iplotx."""

    def __call__(
        self,
        network: GraphType,
        layout: Optional[LayoutType] = None,
        vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
        edge_labels: Optional[Sequence[str] | dict] = None,
    ) -> NetworkData:
        """Create network data object for iplotx from any provider."""
        raise NotImplementedError("Network data providers must implement this method.")

    def check_dependencies(
        self,
    ):
        """Check whether the dependencies for this provider are installed."""
        raise NotImplementedError("Network data providers must implement this method.")

    def graph_type(
        self,
    ):
        """Return the graph type from this provider to check for instances."""
        raise NotImplementedError("Network data providers must implement this method.")


class TreeData(TypedDict):
    """Tree data structure for iplotx."""

    rooted: bool
    directed: bool | str
    root: Optional[Hashable]
    leaves: list[Hashable]
    vertex_df: dict[Hashable, tuple[float, float]]
    edge_df: dict[Hashable, Sequence[tuple[float, float]]]
    layout_coordinate_system: str
    layout_name: str
    ndim: int
    tree_library: NotRequired[str]


class TreeDataProvider(Protocol):
    """Protocol for tree data ingestion provider for iplotx."""

    def __call__(
        self,
        tree: TreeType,
        layout: str | LayoutType,
        orientation: Optional[str] = None,
        directed: bool | str = False,
        vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
        edge_labels: Optional[Sequence[str] | dict] = None,
    ) -> TreeData:
        """Create tree data object for iplotx from any provider."""
        raise NotImplementedError("Tree data providers must implement this method.")

    def check_dependencies(
        self,
    ):
        """Check whether the dependencies for this provider are installed."""
        raise NotImplementedError("Tree data providers must implement this method.")

    def tree_type(
        self,
    ):
        """Return the tree type from this provider to check for instances."""
        raise NotImplementedError("Tree data providers must implement this method.")
