from typing import (
    NotRequired,
    TypedDict,
    Protocol,
    Optional,
    Sequence,
    Hashable,
)
import pandas as pd
from ..typing import (
    GraphType,
    LayoutType,
    TreeType,
)


def _make_layout_columns(ndim):
    columns = [f"_ipx_layout_{i}" for i in range(ndim)]
    return columns


class NetworkData(TypedDict):
    """Network data structure for iplotx."""

    directed: bool
    vertex_df: pd.DataFrame
    edge_df: pd.DataFrame
    ndim: int
    network_library: NotRequired[str]


class NetworkDataProvider(Protocol):
    def __call__(
        self,
        network: GraphType,
        layout: Optional[LayoutType] = None,
        vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
        edge_labels: Optional[Sequence[str] | dict] = None,
    ) -> NetworkData:
        """Create network data object for iplotx from any provider."""
        pass

    def check_dependencies(
        self,
    ):
        """Check whether the dependencies for this provider are installed."""
        pass


class TreeData(TypedDict):
    """Tree data structure for iplotx."""

    rooted: bool
    directed: bool | str
    root: Optional[Hashable]
    leaves: list[Hashable]
    vertex_df: dict[Hashable, tuple[float, float]]
    edge_df: dict[Hashable, Sequence[tuple[float, float]]]
    ndim: int
    tree_library: NotRequired[str]


class TreeDataProvider(Protocol):
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
        pass

    def check_dependencies(
        self,
    ):
        """Check whether the dependencies for this provider are installed."""
        pass
