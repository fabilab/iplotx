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
)


def _make_layout_columns(ndim):
    columns = [f"_ipx_layout_{i}" for i in range(ndim)]
    return columns


class NetworkData(TypedDict):
    """Network data structure for iplotx."""

    vertices: pd.DataFrame
    edges: pd.DataFrame
    directed: bool
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
