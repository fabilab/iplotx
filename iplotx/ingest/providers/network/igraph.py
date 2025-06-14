from typing import (
    Optional,
    Sequence,
)
from collections.abc import Hashable
import numpy as np
import pandas as pd

from ....typing import (
    GraphType,
    LayoutType,
)
from ...heuristics import (
    normalise_layout,
    detect_directedness,
)
from ...typing import (
    NetworkDataProvider,
    NetworkData,
)
from ....utils.internal import (
    _make_layout_columns,
)


class IGraphDataProvider(NetworkDataProvider):
    def __call__(
        self,
        network: GraphType,
        layout: Optional[LayoutType] = None,
        vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
        edge_labels: Optional[Sequence[str] | dict[str]] = None,
    ) -> NetworkData:
        """Create network data object for iplotx from any provider."""

        directed = detect_directedness(network)

        # Recast vertex_labels=False as vertex_labels=None
        if np.isscalar(vertex_labels) and (not vertex_labels):
            vertex_labels = None

        # Vertices are ordered integers, no gaps
        vertex_df = normalise_layout(layout, network=network)
        ndim = vertex_df.shape[1]
        vertex_df.columns = _make_layout_columns(ndim)

        # Vertex labels
        if vertex_labels is not None:
            if np.isscalar(vertex_labels):
                vertex_df["label"] = vertex_df.index.astype(str)
            elif len(vertex_labels) != len(vertex_df):
                raise ValueError(
                    "Vertex labels must be the same length as the number of vertices."
                )
            else:
                vertex_df["label"] = vertex_labels

        # Edges are a list of tuples, because of multiedges
        tmp = []
        for edge in network.es:
            row = {"_ipx_source": edge.source, "_ipx_target": edge.target}
            row.update(edge.attributes())
            tmp.append(row)
        if len(tmp):
            edge_df = pd.DataFrame(tmp)
        else:
            edge_df = pd.DataFrame(columns=["_ipx_source", "_ipx_target"])
        del tmp

        # Edge labels
        if edge_labels is not None:
            if len(edge_labels) != len(edge_df):
                raise ValueError(
                    "Edge labels must be the same length as the number of edges."
                )
            edge_df["label"] = edge_labels

        network_data = {
            "vertex_df": vertex_df,
            "edge_df": edge_df,
            "directed": directed,
            "ndim": ndim,
        }
        return network_data

    def check_dependencies(self) -> bool:
        try:
            import igraph
        except ImportError:
            return False
        return True

    def graph_type(self):
        import igraph as ig

        return ig.Graph
