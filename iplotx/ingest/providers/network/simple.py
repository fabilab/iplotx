from typing import (
    Optional,
    Sequence,
)
from collections.abc import Hashable
import numpy as np
import pandas as pd

from ....typing import (
    LayoutType,
)
from ...typing import (
    NetworkDataProvider,
    NetworkData,
)
from ....utils.internal import (
    _make_layout_columns,
)


class SimpleNetworkDataProvider(NetworkDataProvider):
    def __call__(
        self,
        layout: Optional[LayoutType] = None,
        vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
        edge_labels: Optional[Sequence[str] | dict[str]] = None,
    ) -> NetworkData:
        """Create network data object for iplotx from a simple Python object."""
        network = self.network
        directed = self.is_directed()

        # Recast vertex_labels=False as vertex_labels=None
        if np.isscalar(vertex_labels) and (not vertex_labels):
            vertex_labels = None

        # Vertices are ordered integers, no gaps
        for key in ["nodes", "vertices"]:
            if key in network:
                vertices = network[key]
                break
        else:
            # Infer from edge adjacent vertices, singletons will be missed
            vertices = set()
            for edge in self.network.get("edges", []):
                vertices.add(edge[0])
                vertices.add(edge[1])
        vertices = list(vertices)

        # NOTE: This is underpowered, but it's ok for a simple educational provider
        if isinstance(layout, pd.DataFrame):
            vertex_df = layout.loc[vertices].copy()
        elif isinstance(layout, dict):
            vertex_df = pd.DataFrame(layout).T.loc[vertices]
        else:
            vertex_df = pd.DataFrame(
                index=vertices,
                data=layout,
            )
        ndim = vertex_df.shape[1]
        vertex_df.columns = _make_layout_columns(ndim)

        # Vertex labels
        if vertex_labels is not None:
            if np.isscalar(vertex_labels):
                vertex_df["label"] = vertex_df.index.astype(str)
            elif len(vertex_labels) != len(vertex_df):
                raise ValueError("Vertex labels must be the same length as the number of vertices.")
            else:
                vertex_df["label"] = vertex_labels

        # Edges are a list of tuples, because of multiedges
        tmp = []
        for edge in network.get("edges", []):
            row = {"_ipx_source": edge[0], "_ipx_target": edge[1]}
            tmp.append(row)
        if len(tmp):
            edge_df = pd.DataFrame(tmp)
        else:
            edge_df = pd.DataFrame(columns=["_ipx_source", "_ipx_target"])
        del tmp

        # Edge labels
        if edge_labels is not None:
            edge_df["label"] = edge_labels

        network_data = {
            "vertex_df": vertex_df,
            "edge_df": edge_df,
            "directed": directed,
            "ndim": ndim,
        }
        return network_data

    @staticmethod
    def check_dependencies() -> bool:
        """Check dependencies. Returns True since this provider has no dependencies."""
        return True

    @staticmethod
    def graph_type():
        return dict

    def is_directed(self):
        """Whether the network is directed."""
        return self.network.get("directed", False)

    def number_of_vertices(self):
        """The number of vertices/nodes in the network."""
        for key in ("nodes", "vertices"):
            if key in self.network:
                return len(self.network[key])

        # Default to unique edge adjacent nodes (this will ignore singletons)
        nodes = set()
        for edge in self.network.get("edges", []):
            nodes.add(edge[0])
            nodes.add(edge[1])
        return len(nodes)
