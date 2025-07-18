from typing import (
    Optional,
    Sequence,
)
from collections.abc import Hashable
import importlib
import numpy as np
import pandas as pd

from ....typing import (
    LayoutType,
)
from ...heuristics import (
    normalise_layout,
)
from ...typing import (
    NetworkDataProvider,
    NetworkData,
)
from ....utils.internal import (
    _make_layout_columns,
)


class NetworkXDataProvider(NetworkDataProvider):
    def __call__(
        self,
        layout: Optional[LayoutType] = None,
        vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
        edge_labels: Optional[Sequence[str] | dict[str]] = None,
    ) -> NetworkData:
        """Create network data object for iplotx from a networkx object."""

        import networkx as nx

        network = self.network

        directed = self.is_directed()

        # Recast vertex_labels=False as vertex_labels=None
        if np.isscalar(vertex_labels) and (not vertex_labels):
            vertex_labels = None

        # Vertices are indexed by node ID
        vertex_df = normalise_layout(
            layout,
            network=network,
            nvertices=self.number_of_vertices(),
        ).loc[pd.Index(network.nodes)]
        ndim = vertex_df.shape[1]
        vertex_df.columns = _make_layout_columns(ndim)

        # Vertex internal properties
        tmp = pd.DataFrame(dict(network.nodes.data())).T
        # Arrays become a single column, which we have already anyway
        if isinstance(layout, str) and (layout in tmp.columns):
            del tmp[layout]
        for col in tmp.columns:
            vertex_df[col] = tmp[col]
        del tmp

        # Vertex labels
        if vertex_labels is None:
            if "label" in vertex_df:
                del vertex_df["label"]
        else:
            if np.isscalar(vertex_labels) and (not vertex_labels) and ("label" in vertex_df):
                del vertex_df["label"]
            elif vertex_labels is True:
                if "label" not in vertex_df:
                    vertex_df["label"] = vertex_df.index
            elif (not np.isscalar(vertex_labels)) and (len(vertex_labels) != len(vertex_df)):
                raise ValueError("Vertex labels must be the same length as the number of vertices.")
            elif isinstance(vertex_labels, nx.classes.reportviews.NodeDataView):
                vertex_df["label"] = pd.Series(dict(vertex_labels))
            else:
                vertex_df["label"] = vertex_labels

        # Edges are a list of tuples, because of multiedges
        tmp = []
        for u, v, d in network.edges.data():
            row = {"_ipx_source": u, "_ipx_target": v}
            row.update(d)
            tmp.append(row)
        if len(tmp):
            edge_df = pd.DataFrame(tmp)
        else:
            edge_df = pd.DataFrame(columns=["_ipx_source", "_ipx_target"])
        del tmp

        # Edge labels
        # Even though they could exist in the dataframe, request the to be explicitely mentioned
        if (edge_labels is None) and ("label" in edge_df):
            del edge_df["label"]
        elif edge_labels is not None:
            if np.isscalar(edge_labels):
                if (not edge_labels) and ("label" in edge_df):
                    del edge_df["label"]
                if (edge_labels is True) and ("label" not in edge_df):
                    edge_df["label"] = [str(i) for i in edge_df.index]
            else:
                if len(edge_labels) != len(edge_df):
                    raise ValueError("Edge labels must be the same length as the number of edges.")
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
        return importlib.util.find_spec("networkx") is not None

    @staticmethod
    def graph_type():
        from networkx import Graph

        return Graph

    def is_directed(self):
        import networkx as nx

        return isinstance(self.network, (nx.DiGraph, nx.MultiDiGraph))

    def number_of_vertices(self):
        """The number of vertices/nodes in the network."""
        return self.network.number_of_nodes()
