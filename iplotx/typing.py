from typing import Union, Sequence
from numpy import ndarray
from pandas import DataFrame

from .importing import igraph, networkx


igraphGraph = igraph.Graph if igraph is None else None
if networkx is not None:
    from networkx import Graph as networkxGraph
    from networkx import DiGraph as networkxDiGraph
    from networkx import MultiGraph as networkxMultiGraph
    from networkx import MultiDiGraph as networkxMultiDiGraph

    networkxOmniGraph = Union[
        networkxGraph, networkxDiGraph, networkxMultiGraph, networkxMultiDiGraph
    ]
else:
    networkxOmniGraph = None

if igraphGraph is not None and networkxOmniGraph is not None:
    GraphType = Union[igraphGraph, networkxOmniGraph]
elif igraphGraph is not None:
    GraphType = igraphGraph
else:
    GraphType = networkxOmniGraph

LayoutType = Union[str, Sequence[Sequence[float]], ndarray, DataFrame]

if (igraph is not None) and (networkx is not None):
    # networkx returns generators of sets, igraph has its own classes
    # additionally, one can put list of memberships
    GroupingType = Union[
        Sequence[set],
        igraph.clustering.Clustering,
        igraph.clustering.VertexClustering,
        igraph.clustering.Cover,
        igraph.clustering.VertexCover,
        Sequence[int],
        Sequence[str],
    ]
