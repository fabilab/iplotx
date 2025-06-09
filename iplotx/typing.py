from typing import Union, Sequence, Never
from numpy import ndarray
from pandas import DataFrame

from .importing import igraph, networkx


# FIXME: check this mess with static type checkers and see how far it needs to/can go.
igraphGraph = igraph.Graph if igraph is not None else Never
if networkx is not None:
    networkxOmniGraph = Union[
        networkx.Graph,
        networkx.DiGraph,
        networkx.MultiGraph,
        networkx.MultiDiGraph,
    ]
else:
    networkxOmniGraph = Never

if igraphGraph is not None and networkxOmniGraph is not None:
    GraphType = Union[igraphGraph, networkxOmniGraph]
elif igraphGraph is not None:
    GraphType = igraphGraph
else:
    GraphType = networkxOmniGraph

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

LayoutType = Union[str, Sequence[Sequence[float]], ndarray, DataFrame]

TreeType = dict
