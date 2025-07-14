"""
Typing hints for iplotx.

Some of these types are legit, others are just Any but renamed to improve readability throughout
the codebase.
"""

from typing import (
    Union,
    Sequence,
    Any,
    TypeVar,
)
from collections.abc import Hashable
import numpy as np
import pandas as pd


# NOTE: GraphType is supposed to indicate any kind of graph object that is accepted by
# iplotx's functions, e.g. igraph.Graph or networkx.Graph and subclasses. It is not
# quite possible to really statically type it because providers can add their own
# types - together with protocols to process them - at runtime.
# Nonetheless, for increased readibility we define separately-named types in this
# module to be used throughout the codebase.
GraphType = Any
TreeType = Any

# NOTE: The commented ones are not a mistake: they are supported but cannot be
# statically typed if the user has no igraph installed (it's a soft dependency).
LayoutType = Union[
    str,
    Sequence[Sequence[float]],
    np.ndarray,
    pd.DataFrame,
    dict[Hashable, Sequence[float] | tuple[float, float]],
    # igraph.Layout,
]
GroupingType = Union[
    Sequence[set],
    Sequence[int],
    Sequence[str],
    dict[str, set],
    # igraph.clustering.Clustering,
    # igraph.clustering.VertexClustering,
    # igraph.clustering.Cover,
    # igraph.clustering.VertexCover,
]


T = TypeVar("T")
LeafProperty = Union[
    T,
    Sequence[T],
    dict[Hashable, T],
]

Pair = tuple[T, T]
