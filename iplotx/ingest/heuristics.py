"""
Heuristics module to funnel certain variable inputs (e.g. layouts) into a standard format.
"""

from typing import (
    Any,
)
from collections.abc import Hashable
from collections import defaultdict
import numpy as np
import pandas as pd

from ..layout import compute_tree_layout
from ..typing import (
    GroupingType,
    LayoutType,
)


# TODO: some of this logic should be moved into individual providers
def normalise_layout(layout, network=None, nvertices=None):
    """Normalise the layout to a pandas.DataFrame."""
    from . import network_library

    try:
        import igraph as ig
    except ImportError:
        ig = None

    if layout is None:
        if (network is not None) and (nvertices == 0):
            return pd.DataFrame(np.zeros((0, 2)))
        return None
    if (network is not None) and isinstance(layout, str):
        if network_library(network) == "igraph":
            if hasattr(network, layout):
                layout = network[layout]
            else:
                layout = network.layout(layout)
                # NOTE: This seems like a legit bug in igraph
                # Sometimes (e.g. sugiyama) the layout has more vertices than the network (?)
                layout = np.asarray(layout.coords)[: network.vcount()]
        if network_library(network) == "networkx":
            layout = dict(network.nodes.data(layout))

    if (ig is not None) and isinstance(layout, ig.layout.Layout):
        return pd.DataFrame(layout.coords)
    if isinstance(layout, dict):
        return pd.DataFrame(layout).T
    if isinstance(layout, str):
        raise NotImplementedError("Layout as a string is not supported yet.")
    if isinstance(layout, (list, tuple)):
        return pd.DataFrame(np.array(layout))
    if isinstance(layout, pd.DataFrame):
        return layout
    if isinstance(layout, np.ndarray):
        return pd.DataFrame(layout)
    raise TypeError("Layout could not be normalised.")


def normalise_tree_layout(
    layout: str | Any,
    **kwargs,
) -> pd.DataFrame:
    """Normalise tree layout from a variety of inputs.

    Parameters:
        layout: The tree layout to normalise.
        tree: The correcponding tree object.
        **kwargs: Additional arguments for the subroutines.

    Returns:
        A pandas DataFrame with the normalised tree layout.

    NOTE: This function currently only accepts strings and computes
        the layout internally. This might change in the future.
    """
    if isinstance(layout, str):
        layout = compute_tree_layout(layout, **kwargs)
    else:
        raise NotImplementedError("Only internally computed tree layout currently accepted.")

    if isinstance(layout, dict):
        # Adjust vertex layout
        index = []
        coordinates = []
        for key, coordinate in layout.items():
            index.append(key)
            coordinates.append(coordinate)
        index = pd.Index(index)
        coordinates = np.array(coordinates)
        ndim = len(coordinates[0]) if len(coordinates) > 0 else 2
        layout_columns = [f"_ipx_layout_{i}" for i in range(ndim)]
        layout = pd.DataFrame(
            coordinates,
            index=index,
            columns=layout_columns,
        )

    return layout


def normalise_grouping(
    grouping: GroupingType,
    layout: LayoutType,
) -> dict[Hashable, set]:
    """Normalise network grouping from a variery of inputs.

    Parameters:
        grouping: Network grouping (e.g. vertex cover).
        layout: Network layout.

    Returns:
        A dictionary of sets. Each key is the index of a group, each value is a set of vertices
        included in that group. If all sets are mutually exclusive, this is a vertex clustering,
        otherwise it's only a vertex cover.
    """
    try:
        import igraph as ig
    except ImportError:
        ig = None

    if len(grouping) == 0:
        return {}

    if isinstance(grouping, dict):
        val0 = next(iter(grouping.values()))
        # If already the right data type or compatible, leave as is
        if isinstance(val0, (set, frozenset)):
            return grouping

        # If a dict of integers or strings, assume each key is a vertex id and each value is a
        # group, convert (i.e. invert the dict)
        if isinstance(val0, (int, str)):
            group_dic = defaultdict(set)
            for key, val in grouping.items():
                group_dic[val].add(key)
            return group_dic

    # If an igraph object, convert to a dict of sets
    if ig is not None:
        if isinstance(grouping, ig.clustering.Clustering):
            layout = normalise_layout(layout)
            group_dic = defaultdict(set)
            for i, member in enumerate(grouping.membership):
                group_dic[member].add(i)
            return group_dic

        if isinstance(grouping, ig.clustering.Cover):
            layout = normalise_layout(layout)
            group_dic = defaultdict(set)
            for i, members in enumerate(grouping.membership):
                for member in members:
                    group_dic[member].add(i)
            return group_dic

    # Assume it's a sequence, so convert to list
    grouping = list(grouping)

    # If the values are already sets, assume group indices are integers
    # and values are as is
    if isinstance(grouping[0], set):
        return dict(enumerate(grouping))

    # If the values are integers or strings, assume each key is a vertex id and each value is a
    # group, convert to dict of sets
    if isinstance(grouping[0], (int, str)):
        group_dic = defaultdict(set)
        for i, val in enumerate(grouping):
            group_dic[val].add(i)
        return group_dic

    raise TypeError(
        "Could not standardise grouping from object.",
    )
