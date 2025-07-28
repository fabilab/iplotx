import importlib
import unittest
import pytest
import numpy as np
import pandas as pd
from iplotx.ingest import heuristics

if importlib.util.find_spec("igraph") is None:
    raise unittest.SkipTest("igraph not found, skipping tests")
else:
    import igraph as ig
if importlib.util.find_spec("networkx") is None:
    raise unittest.SkipTest("networkx not found, skipping tests")
else:
    import networkx as nx


def test_empty_layout():
    res = heuristics.normalise_layout(None, network={}, nvertices=0)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (0, 2)

    res = heuristics.normalise_layout(None, network=None, nvertices=0)
    assert res is None


def test_array_layout():
    res = heuristics.normalise_layout(np.zeros((3, 2)), network={}, nvertices=3)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (3, 2)


def test_igraph_layouts():
    g = ig.Graph.Ring(5)
    g["circle"] = g.layout_circle()
    res = heuristics.normalise_layout("circle", network=g)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (5, 2)

    g = ig.Graph.Ring(5)
    res = heuristics.normalise_layout("circle", network=g)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (5, 2)

    g = ig.Graph.Ring(5)
    layout = g.layout_circle()
    res = heuristics.normalise_layout(layout, network=g)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (5, 2)


def test_networkx_layouts():
    G = nx.house_graph()
    nx.set_node_attributes(G, {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0.5, 2.0)}, "pos")
    res = heuristics.normalise_layout("pos", network=G)
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (5, 2)


accepted_groupings = [
    [{0, 1}, {0, 2}, {2, 4}],
    [0, 1, 0, 2, 2],
    {0: {0, 1}, 1: {0, 2}},
    {0: 0, 1: 1, 2: 0, 3: 0},
    ig.clustering.Clustering([0, 0, 1, 2, 2]),
    ig.clustering.Cover([[0, 1], [0, 1, 2], [2, 4]]),
]


@pytest.mark.parametrize("grouping", accepted_groupings)
def test_grouping_dict(grouping):
    g = ig.Graph.Ring(5)
    layout = g.layout_circle()
    res = heuristics.normalise_grouping(
        grouping,
        layout,
    )
    assert isinstance(res, dict)
    for k, v in res.items():
        assert isinstance(k, int)
        assert isinstance(v, (set, frozenset))


def test_grouping_sequence_invalid():
    g = ig.Graph.Ring(5)
    layout = g.layout_circle()
    grouping = [[0], 1, 0, 2, 2]
    with pytest.raises(TypeError):
        heuristics.normalise_grouping(
            grouping,
            layout,
        )
