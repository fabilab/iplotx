import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iplotx as ipx


@pytest.fixture
def tree():
    tree = {
        "children": [
            {
                "children": [
                    {},
                    {},
                ],
            },
            {
                "children": [
                    {
                        "children": [
                            {"branch_length": 1.5},
                            {},
                        ],
                    },
                    {
                        "children": [
                            {},
                            {},
                        ],
                    },
                ],
            },
        ],
    }
    tree = ipx.ingest.providers.tree.simple.SimpleTree.from_dict(tree)
    return tree


def test_unsupported_layout(tree):
    with pytest.raises(ValueError):
        ipx.artists.TreeArtist(
            tree,
            layout="unsupported",
        )


def test_get_layout(tree):
    art = ipx.artists.TreeArtist(
        tree,
    )
    assert isinstance(art.get_layout(), pd.DataFrame)


def test_style_subtree(tree):
    art = ipx.artists.TreeArtist(
        tree,
    )
    art.style_subtree(
        [tree.children[0]],
        style={
            "vertex": {"size": 100},
            "edge": {"color": "red"},
        },
    )


def test_style_subtree_sequence(tree):
    art = ipx.artists.TreeArtist(
        tree,
    )
    art.style_subtree(
        [tree.children[0]],
        style=[
            {
                "vertex": {"size": 100},
                "edge": {"color": "red"},
            }
        ],
    )


def test_draw_invisible(tree):
    art = ipx.artists.TreeArtist(
        tree,
    )
    art.set_visible(False)
    fig = plt.figure()
    try:
        assert art.draw(fig.canvas.get_renderer()) is None
    finally:
        plt.close(fig)


def test_cascade_maxdepth(tree):
    for layout, orientation in [
        ("horizontal", "right"),
        ("horizontal", "left"),
        ("vertical", "ascending"),
        ("vertical", "descending"),
        ("radial", "clockwise"),
        ("radial", "counterclockwise"),
    ]:
        with ipx.style.context(
            [
                "tree",
                {
                    "cascade": {"facecolor": {tree.children[0]: "blue"}},
                    "layout": {"orientation": orientation},
                },
            ]
        ):
            ipx.artists.TreeArtist(
                tree,
                layout=layout,
            )


def test_cascade_maxdepth_leaflabels(tree):
    for layout, orientation in [
        ("horizontal", "right"),
        ("horizontal", "left"),
        ("vertical", "ascending"),
        ("vertical", "descending"),
        ("radial", "clockwise"),
        ("radial", "counterclockwise"),
    ]:
        with ipx.style.context(
            [
                "tree",
                {
                    "cascade": {
                        "facecolor": {tree.children[0]: "blue"},
                        "extend": "leaf_labels",
                    },
                    "layout": {"orientation": orientation},
                    "leaf": {"deep": True},
                },
            ]
        ):
            ipx.artists.TreeArtist(
                tree,
                layout=layout,
                leaf_labels=["A", "B", "C", "D", "E", "F"],
            )


def test_cascades_leaflabels_missing(tree):
    with ipx.style.context(
        [
            "tree",
            {
                "cascade": {
                    "facecolor": {tree.children[0]: "blue"},
                    "extend": "leaf_labels",
                },
                "leaf": {"deep": True},
            },
        ]
    ):
        with pytest.raises(ValueError):
            ipx.artists.TreeArtist(
                tree,
            )


def test_edges_angular(tree):
    with ipx.style.context(
        layout_angular=True,
    ):
        ipx.artists.TreeArtist(
            tree,
        )


def test_edges_nowaypoints_bool(tree):
    with ipx.style.context(
        layout_angular=True,
        edge_waypoints=False,
    ):
        ipx.artists.TreeArtist(
            tree,
        )


def test_edges_cmap(tree):
    with ipx.style.context(
        edge_cmap="viridis",
        edge_color=[2, 3, 5],
    ):
        art = ipx.artists.TreeArtist(
            tree,
        )
    # We cannot test the entire thing without plotting
    # because matplotlib delays cmap to rendering time
    # but at least we can check the _A array
    np.testing.assert_array_equal(
        art.get_edges()._A[:6],
        np.ma.MaskedArray(
            [2, 3, 5, 2, 3, 5],
        ),
    )


def test_leafedges_cmap(tree):
    with ipx.style.context(
        leafedge_cmap="viridis",
        leafedge_color=[2, 3, 5],
        leaf_deep=True,
    ):
        art = ipx.artists.TreeArtist(
            tree,
        )
    # We cannot test the entire thing without plotting
    # because matplotlib delays cmap to rendering time
    # but at least we can check the _A array
    np.testing.assert_array_equal(
        art.get_leaf_edges()._A,
        np.ma.MaskedArray(
            [2, 3, 5, 2, 3, 5],
        ),
    )
