import pytest
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
