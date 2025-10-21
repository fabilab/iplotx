import pytest
import matplotlib as mpl

mpl.use("agg")
import iplotx as ipx
from utils import image_comparison


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


@image_comparison(baseline_images=["tree_nogap"], remove_text=True)
def test_nogap(tree):
    ipx.doubletree(
        tree,
        tree,
        kwargs_right=dict(
            edge_color="steelblue",
        ),
        gap=0,
    )


@image_comparison(baseline_images=["tree_gap"], remove_text=True)
def test_gap(tree):
    ipx.doubletree(
        tree,
        tree,
        kwargs_left=dict(
            leaf_deep=True,
        ),
        kwargs_right=dict(
            edge_color="steelblue",
            leaf_deep=True,
        ),
        gap=1.0,
    )
