from io import StringIO
import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import iplotx as ipx

from utils import image_comparison

Bio = pytest.importorskip("Bio")
from Bio import Phylo  # noqa: E402


@pytest.fixture
def tree():
    tree = next(
        Phylo.NewickIO.parse(
            StringIO(
                "(()(()((()())(()()))))",
            )
        )
    )
    return tree


def test_get_layout(tree):
    art = ipx.artists.TreeArtist(
        tree,
    )
    assert isinstance(art.get_layout(), pd.DataFrame)


def test_get_edge_layout(tree):
    art = ipx.artists.TreeArtist(
        tree,
    )
    with pytest.raises(KeyError):
        art.get_layout(kind="edge")


def test_get_invalid_layout(tree):
    art = ipx.artists.TreeArtist(
        tree,
    )
    with pytest.raises(ValueError):
        art.get_layout(kind="invalid")


def test_get_zero_layout(tree):
    art = ipx.artists.TreeArtist(
        tree,
    )
    art._ipx_internal_data["vertex_df"] = art._ipx_internal_data["vertex_df"][:0]
    bbox = art.get_datalim(None)
    assert isinstance(bbox, mpl.transforms.Bbox)


def test_get_elements(tree):
    art = ipx.artists.TreeArtist(
        tree,
        leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.get_terminals())},
    )
    assert isinstance(art.get_vertices(), ipx.artists.VertexCollection)
    assert isinstance(art.get_edges(), ipx.artists.EdgeCollection)
    assert isinstance(art.get_leaf_vertices(), ipx.artists.VertexCollection)
    assert isinstance(art.get_leaf_edges(), ipx.artists.EdgeCollection)
    assert art.get_leaf_edge_labels() is None

    with ipx.style.context(
        [
            "tree",
            {"leaf": {"deep": False}},
        ]
    ):
        art = ipx.artists.TreeArtist(
            tree,
        )
        assert art.get_leaf_edges() is None
        assert art.get_leaf_edge_labels() is None

    assert art.get_vertex_labels() is None


def test_leaf_labels_array(tree):
    for leaf in tree.get_terminals():
        leaf.name = "hello"
    ipx.artists.TreeArtist(
        tree,
        leaf_labels=True,
    )


@pytest.mark.parametrize(
    "layout,orientation",
    [
        ["horizontal", "right"],
        ["horizontal", "left"],
        ["vertical", "descending"],
        ["vertical", "ascending"],
        ["radial", "clockwise"],
        ["radial", "counterclockwise"],
    ],
)
def test_vertical_leaf_orientations(tree, layout, orientation):
    with ipx.style.context(["tree", {"layout_orientation": orientation}]):
        ipx.artists.TreeArtist(
            tree,
            layout=layout,
            leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.get_terminals())},
        )


def test_leaf_vertices_invalid_orientation(tree):
    with ipx.style.context(["tree", {"layout_orientation": "unsupported"}]):
        with pytest.raises(ValueError):
            ipx.artists.TreeArtist(
                tree,
                layout="vertical",
                leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.get_terminals())},
            )


def test_get_maxdepth_leaf_labels(tree):
    art = ipx.artists.TreeArtist(
        tree,
        leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.get_terminals())},
    )
    maxdepth = art._get_maxdepth_leaf_labels()
    assert isinstance(maxdepth, float)


def test_style_subtree(tree):
    art = ipx.artists.TreeArtist(
        tree,
        leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.get_terminals())},
    )
    art.style_subtree(
        tree.clade[0, 1],
        {"edge": {"color": "red"}},
    )


def test_get_edge_labels(tree):
    art = ipx.artists.TreeArtist(
        tree,
        edge_labels=["A", "B", "C", "D", "E", "F"],
    )
    np.testing.assert_array_equal(
        art.get_edge_labels().get_texts()[:6],
        np.array(["A", "B", "C", "D", "E", "F"]),
    )
    np.testing.assert_array_equal(
        art.get_edge_labels().get_text()[:6],
        np.array(["A", "B", "C", "D", "E", "F"]),
    )


@image_comparison(baseline_images=["tree_basic"], remove_text=True)
def test_basic(tree):
    fig, ax = plt.subplots(figsize=(3, 3))
    ipx.plotting.tree(
        tree=tree,
        ax=ax,
        layout="horizontal",
    )


@image_comparison(baseline_images=["tree_radial"], remove_text=True)
def test_radial(tree):
    fig, ax = plt.subplots(figsize=(3, 3))
    ipx.plotting.tree(
        tree=tree,
        ax=ax,
        layout="radial",
        aspect=1,
    )


@image_comparison(baseline_images=["leaf_labels"], remove_text=True)
def test_leaf_labels(tree):
    leaf_labels = {leaf: str(i + 1) for i, leaf in enumerate(tree.get_terminals())}

    fig, ax = plt.subplots(figsize=(4, 4))
    ipx.plotting.tree(
        tree=tree,
        ax=ax,
        layout="horizontal",
        vertex_labels=leaf_labels,
        margins=0.1,
    )


@image_comparison(baseline_images=["leaf_labels_hmargin"], remove_text=True)
def test_leaf_labels_hmargin(tree):
    leaf_labels = {leaf: str(i + 1) for i, leaf in enumerate(tree.get_terminals())}
    vertex_label_hmargin = {key: [10, 22][(int(x) - 1) % 2] for key, x in leaf_labels.items()}

    fig, ax = plt.subplots(figsize=(4, 4))
    ipx.plotting.tree(
        tree=tree,
        ax=ax,
        layout="horizontal",
        vertex_labels=leaf_labels,
        vertex_label_hmargin=vertex_label_hmargin,
        margins=(0.15, 0),
    )


@image_comparison(baseline_images=["show_support"], remove_text=True)
def test_show_support(tree):
    for node in tree.get_nonterminals():
        if node != tree.root:
            node.confidence = 0.9

    fig, ax = plt.subplots(figsize=(4, 4))
    ipx.plotting.tree(
        tree=tree,
        ax=ax,
        layout="horizontal",
        show_support=True,
    )


@image_comparison(baseline_images=["cascades"], remove_text=True)
def test_cascades(tree):
    backgrounds = {
        tree.get_nonterminals()[3]: "turquoise",
        tree.get_terminals()[0]: "tomato",
        tree.get_terminals()[1]: "purple",
    }

    ipx.plotting.tree(
        tree,
        cascade_facecolor=backgrounds,
        cascade_edgecolor="none",
    )


@image_comparison(baseline_images=["leafedges"], remove_text=True)
def test_leafedges(tree):
    ipx.plotting.tree(
        tree,
        leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.get_terminals())},
    )


@image_comparison(baseline_images=["directed_child"], remove_text=True)
def test_directed_child(tree):
    ipx.plotting.tree(
        tree,
        directed=True,
    )
