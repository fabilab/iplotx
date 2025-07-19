import pytest
import unittest
import importlib
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import iplotx as ipx
from utils import image_comparison

if importlib.util.find_spec("ete4") is None:
    raise unittest.SkipTest("ete4 not found, skipping tests")


@pytest.fixture
def small_tree():
    from ete4 import Tree

    tree = Tree(
        "((),((),(((),()),((),()))));",
    )

    return tree


@image_comparison(baseline_images=["tree_basic"], remove_text=True)
def test_basic(small_tree):
    fig, ax = plt.subplots(figsize=(3, 3))
    ipx.plotting.tree(
        tree=small_tree,
        ax=ax,
        layout="horizontal",
    )


@image_comparison(baseline_images=["tree_radial"], remove_text=True)
def test_radial(small_tree):
    fig, ax = plt.subplots(figsize=(3, 3))
    ipx.plotting.tree(
        tree=small_tree,
        ax=ax,
        layout="radial",
        aspect=1,
    )


@image_comparison(baseline_images=["leaf_labels"], remove_text=True)
def test_leaf_labels(small_tree):
    leaf_labels = {leaf: str(i + 1) for i, leaf in enumerate(small_tree.leaves())}

    fig, ax = plt.subplots(figsize=(4, 4))
    ipx.plotting.tree(
        tree=small_tree,
        ax=ax,
        layout="horizontal",
        vertex_labels=leaf_labels,
        margins=0.1,
    )


@image_comparison(baseline_images=["leaf_labels_hmargin"], remove_text=True)
def test_leaf_labels_hmargin(small_tree):
    leaf_labels = {leaf: str(i + 1) for i, leaf in enumerate(small_tree.leaves())}

    fig, ax = plt.subplots(figsize=(4, 4))
    ipx.plotting.tree(
        tree=small_tree,
        ax=ax,
        layout="horizontal",
        vertex_labels=leaf_labels,
        vertex_label_hmargin=[10, 22],
        margins=(0.15, 0),
    )


@image_comparison(baseline_images=["split_edges"], remove_text=True)
def test_split_edges(small_tree):
    fig, ax = plt.subplots(figsize=(4, 4))
    ipx.plotting.tree(
        tree=small_tree,
        ax=ax,
        layout="horizontal",
        edge_split_linestyle=":",
    )
