from io import StringIO
import os
import unittest
import pytest
import numpy as np
import matplotlib as mpl

try:
    from Bio import Phylo
except ImportError:
    raise unittest.SkipTest("biopython not found, skipping tests")

mpl.use("agg")
import matplotlib.pyplot as plt

import iplotx as ipx

from utils import image_comparison


class TreeTestRunner(unittest.TestCase):

    @property
    def small_tree(self):
        tree = next(
            Phylo.NewickIO.parse(
                StringIO(
                    "(()(()((()())(()()))))",
                )
            )
        )
        return tree

    @image_comparison(baseline_images=["tree_basic"], remove_text=True)
    def test_basic(self):
        tree = self.small_tree
        fig, ax = plt.subplots(figsize=(3, 3))
        ipx.plotting.tree(
            tree=tree,
            ax=ax,
            layout="horizontal",
        )

    @image_comparison(baseline_images=["tree_radial"], remove_text=True)
    def test_radial(self):
        tree = self.small_tree
        fig, ax = plt.subplots(figsize=(3, 3))
        ipx.plotting.tree(
            tree=tree,
            ax=ax,
            layout="radial",
            aspect=1,
        )

    @image_comparison(baseline_images=["leaf_labels"], remove_text=True)
    def test_leaf_labels(self):
        tree = self.small_tree
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
    def test_leaf_labels_hmargin(self):
        tree = self.small_tree
        leaf_labels = {leaf: str(i + 1) for i, leaf in enumerate(tree.get_terminals())}
        vertex_label_hmargin = {
            key: [10, 22][(int(x) - 1) % 2] for key, x in leaf_labels.items()
        }

        fig, ax = plt.subplots(figsize=(4, 4))
        ipx.plotting.tree(
            tree=tree,
            ax=ax,
            layout="horizontal",
            vertex_labels=leaf_labels,
            vertex_label_hmargin=vertex_label_hmargin,
            margins=(0.15, 0),
        )
