from io import StringIO
import importlib
import unittest
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import iplotx as ipx

from utils import image_comparison

if importlib.util.find_spec("skbio") is None:
    raise unittest.SkipTest("skbio not found, skipping tests")


class TreeTestRunner(unittest.TestCase):
    @property
    def small_tree(self):
        from skbio import TreeNode

        tree = TreeNode.read(
            StringIO(
                "((),((),(((),()),((),()))));",
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
        leaf_labels = {leaf: str(i + 1) for i, leaf in enumerate(tree.tips())}

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
        leaf_labels = {leaf: str(i + 1) for i, leaf in enumerate(tree.tips())}

        fig, ax = plt.subplots(figsize=(4, 4))
        ipx.plotting.tree(
            tree=tree,
            ax=ax,
            layout="horizontal",
            vertex_labels=leaf_labels,
            vertex_label_hmargin=[10, 22],
            margins=(0.15, 0),
        )
