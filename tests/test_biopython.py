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
from iplotx.importing import networkx as nx


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
        plt.close("all")

        tree = self.small_tree
        fig, ax = plt.subplots(figsize=(3, 3))
        ipx.plotting.tree(
            tree=tree,
            ax=ax,
            layout="horizontal",
        )
