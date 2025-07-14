"""
Styling clades
==============

This Biopython-inspired example shows how to style clade edges for a node and its descendants.
"""

from Bio import Phylo
from io import StringIO
import iplotx as ipx
import matplotlib.pyplot as plt
from collections import defaultdict

tree = Phylo.read(
    StringIO("(((A,B),(C,D)),(E,F,G));"),
    format="newick",
)

mrca = tree.common_ancestor({"name": "E"}, {"name": "F"})

art = ipx.tree(
    tree,
    leaf_deep=True,
    leaf_labels=True,
    cascade_facecolor={
        mrca: "salmon",
        tree.clade[0, 1]: "steelblue",
    },
    cascade_extend=True,
    edge_color={
        mrca: "purple",
        tree.clade[0, 1]: "navy",
    },
)
