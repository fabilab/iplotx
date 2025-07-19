"""
Subtrees and cascades
=====================

This Biopython-inspired example shows how to style clade edges for a
node and its descendants.

"""

from Bio import Phylo
from io import StringIO
import iplotx as ipx

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
)
art.style_subtree(
    [mrca],
    {"edge": {"color": "purple"}},
)
art.style_subtree(
    [tree.clade[0, 1]],
    {"edge": {"color": "navy"}},
)
