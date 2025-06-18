"""
Tree with cascading background
==============================

This example shows how to use `iplotx` to add cascading backgrounds to trees.
"Cascading" here means that each patch (rectangle/wedge/etc.) will cover a node
and all descendants, down to the leaves.
"""

from Bio import Phylo
from io import StringIO
import matplotlib.pyplot as plt
import iplotx as ipx

# Make a tree from a string in Newick format
tree = next(
    Phylo.NewickIO.parse(
        StringIO(
            "(()(()((()())(()()))))",
        )
    )
)

backgrounds = {
    tree.get_nonterminals()[3]: "turquoise",
    tree.get_terminals()[0]: "tomato",
}

ipx.plotting.tree(
    tree,
    vertex_cascading_facecolor=backgrounds,
)
