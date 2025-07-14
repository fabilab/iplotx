"""
Leaf edges and cascades
=======================

This short example demonstrates how to combine leaf edges and cascades.
"""

from Bio import Phylo
from io import StringIO
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
    tree.get_terminals()[1]: "purple",
}

ipx.plotting.tree(
    tree,
    cascade_facecolor=backgrounds,
    cascade_extend=True,
    leaf_labels={leaf: f"L{i + 1}" for i, leaf in enumerate(tree.get_terminals())},
    edge_color="#111",
    leafedge_linewidth=2,
    leafedge_color="steelblue",
    leafedge_linestyle=":",
)
