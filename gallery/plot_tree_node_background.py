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
    tree.get_terminals()[1]: "purple",
}

ipx.plotting.tree(
    tree,
    vertex_cascading_facecolor=backgrounds,
)

# %%
# Cascading patches have a style option "extend" which affects whether the patches extend to the end of the deepest leaf:

ipx.plotting.tree(
    tree,
    vertex_cascading_facecolor=backgrounds,
    vertex_cascading_extend=True,
)

# %%
# Cascading patches work with radial layouts as well:

# sphinx_gallery_thumbnail_number = 3
ipx.plotting.tree(
    tree,
    layout="radial",
    vertex_cascading_facecolor=backgrounds,
    vertex_cascading_extend=True,
    aspect=1,
)
