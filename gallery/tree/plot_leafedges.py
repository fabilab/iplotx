"""
Leaf edges
==========

This example shows how to use leaf edges, dashed lines connecting each leaf to the
deepest leaf and label.
"""

from ete4 import Tree
import iplotx as ipx

tree = Tree(
    "((),((),(((),()),((),()))));",
)

ipx.plotting.tree(
    tree,
    leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.leaves())},
)

# %%
# Leaf edges are used by default when leaf labels are present and deep
# leaves are set (default). If you use shallow leaves, leaf edges are
# not plotted:

ipx.plotting.tree(
    tree,
    leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.leaves())},
    leaf_deep=False,
)

# %%
# Similarly, if there are no leaf labels, leaf edges are not plotted:

ipx.plotting.tree(
    tree,
)

# %%
# You can finely tune this behaviour. For instance, you can force leaf edges
# despite absence of leaf labels, and change their visual appearance:

ipx.plotting.tree(
    tree,
    leaf_deep=True,
    leafedge_color="tomato",
    leafedge_linewidth=2,
    leafedge_linestyle="-.",
)
