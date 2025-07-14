"""
Tree of trees
=============

This example shows how to plot trees at the leaf of a tree, using the `iplotx` library.
Note that there are a few conceivable ways to do similar things as this in
`matplotlib` and this is just one such approach, using inset axes.
"""

from ete4 import Tree
import matplotlib.pyplot as plt
import iplotx as ipx

tree = Tree(
    "(,((),(((),()),((),()))));",
)
subtree_strings = [
    "((a,b),(c,d));",
    "(a,((b,c),d));",
    "(a,b,c,d);",
    "(a,(b,(c,d)));",
    "(a,b);",
    "(a,(b,c,d));",
]

# Plot the initial tree
fig, ax = plt.subplots(figsize=(8, 5))
art = ipx.tree(
    tree,
    ax=ax,
    aspect=1,
    edge_color="grey",
)

for i, leaf in enumerate(tree.leaves()):
    # Create an axis in the rough proximity
    # and with the rough diemension of a leaf,
    # with some padding
    x, y = art.get_layout().T[leaf].values
    subax = ax.inset_axes(
        (x + 0.1, y - 0.4, 1.2, 0.8),
        transform=ax.transData,
    )

    # For each leaf, make a tree
    subtree = Tree(subtree_strings[i])
    # Plot the tree with dashed lines
    ipx.tree(
        subtree,
        ax=subax,
        edge_linestyle="--",
    )

# %%
# Combining this with other elements, e.g. cascading backgrounds, is possible too:

fig, ax = plt.subplots(figsize=(8, 5))
art = ipx.tree(
    tree,
    ax=ax,
    aspect=1,
    edge_color="grey",
    cascade_facecolor={
        list(tree.leaves())[0]: "lightblue",
        list(tree.traverse("preorder"))[2]: "pink",
    },
)
for i, leaf in enumerate(tree.leaves()):
    x, y = art.get_layout().T[leaf].values
    subax = ax.inset_axes(
        (x + 0.1, y - 0.4, 1.2, 0.8),
        transform=ax.transData,
    )
    # Set the inset axes background transparent
    subax.patch.set_alpha(0)

    subtree = Tree(subtree_strings[i])
    ipx.tree(
        subtree,
        ax=subax,
        edge_linestyle="--",
    )
