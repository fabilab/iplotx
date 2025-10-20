"""
Double tree
===========

This example shows how to use `iplotx` to plot two trees facing each other, which is typical in coevolutionary studies.
"""

from ete4 import Tree
from matplotlib import pyplot as plt
import iplotx as ipx

tree1 = Tree(
    "((),((),(((),()),((),()))));",
)


tree2 = Tree(
    "((),((),(),((),())),());",
)

fig, ax = plt = plt.subplots(figsize=(9, 4))

# Plot first tree on the left
ipx.plotting.tree(
    tree1,
    ax=ax,
    aspect=1,
    edge_color="tomato",
    leaf_deep=True,
)

# Plot second tree on the right, facing left
ipx.plotting.tree(
    tree2,
    ax=ax,
    aspect=1,
    edge_color="steelblue",
    layout="horizontal",
    layout_orientation="left",
    layout_start=(11, 0),
    leaf_deep=True,
)

# Add lines connecting corresponding leaves
matches = [
    (0, 0),
    (1, 4),
    (2, 5),
    (3, 2),
    (4, 1),
    (5, 5),
]
for y1, y2 in matches:
    ax.plot(
        [5.2, 6.8], [y1, y2], color="gray", linewidth=2,
    )
