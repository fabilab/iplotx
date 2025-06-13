"""
ETE4 tree
=========

This example shows how to use `iplotx` to plot trees from `ete4`.
"""

from ete4 import Tree
import iplotx as ipx

tree = Tree(
    "((),((),(((),()),((),()))));",
)

ipx.plotting.tree(
    tree,
    aspect=1,
    edge_color="grey",
    edge_linestyle=["--", "-"],
)

# %%
# `iplotx` can compute a radial tree layout as well, and usual style modifications
# work for trees same as networks:

ipx.plotting.tree(
    tree,
    layout="radial",
    style=[
        "tree",
        {
            "edge": {
                "color": "black",
                "linewidth": 4,
            },
        },
    ],
    aspect=1,
)
