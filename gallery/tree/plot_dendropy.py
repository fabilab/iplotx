"""
Dendropy tree
=============

This example shows how to use ``iplotx`` to plot trees from ``dendropy``.
"""

from dendropy import Tree
import iplotx as ipx

tree = Tree.get(data="((,(,((,),(,)))));", schema="newick")

ipx.plotting.tree(
    tree,
    aspect=1,
    edge_color="grey",
    edge_linestyle=["--", "-"],
)

# %%
# `iplotx` can compute a radial tree layout as well, and usual style modifications
# work for trees same as networks.

# sphinx_gallery_thumbnail_number = 2
ipx.plotting.tree(
    tree,
    layout="radial",
    layout_orientation="right",
    style=[
        "tree",
        {
            "edge": {
                "color": "navy",
                "linewidth": 4,
            },
            "leaf": {
                "label": {
                    "hmargin": 15,
                }
            },
            "leafedge": {
                "color": "steelblue",
                "linewidth": 2,
            },
        },
    ],
    leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.leaf_nodes())},
    aspect=1,
    margin=(0.3, 0.1),
)
