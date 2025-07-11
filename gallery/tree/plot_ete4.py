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
# work for trees same as networks. Moreover, trees have a layout style option to
# choose the starting angle and angular span of the radial layout.

# sphinx_gallery_thumbnail_number = 2
ipx.plotting.tree(
    tree,
    layout="radial",
    layout_orientation="right",
    style=[
        "tree",
        {
            "edge": {
                "color": "deeppink",
                "linewidth": 4,
            },
            "layout": {
                "start": -180,
                "span": 180,
            },
            "leaf": {
                "label": {
                    "hmargin": 15,
                }
            },
            "leafedge": {
                "color": "purple",
                "linewidth": 2,
            },
        },
    ],
    leaf_labels={leaf: str(i + 1) for i, leaf in enumerate(tree.leaves())},
    aspect=1,
    margin=0.1,
)
