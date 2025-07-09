"""
scikit-bio tree
===============

This example shows how to use `iplotx` to plot trees from `skbio` or scikit-bio.
"""

from io import StringIO
from skbio import TreeNode
import iplotx as ipx

tree = TreeNode.read(
    StringIO(
        "((),((),(((),()),((),()))));",
    )
)
ipx.plotting.tree(
    tree,
    layout="radial",
    aspect=1,
    edge_color="purple",
)

# %%
# `iplotx` can compute a standard rectangular tree layout as well, and usual style modifications
# work for trees same as networks:

ipx.plotting.tree(
    tree,
    style=[
        "tree",
        {
            "edge": {
                "color": "deeppink",
                "linewidth": 4,
            },
        },
    ],
)
