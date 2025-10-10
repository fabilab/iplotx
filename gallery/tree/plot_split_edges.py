"""
Split edges
===========

This example shows how to use split edges, i.e. how to apply different styles to
tree edges that are composed of multiple segments. See also :doc:`../zero_dependency/plot_simpletreedataprovider`
for the basic example of how to plot a tree with ``iplotx``'s internal tree representation.

.. note::
    The current implementation only supports two-piece edges.
"""

from collections import defaultdict
import matplotlib.pyplot as plt
import iplotx as ipx

tree = {
    "children": (
        {},
        {
            "children": (
                {
                    "children": (
                        {},
                        {}
                    )
                },
                {
                    "children": (
                        {},
                        {}
                    )
                }
            )
        }
    )
}

tree = ipx.ingest.providers.tree.simple.SimpleTree.from_dict(tree)

ipx.tree(
    tree,
    style="tree",
    edge_split_linestyle=":",
    edge_split_color=defaultdict(lambda: "k", {tree.children[1]: "red"}),
)
