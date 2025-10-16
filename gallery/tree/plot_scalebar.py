"""
Scale bar
=========

This example shows how to add a scale bar to an ``iplotx`` tree.
"""

from dendropy import Tree
import iplotx as ipx

tree = Tree.get(data="((,(,((,),(,)))));", schema="newick")

tree_artist = ipx.plotting.tree(
    tree,
    edge_color="grey",
)
tree_artist.scalebar(
    loc="upper left",
)
