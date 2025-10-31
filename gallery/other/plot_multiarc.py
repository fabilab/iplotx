"""
Multiarc
========

This example shows how to plot a bunch of arcs.
"""

import igraph as ig
import matplotlib.pyplot as plt
import iplotx as ipx

g = ig.Graph.Ring(4, directed=True)
layout = [[1, 0], [0, 1], [-1, 0], [0, -1]]

ipx.network(
    g,
    layout,
    vertex_labels=True,
    aspect=1,
    edge_tension=1,
    edge_arc=True,
    edge_arrow={"marker": ">","width": 6},
    edge_linewidth=3,
)
