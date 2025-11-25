"""
Multicolor edges
================

Sometimes the need arises to color edges with multiple colors at the same time. That's what multigraphs are for.
This example shows the concept in action.

"""

import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

g = nx.MultiGraph()
g.add_edges_from([(0, 1), (1, 2), (0, 2), (0, 1)])

layout = [(0, 0), (1, 0), (0.5, 1)]

ipx.network(
    g,
    layout,
    edge_color=["black", "gold", "tomato", "tomato"],
    edge_paralleloffset=8,
    edge_linewidth=4,
    vertex_size=40,
    vertex_facecolor="none",
    vertex_edgecolor="black",
    vertex_linewidth=4,
)
