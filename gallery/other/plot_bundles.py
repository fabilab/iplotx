"""
Edge strings
============

This example shows how to plot edge strings around a circle.
"""

import networkx as nx
import iplotx as ipx

g = nx.scale_free_graph(30)

# Remove multi-edges and self-loops for clarity
edges = [e[:2] for e in g.edges if (e[0] != e[1]) and (e[2] == 0)]
g = nx.from_edgelist(edges)

layout = nx.circular_layout(g)

ipx.network(
    g,
    layout,
    figsize=(7, 7),
    aspect=1.0,
    node_size=10,
    edge_curved=True,
    edge_tension=-1,
)
