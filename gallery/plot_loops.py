"""
Loops
=====

This minimal example shows how to style loops, edges that connect a node to itself.
"""

import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

# Create a graph and add a self-loop to node 0
G = nx.complete_graph(3, create_using=nx.DiGraph)
pos = nx.circular_layout(G)
ne = G.number_of_edges()

# Add self-loops to the remaining nodes
G.add_edge(0, 0)
edgelist = [(1, 1), (2, 2)]
G.add_edges_from(edgelist)

# Style the edge lines
linestyle = {e: "-" if e not in edgelist else "--" for e in G.edges()}


ipx.plot(
    G,
    layout=pos,
    vertex_labels=True,
    style={
        "vertex": {
            "size": 30,
            "facecolor": "lightblue",
            "edgecolor": "none",
            "label": {
                "color": "black",
            },
        },
        "edge": {
            "linestyle": linestyle,
            "offset": 0,
            "looptension": 3.5,
        },
        "arrow": {
            "marker": "|>",
        },
    },
)
