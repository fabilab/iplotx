"""
Traveling salesman
==================

This example from networkx demonstrates simple styling of nodes and edges.
"""

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.approximation as nx_app
import math
import iplotx as ipx


G = nx.random_geometric_graph(20, radius=0.4, seed=3)
pos = nx.get_node_attributes(G, "pos")

# Depot should be at (0,0)
pos[0] = (0.5, 0.5)

H = G.copy()


# Calculating the distances between the nodes as edge's weight.
for i in range(len(pos)):
    for j in range(i + 1, len(pos)):
        dist = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
        dist = dist
        G.add_edge(i, j, weight=dist)

cycle = nx_app.christofides(G, weight="weight")
edge_list = list(nx.utils.pairwise(cycle))
edge_color = {
    tuple(e): "red" if e in edge_list or e[::-1] in edge_list else "none" for e in G.edges()
}
nx.set_edge_attributes(G, edge_color, "color")

ipx.plot(
    H,
    layout=pos,
    style={
        "edge": {
            "color": "blue",
            "linewidth": 1,
        },
        "vertex": {
            "facecolor": "none",
            "edgecolor": "none",
        },
    },
)
ipx.plot(
    G,
    ax=plt.gca(),
    layout=pos,
    vertex_labels=True,
    style={
        "edge": {
            "color": G.edges.data("color"),
            "linewidth": 3,
        },
        "vertex": {
            "size": 20,
            "facecolor": "steelblue",
            "edgecolor": "none",
            "label": {
                "color": "black",
            },
        },
    },
)

print("The route of the traveller is:", cycle)
