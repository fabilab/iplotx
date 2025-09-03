"""
Shortest path
=============

This example from networkx shows how to use edge labels and their rotation.
"""

import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

G = nx.Graph()
G.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H"])
G.add_edge("A", "B", weight=4)
G.add_edge("A", "H", weight=8)
G.add_edge("B", "C", weight=8)
G.add_edge("B", "H", weight=11)
G.add_edge("C", "D", weight=7)
G.add_edge("C", "F", weight=4)
G.add_edge("C", "I", weight=2)
G.add_edge("D", "E", weight=9)
G.add_edge("D", "F", weight=14)
G.add_edge("E", "F", weight=10)
G.add_edge("F", "G", weight=2)
G.add_edge("G", "H", weight=1)
G.add_edge("G", "I", weight=6)
G.add_edge("H", "I", weight=7)

# Find the shortest path from node A to node E
path = nx.shortest_path(G, "A", "E", weight="weight")

# Create a list of edges in the shortest path
path_edges = list(zip(path, path[1:]))
nx.set_node_attributes(G, nx.spring_layout(G, seed=37), "pos")
nx.set_edge_attributes(
    G,
    {
        (u, v): {
            "color": (
                "red" if (u, v) in path_edges or tuple(reversed((u, v))) in path_edges else "black"
            ),
            "label": d["weight"],
        }
        for u, v, d in G.edges(data=True)
    },
)

fig, ax = plt.subplots()
ipx.plot(
    G,
    ax=ax,
    layout="pos",
    vertex_labels=True,
    edge_labels=True,
)
