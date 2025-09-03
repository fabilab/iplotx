"""
Simple path
===========

A minimal example of using `iplotx` together with `networkx`.
"""

import networkx as nx
import iplotx as ipx

G = nx.path_graph(8)
pos = nx.spring_layout(G, seed=47)  # Seed layout for reproducibility
ipx.network(G, layout=pos)

# %%
# We can change the color of the vertices and edges with a touch of style:

ipx.network(
    G,
    layout=pos,
    style={
        "vertex": {
            "facecolor": "lightblue",
            "edgecolor": "navy",
            "linewidth": 2.5,
        },
        "edge": {
            "color": "navy",
            "linewidth": 2.5,
        },
    },
)
