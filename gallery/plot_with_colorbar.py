"""
Colorbar
========

This example from networkx shows how to use mappable colors and colormaps and how to connect them to a colorbar.
"""

import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

seed = 13648  # Seed random number generators for reproducibility
G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
pos = nx.spring_layout(G, seed=seed)

node_sizes = [3 + 1.5 * i for i in range(len(G))]
M = G.number_of_edges()
edge_colors = list(range(2, M + 2))
edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
cmap = plt.cm.plasma

fig, ax = plt.subplots(figsize=(6, 4))

arts = ipx.plot(
    network=G,
    ax=ax,
    layout=pos,
    style={
        "vertex": {
            "size": node_sizes,
            "facecolor": "indigo",
            "edgecolor": "none",
        },
        "edge": {
            "color": edge_colors,
            "alpha": edge_alphas,
            "cmap": cmap,
            "linewidth": 2,
            "offset": 0,
        },
        "arrow": {
            "marker": ">",
            "width": 5,
        },
    },
)
fig.colorbar(
    arts[0].get_edges().get_edges(),
    ax=ax,
)
