"""
Labels and colors
=================

This example from networkx demonstrates how to use transparency to overlay multiple plots with distinct highlights.
"""

import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

G = nx.cubical_graph()
pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
nx.set_node_attributes(G, pos, "pos")  # Will not be needed after PR 7571
labels = iter(
    [
        r"$a$",
        r"$b$",
        r"$c$",
        r"$d$",
        r"$\alpha$",
        r"$\beta$",
        r"$\gamma$",
        r"$\delta$",
    ]
)
nx.set_node_attributes(
    G,
    {
        n: {
            "color": "tab:red" if n < 4 else "tab:blue",
            "label": next(labels),
        }
        for n in G.nodes()
    },
)

fig, ax = plt.subplots(figsize=(5, 4))
ipx.plot(
    G,
    ax=ax,
    layout="pos",
    vertex_labels=True,
    style={
        "vertex": {
            "facecolor": G.nodes.data("color"),
            "alpha": G.nodes.data("alpha"),
            "size": 40,
            "label": {"size": 22, "color": "whitesmoke"},
            "zorder": 7,
        },
        "edge": {
            "color": "tab:grey",
        },
    },
)
for i in range(2):
    subG = G.subgraph(range(i * 4, (i + 1) * 4))
    ipx.plot(
        subG,
        layout="pos",
        ax=ax,
        style={
            "edge": {
                "color": ["tab:red", "tab:blue"][i],
                "alpha": 0.5,
                "linewidth": 8,
                "zorder": 0,
            },
            "vertex": {
                "size": 35,
                "alpha": 0,
            },
        },
    )
plt.tight_layout()
