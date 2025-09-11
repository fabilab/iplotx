"""
Cell cycle arrest
=================

This example visualises protein protein interactions involved in the "cell cycle arrest" pathway.

It also shows how to use ``iplotx`` to combine network plots with other matplotlib chart types.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx


node_data = pd.read_csv("data/cell_cycle_arrest_string_network_coordinates.tsv", sep="\t", index_col=0)
edge_data = pd.read_csv("data/cell_cycle_arrest_string_interactions_short.tsv", sep="\t")

g = nx.Graph()
g.add_edges_from(edge_data.iloc[:, :2].values)

layout = node_data[["x_position", "y_position"]]
node_data["color_ipx"] = [list(map(lambda x: float(x) / 256.0, x[4:-1].split(","))) for x in node_data["color"]]
node_data["label_color"] = ["white" if np.mean(x) < 0.4 else "black" for x in node_data["color_ipx"]]


fig = plt.figure(figsize=(9, 6))
ax = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((2, 3), (0, 2))
ax3 = plt.subplot2grid((2, 3), (1, 2))
ipx.network(
    g,
    layout=layout,
    ax=ax,
    vertex_labels=True,
    vertex_marker="r",
    vertex_size="label",
    vertex_facecolor=node_data["color_ipx"].to_dict(),
    vertex_label_color=node_data["label_color"].to_dict(),
    vertex_edgecolor="k",
    vertex_linewidth=0.5,
    edge_alpha=0.2,
    vertex_zorder=3,

)

degrees = np.bincount(list(dict(g.degree()).values()))
bindegs = np.arange(len(degrees))
ax2.bar(bindegs, degrees, color="gray")
ax2.set_xlabel("Node degree")
ax2.set_ylabel("Number of\nproteins")

pagerank = list(nx.pagerank(g).values())
ax3.ecdf(pagerank, color="black", complementary=True)
ax3.axvline(0.1, ls="--", lw=2, color="tomato")
ax3.set_xlabel("Node PageRank")
ax3.set_ylabel("Cumulative")

fig.tight_layout()
