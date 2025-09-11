"""
Breast cancer proteins
======================

This example visualises network of putatively interacting proteins according to the STRING database, filtered to include only entries related to breast cancer.

Vertices are lifted above the edges by increasing their zorder above the edge level, which is 2.
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx


node_data = pd.read_csv("data/breast_cancer_string_network_coordinates.tsv", sep="\t", index_col=0)
edge_data = pd.read_csv("data/breast_cancer_string_interactions_short.tsv", sep="\t")

g = nx.Graph()
g.add_edges_from(edge_data.iloc[:, :2].values)

layout = node_data[["x_position", "y_position"]]
node_data["color_ipx"] = [list(map(lambda x: float(x) / 256.0, x[4:-1].split(","))) for x in node_data["color"]]
node_data["label_color"] = ["white" if np.mean(x) < 0.4 else "black" for x in node_data["color_ipx"]]


fig, ax = plt.subplots(figsize=(8, 8))
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
fig.tight_layout()
