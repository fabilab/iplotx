"""
Food network
============

This example visualises a food network, with data taken from https://doi.org/10.3389/fevo.2020.588430 (Supplementary Table S1).
"""

from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx


# Read in the data on how many times scat evidenced was found that each animal (column) ate each food item (row)
table = pd.read_csv("data/fevo-08-588430_DataSheet1_S1.csv", skiprows=2, usecols=[0, 1, 2, 3, 6, 9, 12, 15, 18, 21]).iloc[1:]


# Only part of the table is relevant to build the network
adjacency_matrix = table.set_index("OTUs").iloc[:, -6:].astype(int)
# Clean up column names a bit
adjacency_matrix.columns = [x.split(" (")[0] for x in adjacency_matrix.columns]

# Convert to a directed edge list
edge_data = adjacency_matrix.T.stack()
edge_data = edge_data[edge_data > 0].reset_index()
edge_data.columns = ["predator", "prey", "weight"]

# Build graph
g = nx.DiGraph()
g.add_weighted_edges_from(edge_data.values)

# Compute force-directed layout
layout = nx.spring_layout(g, seed=42)
layout.update({
    "Bobcat": [0.4, 0.4],
    "Jackrabbit": [-0.4, -0.4],
    "Gray Fox": [-0.4, 0.4],
    "Puma": [0.4, -0.4],
    "Deer": [0.2, 0.0],
    "Coyote": [-0.2, 0.0],
})

# Compute vertex labels and sizes
vertex_labels = {}
vertex_sizes = {}
for node in g.nodes():
    n_preys = g.out_degree(node)
    if n_preys == 0:
        vertex_labels[node] = ""
        vertex_sizes[node] = 15
    else:
        vertex_labels[node] = node
        vertex_sizes[node] = 55

legend_colord = {
    "Herb": "lawngreen",
    "Woody": "green",
    "Grass": "limegreen",
    "Small Mammal": "khaki",
    "Large Mammal": "saddlebrown",
    "Predator": "tomato",
}
vertex_facecolor = table.set_index("OTUs")["Functional Group"].map(legend_colord).to_dict()
# Add predators
for predator in adjacency_matrix.columns:
    vertex_facecolor[predator] =  legend_colord["Predator"]

edge_color = {key: vertex_facecolor[key[1]] for key in g.edges()}
edge_linewidth = {(u, v): 0.5 * z['weight'] for u, v, z in g.edges(data=True)}


fig, ax = plt.subplots(figsize=(8, 6.6))
ipx.network(
    g,
    layout=layout,
    ax=ax,
    vertex_labels=vertex_labels,
    vertex_size=vertex_sizes,
    vertex_facecolor=vertex_facecolor,
    vertex_edgecolor="black",
    vertex_linewidth=1,
    vertex_label_color="black",
    vertex_zorder=3,
    edge_curved=True,
    edge_tension=2,
    edge_alpha=0.7,
    edge_linewidth=edge_linewidth,
    edge_color=edge_color,
)
ax.legend(
    [plt.Circle((0, 0), 0, facecolor=color, edgecolor="black") for color in legend_colord.values()],
    list(legend_colord.keys()),
    loc="lower center",
    ncol=6,
    frameon=False,
)
fig.tight_layout()

# %%
# We can also visualise each predator individually to get a sense of their differences, using subplots:

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
axs = axs.ravel()
for predator, ax in zip(adjacency_matrix.columns, axs):
    subgraph = g.subgraph([predator] + list(g.successors(predator)))
    sublayout = {k: v for k, v in layout.items() if k in subgraph.nodes()}
    sub_vertex_sizes = {k: 0.5 * v for k, v in vertex_sizes.items() if k in subgraph.nodes()}

    ipx.network(
        subgraph,
        layout=sublayout,
        ax=ax,
        vertex_labels=False,
        vertex_size=sub_vertex_sizes,
        vertex_facecolor=vertex_facecolor,
        vertex_edgecolor="black",
        vertex_linewidth=1,
        vertex_zorder=3,
        edge_curved=True,
        edge_tension=2,
        edge_alpha=0.7,
        edge_color=edge_color,
        edge_linewidth=edge_linewidth,
        title=predator,
    )
