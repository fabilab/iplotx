"""
TCA cycle
=========

This example visualises a subset of the tricarboxylic acid (TCA) cycle, also known as the Krebs cycle or citric acid cycle.

"""

from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx



g = nx.DiGraph()
g.add_edges_from([
    ("oxaloacetate", "citrate"),
    ("citrate", "isocitrate"),
    ("isocitrate", "alpha-ketoglutarate"),
    ("alpha-ketoglutarate", "succinyl-CoA"),
    ("succinyl-CoA", "succinate"),
    ("succinate", "fumarate"),
    ("fumarate", "malate"),
    ("malate", "oxaloacetate"),
])
edge_labels = {
    (u, v): str(i+1) for i, (u, v) in enumerate(g.edges)
}


layout = pd.DataFrame(nx.circular_layout(g)).T
layout.values[:] = layout.values @ np.array([[0, -1], [1, 0]])
layout[1] *= -1

g.add_edges_from([
    ("acetyl CoA", "CoA-SH"),
    ("NAD+", "NADH"),
    ("NAD+ ", "NADH "),
    ("NAD+  ", "NADH  "),
    ("GDP", "GTP"),
    ("GTP", "GDP"),
    ("ADP", "ATP"),
])

layout.loc["acetyl CoA"] = [0.27, 1.18]
layout.loc["CoA-SH"] = [0.75, 1.05]
layout.loc["NAD+"] = [0.68, -0.15]
layout.loc["NADH"] = [0.61, -0.45]
layout.loc["NAD+ "] = [0.65, -1.03]
layout.loc["NADH "] = [0.35, -1.2]
layout.loc["NAD+  "] = [-0.65, 1.03]
layout.loc["NADH  "] = [-0.35, 1.2]
layout.loc["GDP"] = [-0.25, -1.18]
layout.loc["GTP"] = [-0.7, -0.98]
layout.loc["ATP"] = [-0.35, -1.45]
layout.loc["ADP"] = [-0.8, -1.25]

edge_tension = {
    ("oxaloacetate", "citrate"): -0.8,
    ("citrate", "isocitrate"): -0.8,
    ("isocitrate", "alpha-ketoglutarate"): -0.8,
    ("alpha-ketoglutarate", "succinyl-CoA"): -0.8,
    ("succinyl-CoA", "succinate"): -0.8,
    ("succinate", "fumarate"): -0.8,
    ("fumarate", "malate"): -0.8,
    ("malate", "oxaloacetate"): -0.8,
    ("acetyl CoA", "CoA-SH"): 5.7,
    ("NAD+", "NADH"): -10.0,
    ("NAD+ ", "NADH "): 8.0,
    ("NAD+  ", "NADH  "): 8.0,
    ("GDP", "GTP"): 4.0,
    ("GTP", "GDP"): 3.2,
    ("ADP", "ATP"): -3.2,
}

edge_color = defaultdict(
    lambda: "black",
    {
        ("acetyl CoA", "CoA-SH"): "tomato",
        ("ADP", "ATP"): "gold",
    },
)

vertex_facecolor = defaultdict(
    lambda: "white",
    {
        "acetyl CoA": "tomato",
        "CoA-SH": "tomato",
        "ADP": "gold",
        "ATP": "gold",
    },
)

for (u, v) in g.edges:
    if (u, v) not in edge_labels:
        edge_labels[(u, v)] = ""


# Visualise
fig, ax = plt.subplots(figsize=(7, 8.5))
ipx.network(
    g,
    layout=layout,
    ax=ax,
    style="hollow",
    vertex_labels=True,
    edge_curved=True,
    edge_tension=edge_tension,
    edge_labels=edge_labels,
    edge_label_rotate=True,
    edge_color=edge_color,
    vertex_facecolor=vertex_facecolor,
    vertex_alpha=0.7,
)
fig.tight_layout()
