"""
Basics of styling
======================

This example shows a few simple styling options available in `iplotx`.
"""

import igraph as ig
import matplotlib.pyplot as plt
import iplotx as ipx

g = ig.Graph.Ring(5)
layout = g.layout("circle").coords
style = {
    "vertex": {
        "facecolor": ["purple", "pink", "black"],
        "edgecolor": "black",
    },
    "edge": {
        "linestyle": "--",
        "linewidth": 2,
        "color": "deeppink",
    },
}
fig, ax = plt.subplots(figsize=(3, 3))
ipx.plot(g, ax=ax, layout=layout, style=style)

# %%
# There is no question that this network looks much nicer now.
