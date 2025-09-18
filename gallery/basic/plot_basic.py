"""
Basic network plotting
======================

This example shows how to plot a simple graph with `iplotx`.
"""

import igraph as ig
import matplotlib.pyplot as plt
import iplotx as ipx

g = ig.Graph.Ring(5)
layout = g.layout("circle").coords
fig, ax = plt.subplots(figsize=(3, 3))
ipx.plot(g, ax=ax, layout=layout)
