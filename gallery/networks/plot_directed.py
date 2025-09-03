"""
Simple directed graph
======================

This example shows how to plot a simple graph with `iplotx`.
"""

import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

g = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
layout = nx.layout.circular_layout(g)
fig, ax = plt.subplots(figsize=(3, 3))
ipx.plot(g, ax=ax, layout=layout)
