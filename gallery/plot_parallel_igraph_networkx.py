"""
igraph/networkx compatibility
=============================

The same graph in igraph and networkx, plotted with iplotx.
"""

import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

g1 = ig.Graph.Ring(5, directed=True)
g2 = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
layout1 = g1.layout("circle").coords
layout2 = nx.layout.circular_layout(g2)

fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].set_title("igraph")
ipx.plot(g1, ax=axs[0], layout=layout1)
axs[1].set_title("networkx")
ipx.plot(g2, ax=axs[1], layout=layout2)
