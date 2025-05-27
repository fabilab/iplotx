"""
igraph/networkx compatibility
=============================

The same graph in igraph and networkx, plotted with iplotx.
"""

import igraph as ig
import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

fig, axs = plt.subplots(1, 2, figsize=(6, 3))

axs[0].set_title("igraph")
g1 = ig.Graph.Ring(5, directed=True)
layout1 = g1.layout("circle")
ipx.plot(g1, ax=axs[0], layout=layout1)

axs[1].set_title("networkx")
g2 = nx.cycle_graph(n=5, create_using=nx.DiGraph)
layout2 = nx.layout.circular_layout(g2)
ipx.plot(g2, ax=axs[1], layout=layout2)
