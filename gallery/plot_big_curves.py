"""
Big curves
==========

This example showcases the ability of `iplotx` to curve edges at a certain tension setting.
"""

import igraph as ig
import matplotlib.pyplot as plt
import iplotx as ipx


g = ig.Graph(edges=[(0, 1), (0, 0)])
layout = [[0, 0], [1, 0]]

fig, ax = plt.subplots()
ipx.plot(
    g,
    layout=layout,
    ax=ax,
    edge_curved=True,
    edge_tension=10,
    edge_looptension=10,
)
