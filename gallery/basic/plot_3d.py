"""
3D layouts
==========

This example shows how to visualise graphs or networks in 3D using `iplotx`. Of course, a 3D layout is needed
for this. Here, we use the Fruchterman-Reingold layout algorithm from ``igraph`` to generate a 3D layout.

.. warning::
    3D visualisation does not support all features of 2D visualisation yet. Curved edges, waypoints, and labels
    are currently unsupported. PRs are welcome!
"""

import igraph as ig
import iplotx as ipx

# Make the graph
g = ig.Graph.Erdos_Renyi(30, m=50)

# Make a 3D layout
layout = g.layout_fruchterman_reingold_3d()

# Visualise the graph in 3D
ipx.network(
    g,
    layout,
    vertex_alpha=0.7,
    edge_alpha=0.4,
    figsize=(8, 8),
)
