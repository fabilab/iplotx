"""
Filled edges
============

Most of the time edges have no "filling", but what if they did? Then you would
get this crazy example.

.. tip::
  Edges are not supposed to be filled, use at your own risk!
"""

import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

g = nx.Graph()
g.add_edges_from([(0, 1), (1, 2), (0, 2)])

layout = [(0, 0), (1, 0), (2, 0)]

artist = ipx.network(
    g,
    layout,
    edge_arc=True,
    edge_tension=-1,
    vertex_facecolor="#005F73",
    vertex_edgecolor="#001219",
    vertex_linewidth=2,
)[0]

edge_artist = artist.get_edges()
edge_artist.set_facecolors([
    "#94D2BD66",
    "#EE9B0066",
    "#AE201266",
])
