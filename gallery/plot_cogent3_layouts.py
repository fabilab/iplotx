"""
Cogent3 layouts
===============

This example shows how to reproduce `cogent3` layouts using `iplotx`.

.. note::
  `cogent3` uses `plotly` as their default backend, whereas `iplotx` uses `matplotlib`.
  Each backend has pros and cons. In general, `plotly` is more directed towards web
  visualisation thanks to their JavaScript library, whereas `matplotlib` is probably
  more popular and supported outside of web environments.
"""

import cogent3
import matplotlib.pyplot as plt
import iplotx as ipx

tree = cogent3.load_tree("data/tree-with-support.json")

fig, ax = plt.subplots(figsize=(5, 4))
art = ipx.tree(
    tree,
    layout="horizontal",
    ax=ax,
    vertex_labels="leaves",
)
