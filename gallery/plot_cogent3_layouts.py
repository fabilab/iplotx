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

# %%
# .. warning::
#   This is currently WIP and will see significant API changes soon.

import cogent3
import matplotlib.pyplot as plt
import iplotx as ipx

tree = cogent3.load_tree("data/tree-with-support.json")

fig, ax = plt.subplots(figsize=(5, 4))
art = ipx.tree(
    tree,
    ax=ax,
    leaf_labels=True,
    leaf_deep=False,
)

# %%
# Angular layouts (without waypoints, looks more pointy) are supported:

fig, ax = plt.subplots(figsize=(5, 4))
art = ipx.tree(
    tree,
    layout="horizontal",
    ax=ax,
    leaf_labels=True,
    leaf_deep=False,
    angular=True,
)

# %%
# Vertical and radial layouts are also supported along the same lines:

fig, axs = plt.subplots(3, figsize=(5, 12))
for ax, layout in zip(axs, ("vertical", "radial", "horizontal")):
    art = ipx.tree(
        tree,
        layout=layout,
        orientation="left" if layout == "horizontal" else None,
        ax=ax,
        leaf_labels=True,
        leaf_deep=False,
        angular=True,
    )
