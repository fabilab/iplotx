"""
Social circles
==============

This example inspired by SocNetV (https://socnetv.org/data/uploads/screenshots/32/socnetv-32-example-dot.png), shows social connections within circles.
It also demonstrates how to combine ``iplotx`` with other matplotlib artists.

"""

import igraph as ig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iplotx as ipx


edge_data = [
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 1),
    (1, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
]
edge_data = pd.DataFrame(edge_data).astype(str)

g = ig.Graph.DataFrame(
    edge_data,
    use_vids=False,
    directed=True,
)


def coords(radius, angle):
    angle = np.deg2rad(angle)
    return radius * np.cos(angle), radius * np.sin(angle)


radii = [1, 3, 5, 6, 7, 12]
layout = [
    coords(radii[0], 0),
    coords(radii[4], -30),
    coords(radii[3], -70),
    coords(radii[2], -100),
    coords(radii[1], -135),
    coords(radii[5], 180),
    coords(radii[5], 180 - 36),
    coords(radii[5], 180 - 36 * 2),
    coords(radii[5], 180 - 36 * 3),
    coords(radii[5], 180 - 36 * 4),
]


fig, ax = plt.subplots(figsize=(8, 8))
ipx.network(
    g,
    layout,
    ax=ax,
    vertex_marker="s",
    vertex_facecolor="red",
    edge_shrink=3,
    edge_zorder=2,
    vertex_zorder=3,
    vertex_label_color="black",
    vertex_labels=list(g.vs["name"]),
    zorder=3,
)

for radius in radii:
    ax.add_patch(
        plt.Circle(
            (0, 0), radius,
            facecolor="none", edgecolor="red", ls="dashed",
            zorder=1,
        )
    )

ax.set(xlim=(-13, 13), ylim=(-13, 13))
fig.tight_layout()
