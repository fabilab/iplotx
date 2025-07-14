"""
Animations
==========

This tutorial shows how to animate `iplotx` visualizations using
`matplotlib.animation.FuncAnimation`.

For illustration purposes, we will animate a simple directed graph,
rotating it around its center. We also modify the opacity of the
vertices and edges, just for fun.
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import igraph as ig
import iplotx as ipx

g = ig.Graph.Ring(3, directed=True)
layout = np.asarray(g.layout("circle").coords)

# The animation will rotate the layout of the graph
thetas = np.linspace(0, 2 * np.pi, 101)[:-1]

fig, ax = plt.subplots()
art = ipx.network(g, ax=ax, layout=layout, aspect=1)[0]
ax.set(xlim=[-2, 2], ylim=[-2, 2])


def rotate(vector, theta):
    """Rotate a 2D vector by an angle theta (in radians) clockwise."""
    return vector @ np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
    )


def update(frame):
    # for each frame, update the vertex positions
    theta = thetas[frame]

    # The new layout is the old layout rotated by theta, clockwise
    layout_new = rotate(layout, theta)

    # Also change transparency, slower
    alpha_vertices = 1 - np.abs(np.sin(theta / 2))
    alpha_edges = 0.25 + 0.75 * np.abs(np.sin(theta / 2))

    # Edit the vertices' positions
    art.get_vertices().set_offsets(layout_new)
    art.get_vertices().set_alpha(alpha_vertices)
    art.get_edges().set_alpha(alpha_edges)
    return (art.get_vertices(), art.get_edges())


# Run the animation
ani = animation.FuncAnimation(
    fig=fig,
    func=update,
    frames=len(thetas),
    interval=30,
)
