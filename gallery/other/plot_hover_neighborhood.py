"""
Hover neighborhoods
===================

This example shows how to highlight a node and its neighborhood by hovering with the mouse on it.

.. warning::
  This example will run in a Python, IPython, or Jupyter session, however the
  interactive functionality is not visible on the HTML page. Download the code
  at the end of this page and run it in a local Python environment to see
  the results.
"""

import matplotlib.pyplot as plt
import igraph as ig
import iplotx as ipx

g = ig.Graph.Erdos_Renyi(n=40, m=120)
layout = g.layout()

fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 0.2]}, figsize=(8, 6))
art = ipx.network(
    g,
    layout=layout,
    ax=axs[0],
    aspect=1,
    vertex_zorder=5,
)[0]
base_nodecolors = art.get_vertices().get_facecolors()
nodecolors = base_nodecolors.copy()
edgecolors = art.get_edges().get_edgecolors()
edgewidths = art.get_edges().get_linewidths()

axs[1].set_axis_off()
txt = axs[1].text(0.5, 0.5, "", ha="center", va="center", wrap=True,
                  fontsize=40)


def hover_callback(event):
    """React to mouse hovering over vertices."""
    if event.inaxes == axs[0]:
        cont, ind = art.get_vertices().contains(event)

        # Reset everyone's color
        is_base = (nodecolors == base_nodecolors).all()

        # If mouse is over a vertex, change the color around there
        # and redraw
        if cont:
            i = ind["ind"][0]
            nodecolors[:] = [0, 0, 0, 0.3]
            nodecolors[g.neighbors(i)] = [1, 0, 0, 0.6]
            nodecolors[i] = [1, 0, 0, 1]
            art.get_vertices().set_facecolors(nodecolors)
            edgecolors[:] = [0, 0, 0, 0.3]
            edgecolors[g.incident(i)] = [0, 0, 0, 1]
            art.get_edges().set_edgecolors(edgecolors)
            edgewidths[:] = 1
            edgewidths[g.incident(i)] = 2
            art.get_edges().set_linewidths(edgewidths)

            txt.set_text(str(i))
        # Otherwise, change back to base and redraw
        elif not is_base:
            nodecolors[:] = base_nodecolors
            vertex_artist.set_facecolors(nodecolors)
            edgecolors[:] = [0, 0, 0, 1]
            art.get_edges().set_edgecolors(edgecolors)
            edgewidths[:] = 1
            art.get_edges().set_linewidths(edgewidths)
            txt.set_text("")
        # If nothing changed, no need to redraw
        else:
            return

        # Redraw if needed
        fig.canvas.draw_idle()


fig.canvas.mpl_connect(
    "motion_notify_event",
    hover_callback,
)
