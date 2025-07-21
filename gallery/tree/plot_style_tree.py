"""
Style hierarchy
===============

This example shows a few elements from  ``iplotx``'s style hierarchy.
"""

import matplotlib.pyplot as plt
import iplotx as ipx

tree = {
    "name": "style",
    "children": [
        {
            "name": "vertex",
            "children": [
                {
                    "name": "size",
                },
                {"name": "..."},
            ],
        },
        {"name": "..."},
    ],
}
tree = ipx.ingest.providers.tree.simple.SimpleTree.from_dict(tree)

fig, ax = plt.subplots(figsize=(4, 2.5))
ipx.tree(
    tree,
    ax=ax,
    vertex_labels=True,
    vertex_size=[(50, 25)] * 5,
    vertex_facecolor="none",
    vertex_edgecolor="black",
    vertex_linewidth=1.5,
    vertex_marker="r",
    vertex_label_horizontalalignment="center",
    vertex_label_hmargin=0,
    margins=(0.1, 0.1),
)
ax.invert_yaxis()
