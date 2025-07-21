"""
Tree element showroom
=====================

This example showcases plot elements for tree styling in ``iplotx``.
"""

import matplotlib.pyplot as plt
import iplotx as ipx

tree = {
    "name": "Vertex\nlabel",
    "children": [
        {
            "children": [
                {},
                {
                    "children": [{}, {}]
                },
            ],
        },
        {},
    ],
}
tree = ipx.ingest.providers.tree.simple.SimpleTree.from_dict(tree)

fig, ax = plt.subplots(figsize=(3.9, 3.3))
ipx.tree(
    tree,
    ax=ax,
    vertex_size=[0] * 6 + [(45, 35)],
    vertex_marker="r",
    vertex_facecolor=["none"] * 6 + ["orange"],
    vertex_alpha=0.5,
    vertex_edgecolor="black",
    vertex_linewidth=1.5,
    vertex_labels=True,
    vertex_label_hmargin=0,
    leaf_labels={tree.children[0].children[0]: "Leaf\nlabel"},
    leaf_label_bbox_facecolor="lightcoral",
    leaf_label_bbox_edgecolor="black",
    leaf_label_bbox_alpha=0.9,
    leaf_deep=True,
    cascade_facecolor={
        tree.children[0].children[1]: "gold",
    },
    edge_linewidth=3,
    edge_split_linewidth=3,
    edge_split_linestyle=["-."] + ["-"] * 6,
    edge_split_color=["tomato"] + ["black"] * 6,
    edge_capstyle="round",
    edge_split_capstyle="round",
    margins=(0.21, 0.08),
)
plt.ion(); plt.show()
