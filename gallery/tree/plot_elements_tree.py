"""
Tree element showroom
=====================

This example showcases plot elements for tree styling in ``iplotx``.
"""

import matplotlib.pyplot as plt
import iplotx as ipx

tree = {
    "name": "V1",
    "children": [
        {
            "children": [
                {"name": "V2"},
                {
                    "children": [{}, {}]
                },
            ],
        },
        {},
    ],
}
tree = ipx.ingest.providers.tree.simple.SimpleTree.from_dict(tree)

fig, ax = plt.subplots(figsize=(4, 4))
ipx.tree(
    tree,
    ax=ax,
    vertex_size=[(25, 20)] + [0] * 5 + [(25, 20)],
    vertex_marker="r",
    vertex_facecolor=["darkorange"] + ["none"] * 5 + ["tomato"],
    vertex_alpha=0.5,
    vertex_edgecolor="black",
    vertex_linewidth=1.5,
    vertex_labels=True,
    vertex_label_hmargin=0,
    leaf_labels=True,
    leaf_deep=True,
    cascade_facecolor={
        tree.children[0].children[1]: "gold",
    },
    edge_linewidth=2,
    edge_split_linewidth=[4] + [2] * 6,
    edge_split_linestyle=["-."] + ["-"] * 6,
    edge_split_color=["tomato"] + ["black"] * 6,
)
plt.ion(); plt.show()
