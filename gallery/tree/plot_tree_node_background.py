"""
Tree cascades
=============

This example shows how to use `iplotx` to add cascading backgrounds to trees.
"Cascading" here means that each patch (rectangle/wedge/etc.) will cover a node
and all descendants, down to the leaves.
"""

from Bio import Phylo
from io import StringIO
import matplotlib.pyplot as plt
import iplotx as ipx

# Make a tree from a string in Newick format
tree = next(
    Phylo.NewickIO.parse(
        StringIO(
            "(()(()((()())(()()))))",
        )
    )
)

backgrounds = {
    tree.get_nonterminals()[3]: "turquoise",
    tree.get_terminals()[0]: "tomato",
    tree.get_terminals()[1]: "purple",
}

ipx.plotting.tree(
    tree,
    cascade_facecolor=backgrounds,
)

# %%
# Cascading patches have a style option "extend" which affects whether the patches extend to the
# end of the deepest leaf:

ipx.plotting.tree(
    tree,
    layout="vertical",
    cascade_facecolor=backgrounds,
    cascade_extend=True,
)

# %%
# Cascading patches work with radial layouts as well:

# sphinx_gallery_thumbnail_number = 3
ipx.plotting.tree(
    tree,
    layout="radial",
    cascade_facecolor=backgrounds,
    cascade_extend=True,
    aspect=1,
)

# %%
# Cascading patches exclude leaf labels by default, but can be extended to cover leaf labels by
# setting the parameter as follows:

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
ipx.plotting.tree(
    tree,
    ax=ax1,
    layout="radial",
    leaf_labels={leaf: f"L{i + 1}" for i, leaf in enumerate(tree.get_terminals())},
    cascade_facecolor=backgrounds,
    cascade_extend=True,  # Exclude leaf labels
    aspect=1,
    margins=0.4,
    title="Exclude leaf labels",
)
ipx.plotting.tree(
    tree,
    ax=ax2,
    layout="radial",
    leaf_labels={leaf: f"L{i + 1}" for i, leaf in enumerate(tree.get_terminals())},
    cascade_facecolor=backgrounds,
    cascade_extend="leaf_labels",  # Include leaf labels
    aspect=1,
    margins=0.4,
    title="Include leaf labels",
)

# %%
# This extension also works with other layouts, such as horizontal and vertical:

layout_and_orientations = {
    "horizontal": ["left", "right"],
    "vertical": ["descending", "ascending"],
    "radial": ["clockwise", "counterclockwise"],
}

fig, axs = plt.subplots(3, 2, figsize=(9, 13.5))
for i, (layout_name, layout_orientations) in enumerate(layout_and_orientations.items()):
    for j, orientation in enumerate(layout_orientations):
        ipx.plotting.tree(
            tree,
            ax=axs[i, j],
            layout=layout_name,
            layout_orientation=orientation,
            leaf_labels={leaf: f"L{i + 1}" for i, leaf in enumerate(tree.get_terminals())},
            cascade_facecolor=backgrounds,
            cascade_extend="leaf_labels",
            aspect=1,
            title=f"{layout_name} ({orientation})",
            margins=0.1,
        )
