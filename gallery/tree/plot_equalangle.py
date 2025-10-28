"""
Equal angle layout
==================

This example showcases the "equal angle" layout. This layout is inspired by the `ggtree <https://yulab-smu.top/treedata-book/chapter4.html>`_ layout with the same name, which originally comes from Joseph Felsenstein's book "Inferring Phylogenies".
"""

from cogent3.phylo import nj
import numpy as np
import iplotx as ipx
import matplotlib.pyplot as plt

nleaves = 14
distance_dict = {}
for i in range(nleaves):
    for j in range(i):
        distance_dict[(str(i), str(j))] = np.random.rand()
tree = nj.nj(distance_dict)

ipx.plotting.tree(
    tree,
    layout="equalangle",
)


# %%
# The "equal daylight" layout is an adjustment of the equal angle layout that attempts to spread out leaves more evenly
# for imbalanced trees. ``iplotx`` has an experimental implementation of this layout.
#
# .. warning:: "Experimental" means you can use it but the API and resulting layout may change in future releases.

ipx.plotting.tree(
    tree,
    layout="daylight",
)
