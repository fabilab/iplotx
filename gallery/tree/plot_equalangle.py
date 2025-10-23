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
