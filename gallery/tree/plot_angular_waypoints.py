"""
Angular layout and edge waypoints
=================================

This example demonstrates `iplotx`'s capability to style tree appearance with
an angular look, either for the whole tree or specific edges.

.. tip::
    "Waypoints" and "angular" are opposites. When an edge uses waypoints, you
    can say it is using an angular style.

"""

from Bio import Phylo
from io import StringIO
import matplotlib.pyplot as plt
import iplotx as ipx

tree = Phylo.read(
    StringIO("(((A,B),(C,D)),(E,F,G));"),
    format="newick",
)

fig, axs = plt.subplots(2, 2, figsize=(9, 9))
ipx.tree(
    tree,
    ax=axs[0, 0],
    title="Default\n(with waypoints)",
)
ipx.tree(
    tree,
    ax=axs[0, 1],
    layout_angular=True,
    title="Angular",
)
ipx.tree(
    tree,
    ax=axs[1, 0],
    edge_waypoints=False,
    title="Angular\n(via edge_waypoints)",
)
ipx.tree(
    tree,
    ax=axs[1, 1],
    edge_waypoints=[True, False],
    title="Mixed",
)

# %%
# .. warning::
#     For waypoints, ``None`` and the string ``"none"`` are interpreted differently
#     by ``iplotx``. The latter ``"none"`` means no waypoints, whereas the Python
#     singleton ``None`` means letting ``iplotx`` determine the correct type of
#     waypoints. ``None`` is equivalent to ``True``, while ``"none"`` is equivalent
#     to ``False``.
#
# You can even specify angularity on a per-edge basis with a defaultdict:

from collections import defaultdict

ipx.tree(
    tree,
    edge_waypoints=defaultdict(lambda: True, {tree.clade[0, 1]: False}),
    title="Mixed\n(per-edge)",
)

# %%
# .. note::
#   ``iplotx``'s fallback for per-element styling is the constructor for the
#   **type** used by the first value of the dictionary, in this case ``bool()``
#   which returns ``False``. Therefore, if we are ok with angular edges as the
#   fallback, we can skip the defaultdict altogether:

ipx.tree(
    tree,
    edge_waypoints={tree.clade[0, 1]: True},
    title="Mixed per-edge\n(default fallback)",
)

# %%
# You can also mix booleans and other types, although it is not very readable:

ipx.tree(
    tree,
    edge_waypoints=[True, "none"],
    title="Mixed",
)
