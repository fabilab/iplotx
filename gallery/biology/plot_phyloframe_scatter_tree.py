"""
Phyloframe scatter tree
=======================

This example shows how to combine a phylogenetic tree with a scatter plot
overlay using `phyloframe <https://github.com/mmore500/phyloframe>`_'s
iplotx integration.

Phyloframe represents phylogenies as DataFrames in the `alife standard
<https://alife-data-standards.github.io/alife-data-standards/>`_ format.
When installed, phyloframe registers an iplotx data provider so that
alife-standard DataFrames can be passed directly to :func:`iplotx.tree`.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import iplotx as ipx
from phyloframe import legacy as pfl
from phyloframe.legacy import alifestd_to_iplotx_pandas

# %%
# Build a synthetic phylogeny DataFrame
# --------------------------------------
# Create a small alife-standard phylogeny with trait metadata.

rng = np.random.default_rng(42)
n_nodes = 31

ancestor_id = np.zeros(n_nodes, dtype=int)
# Root points to itself (alife standard convention)
ancestor_id[0] = 0
# Build a balanced binary tree
for i in range(1, n_nodes):
    ancestor_id[i] = (i - 1) // 2

phylogeny_df = pd.DataFrame({
    "id": np.arange(n_nodes),
    "ancestor_id": ancestor_id,
    "ancestor_list": [f"[{a}]" for a in ancestor_id],
    "origin_time": [
        int(np.floor(np.log2(i + 1))) if i > 0 else 0
        for i in range(n_nodes)
    ],
    "trait": rng.normal(size=n_nodes),
})

# Collapse unifurcations (no-op here but standard practice)
phylogeny_df = pfl.alifestd_collapse_unifurcations(
    phylogeny_df, mutate=True,
)

# %%
# Draw the tree and overlay a scatter plot
# -----------------------------------------
# Pass the phylogeny DataFrame directly to :func:`iplotx.tree`.
# Phyloframe's data provider handles conversion automatically —
# no dendropy or other tree library needed.

fig, ax = plt.subplots(figsize=(8, 6))

tree_artist = ipx.tree(
    alifestd_to_iplotx_pandas(phylogeny_df),
    ax=ax,
    layout="vertical",
    margins=0.0,
    edge_linewidth=1.5,
)

# Read node positions from the tree layout
ipx_layout = tree_artist.get_layout()
xs, ys = ipx_layout.T.to_numpy()

# Build a lookup from node name (= string id) to position
pos = {node.name: (x, y) for node, (x, y) in zip(ipx_layout.index, zip(xs, ys))}

# Map positions back onto the DataFrame
phylogeny_df["x"] = phylogeny_df["id"].astype(str).map(lambda k: pos.get(k, (np.nan,))[0])
phylogeny_df["y"] = phylogeny_df["id"].astype(str).map(lambda k: pos[k][1] if k in pos else np.nan)

# Scatter plot coloured by trait value
sc = ax.scatter(
    phylogeny_df["x"],
    phylogeny_df["y"],
    c=phylogeny_df["trait"],
    cmap="coolwarm",
    s=40,
    zorder=5,
    edgecolors="white",
    linewidths=0.5,
)
fig.colorbar(sc, ax=ax, label="Trait value")
ax.set_title("Phylogeny coloured by trait")
fig.tight_layout()

# %%
# Tree from a Newick string
# --------------------------
# Phyloframe can also parse Newick strings into alife-standard DataFrames
# via :func:`phyloframe.legacy.alifestd_from_newick`.

newick = "((Human:1,Chimp:1.2):3,(Mouse:2,(Zebrafish:3,Fugu:2.5):1.5):4);"
newick_df = pfl.alifestd_from_newick(newick)

fig2, ax2 = plt.subplots(figsize=(8, 4))
ipx.tree(
    alifestd_to_iplotx_pandas(newick_df),
    ax=ax2,
    layout="horizontal",
    leaf_labels=True,
    edge_linewidth=2,
    margins=(0.2, 0),
)
ax2.set_title("Phylogeny from Newick string")
fig2.tight_layout()
