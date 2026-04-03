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
import polars as pl
import iplotx as ipx
from phyloframe import legacy as pfl
from phyloframe.legacy import alifestd_to_iplotx_polars

# %%
# A hardcoded primate phylogeny
# ------------------------------
# Define a small primate phylogeny as a Polars DataFrame in the alife
# standard format.  Only ``id``, ``ancestor_id``, and ``origin_time``
# columns are required — no ``ancestor_list`` column needed.

phylogeny_df = pl.DataFrame({
    "id":          [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12],
    "ancestor_id": [0,  0,  0,  1,  1,  2,  2,  3,  3,  5,  5,  6,  6],
    "origin_time": [0,  2,  3,  4,  5,  6,  5,  7,  7,  8,  8,  7,  7],
    "taxon_label": ["", "", "", "", "", "", "", "Human", "Chimp",
                    "Gorilla", "Orangutan", "Macaque", "Marmoset"],
    "body_mass_kg": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, 70, 45, 160, 60, 8, 0.4],
})

# %%
# Draw the tree and overlay a scatter plot
# -----------------------------------------
# Pass the phylogeny DataFrame directly to :func:`iplotx.tree` via
# phyloframe's data provider — no dendropy or other tree library needed.

fig, ax = plt.subplots(figsize=(8, 6))

tree_artist = ipx.tree(
    alifestd_to_iplotx_polars(phylogeny_df),
    ax=ax,
    layout="vertical",
    leaf_labels=True,
    margins=0.0,
    edge_linewidth=1.5,
)

# Read node positions from the tree layout
ipx_layout = tree_artist.get_layout()
xs, ys = ipx_layout.T.to_numpy()

# Build a lookup from node name to position
pos = {
    node.name: (x, y)
    for node, (x, y) in zip(ipx_layout.index, zip(xs, ys))
}

# Map positions back onto the DataFrame and drop internal nodes
phylogeny_df = phylogeny_df.with_columns(
    pl.col("id").cast(pl.Utf8).replace_strict(
        {k: v[0] for k, v in pos.items()}, default=None,
    ).cast(pl.Float64).alias("x"),
    pl.col("id").cast(pl.Utf8).replace_strict(
        {k: v[1] for k, v in pos.items()}, default=None,
    ).cast(pl.Float64).alias("y"),
)
leaves = phylogeny_df.filter(pl.col("body_mass_kg").is_not_null())

# Scatter plot sized by body mass
sc = ax.scatter(
    leaves["x"].to_numpy(),
    leaves["y"].to_numpy(),
    c=np.log10(leaves["body_mass_kg"].to_numpy()),
    cmap="YlOrRd",
    s=80,
    zorder=5,
    edgecolors="black",
    linewidths=0.5,
)
fig.colorbar(sc, ax=ax, label="log₁₀ body mass (kg)")
ax.set_title("Primate phylogeny coloured by body mass")
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
    alifestd_to_iplotx_polars(newick_df.pipe(pl.from_pandas)),
    ax=ax2,
    layout="horizontal",
    leaf_labels=True,
    edge_linewidth=2,
    margins=(0.2, 0),
)
ax2.set_title("Phylogeny from Newick string")
fig2.tight_layout()
