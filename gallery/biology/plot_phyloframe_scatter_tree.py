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
import polars as pl
import seaborn as sns
import iplotx as ipx
from phyloframe import legacy as pfl
from phyloframe.legacy import alifestd_to_iplotx_polars


def draw_scatter_tree(
    phylogeny_df,
    *,
    hue=None,
    size=None,
    style=None,
    ax=None,
    layout="vertical",
    scatter_kws=None,
    tree_kws=None,
):
    """Draw a phylogenetic tree with a seaborn scatter overlay.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        Alife-standard phylogeny with optional metadata columns.
    hue, size, style : str, optional
        Column names forwarded to :func:`seaborn.scatterplot`.
    ax : matplotlib.axes.Axes, optional
        Target axes. Created if *None*.
    layout : str
        Tree layout forwarded to :func:`iplotx.tree`.
    scatter_kws : dict, optional
        Extra keyword arguments for :func:`seaborn.scatterplot`.
    tree_kws : dict, optional
        Extra keyword arguments for :func:`iplotx.tree`.
    """
    if ax is None:
        ax = plt.gca()
    if scatter_kws is None:
        scatter_kws = {}
    if tree_kws is None:
        tree_kws = {}

    tree_artist = ipx.tree(
        alifestd_to_iplotx_polars(phylogeny_df),
        ax=ax,
        layout=layout,
        **{"margins": 0.0, "edge_linewidth": 1.5, **tree_kws},
    )

    # Read node positions from the tree layout
    ipx_layout = tree_artist.get_layout()
    xs, ys = ipx_layout.T.to_numpy()
    pos = {
        node.name: (x, y)
        for node, (x, y) in zip(ipx_layout.index, zip(xs, ys))
    }

    # Map positions back onto the DataFrame
    plot_df = phylogeny_df.with_columns(
        pl.col("id").cast(pl.Utf8).replace_strict(
            {k: v[0] for k, v in pos.items()}, default=None,
        ).cast(pl.Float64).alias("__x__"),
        pl.col("id").cast(pl.Utf8).replace_strict(
            {k: v[1] for k, v in pos.items()}, default=None,
        ).cast(pl.Float64).alias("__y__"),
    ).to_pandas()

    sns.scatterplot(
        plot_df,
        x="__x__",
        y="__y__",
        hue=hue,
        size=size,
        style=style,
        ax=ax,
        **{"legend": False, "zorder": 5, **scatter_kws},
    )

    return ax


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
    "diet":        ["", "", "", "", "", "", "", "omnivore", "omnivore",
                    "herbivore", "herbivore", "omnivore", "omnivore"],
})

# %%
# Draw the scatter tree
# ----------------------
# Call ``draw_scatter_tree`` with seaborn aesthetics mapped to metadata
# columns.  The function handles tree rendering, layout extraction, and
# the scatter overlay in one step.

fig, ax = plt.subplots(figsize=(8, 6))
draw_scatter_tree(
    phylogeny_df,
    hue="diet",
    size="body_mass_kg",
    ax=ax,
    layout="vertical",
    scatter_kws={"edgecolor": "black", "linewidth": 0.5, "legend": "brief"},
    tree_kws={"leaf_labels": True},
)
ax.set_title("Primate phylogeny by diet and body mass")
fig.tight_layout()

# %%
# Tree from a Newick string
# --------------------------
# Phyloframe can also parse Newick strings into alife-standard DataFrames
# via :func:`phyloframe.legacy.alifestd_from_newick`.

newick = "((Human:1,Chimp:1.2):3,(Mouse:2,(Zebrafish:3,Fugu:2.5):1.5):4);"
newick_df = pfl.alifestd_from_newick(newick).pipe(pl.from_pandas)

fig2, ax2 = plt.subplots(figsize=(8, 4))
draw_scatter_tree(
    newick_df,
    ax=ax2,
    layout="horizontal",
    tree_kws={"leaf_labels": True, "edge_linewidth": 2, "margins": (0.2, 0)},
)
ax2.set_title("Phylogeny from Newick string")
fig2.tight_layout()
