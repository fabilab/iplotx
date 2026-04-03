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

This example adapts ``draw_scatter_tree`` from `hstrat-synthesis
<https://github.com/mmore500/hstrat-synthesis/blob/main/pylib/tree/_draw_scatter_tree.py>`_
to use phyloframe's native iplotx provider instead of dendropy.

"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import iplotx as ipx
from phyloframe.legacy import alifestd_to_iplotx_polars


def draw_scatter_tree(
    phylogeny_df,
    *,
    hue=None,
    size=None,
    style=None,
    c=None,
    ax=None,
    layout="vertical",
    scatter_shuffle=False,
    scatter_kws=None,
    tree_kws=None,
):
    """Draw a phylogenetic tree with a seaborn scatter overlay.

    Adapted from `hstrat-synthesis
    <https://github.com/mmore500/hstrat-synthesis/blob/main/pylib/tree/_draw_scatter_tree.py>`_.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        Alife-standard phylogeny with optional metadata columns.
    hue, size, style : str, optional
        Column names forwarded to :func:`seaborn.scatterplot`.
    c : str, sequence, or None
        Colour values; a column name or explicit array.
    ax : matplotlib.axes.Axes, optional
        Target axes.  Created if *None*.
    layout : str
        Tree layout forwarded to :func:`iplotx.tree`.
    scatter_shuffle : bool or int
        Shuffle scatter points to reduce overplotting.
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

    # Extract node positions — radial layouts need Cartesian offsets
    ipx_layout = tree_artist.get_layout()
    if layout == "radial":
        xs, ys = tree_artist.get_nodes().get_offsets().T
    else:
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

    # Optionally shuffle to reduce overplotting
    if scatter_shuffle:
        rng = np.random.RandomState(
            int(scatter_shuffle) if isinstance(scatter_shuffle, int)
            and not isinstance(scatter_shuffle, bool) else None,
        )
        plot_df = plot_df.iloc[rng.permutation(len(plot_df))]

    # Resolve colour argument
    if isinstance(c, str):
        c = plot_df[c].fillna("none").tolist()
    elif c is None:
        c = "none"

    sns.scatterplot(
        plot_df,
        x="__x__",
        y="__y__",
        hue=hue,
        size=size,
        style=style,
        c=c,
        ax=ax,
        **{"legend": False, "zorder": 5, **scatter_kws},
    )

    return ax


# %%
# Radial scatter tree
# --------------------
# A small vertebrate phylogeny displayed with the ``"radial"`` layout.

vertebrate_df = pl.DataFrame({
    "id":          [0, 1, 2, 3, 4, 5, 6],
    "ancestor_id": [0, 0, 0, 1, 1, 2, 2],
    "origin_time": [0, 2, 3, 5, 5, 6, 6],
    "taxon_label": ["", "", "", "Salmon", "Frog", "Parrot", "Cat"],
    "group":       ["", "", "", "fish", "amphibian", "bird", "mammal"],
})

fig, ax = plt.subplots(figsize=(6, 6))
draw_scatter_tree(
    vertebrate_df,
    hue="group",
    ax=ax,
    layout="radial",
    scatter_kws={
        "edgecolor": "white",
        "linewidth": 0.8,
        "s": 90,
        "legend": "brief",
        "palette": "Set2",
    },
    tree_kws={"leaf_labels": True, "aspect": 1},
)
ax.set_title("Vertebrate phylogeny (radial)")
fig.tight_layout()

# %%
# Vertical scatter tree
# ----------------------
# A larger primate phylogeny with scatter points coloured by diet and
# sized by body mass, using the default ``"vertical"`` layout.

primate_df = pl.DataFrame({
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

fig2, ax2 = plt.subplots(figsize=(8, 6))
draw_scatter_tree(
    primate_df,
    hue="diet",
    size="body_mass_kg",
    ax=ax2,
    layout="vertical",
    scatter_kws={
        "edgecolor": "black",
        "linewidth": 0.5,
        "legend": "brief",
        "palette": "dark",
    },
    tree_kws={"leaf_labels": True},
)
ax2.set_title("Primate phylogeny by diet and body mass")
fig2.tight_layout()
