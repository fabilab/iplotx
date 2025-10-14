from typing import Optional, Sequence
from contextlib import nullcontext
import numpy as np
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt

from .typing import (
    GraphType,
    LayoutType,
    GroupingType,
    TreeType,
)
from .network import NetworkArtist
from .network.groups import GroupingCollection
from .tree import TreeArtist
from .style import context


def network(
    network: Optional[GraphType] = None,
    layout: Optional[LayoutType] = None,
    grouping: Optional[GroupingType] = None,
    vertex_labels: Optional[list | dict | pd.Series | bool] = None,
    node_labels: Optional[list | dict | pd.Series | bool] = None,
    edge_labels: Optional[Sequence] = None,
    ax: Optional[mpl.axes.Axes] = None,
    style: str | dict | Sequence[str | dict] = (),
    title: Optional[str] = None,
    aspect: Optional[str | float] = None,
    margins: float | tuple[float, float] | tuple[float, float, float] = 0,
    strip_axes: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    **kwargs,
) -> list[mpl.artist.Artist]:
    """Plot this network and/or vertex grouping using the specified layout.

    Parameters:
        network: The network to plot. Can be a networkx or igraph graph.
        layout: The layout to use for plotting. If None, a layout will be looked for in the
            network object and, if none is found, an exception is raised. Defaults to None.
        vertex_labels: The labels for the vertices. If None or False, no vertex labels
            will be drawn. If a list, the labels are taken from the list. If a dict, the keys
            should be the vertex IDs and the values should be the labels. If True (a single
            bool value), the vertex IDs will be used as labels.
        node_labels: Same meaning as vertex_labels. This is an alias to help users who prefer
            the word "node" over "vertex". If both vertex_labels and node_labels are specified,
            "node_labels" overrides "vertex_labels".
        edge_labels: The labels for the edges. If None, no edge labels will be drawn. Defaults
            to None.
        ax: The axis to plot on. If None, a new figure and axis will be created. Defaults to
            None.
        style: Apply this style for the objects to plot. This can be a sequence (e.g. list)
            of styles and they will be applied in order.
        title: If not None, set the axes title to this value.
        aspect: If not None, set the aspect ratio of the axis to this value. In 2D, the most
            common value is 1.0, which proportionates x- and y-axes. In 3D, only string
            values are accepted (see the documentation of Axes.set_aspect).
        margins: How much margin to leave around the plot. A higher value (e.g. 0.1) can be
            used as a quick fix when some vertex shapes reach beyond the plot edge. This is
            a fraction of the data limits, so 0.1 means 10% of the data limits will be left
            as margin. A pair (in 2D) or triplet (in 3D) of floats can also be provided and
            applied to each axis separately.
        strip_axes: If True, remove axis spines and ticks. In 3D, only ticks are removed.
        figsize: If ax is None, a new matplotlib Figure is created. This argument specifies
            the (width, height) dimension of the figure in inches. If ax is not None, this
            argument is ignored. If None, the default matplotlib figure size is used.
        kwargs: Additional arguments are treated as an alternate way to specify style. If
            both "style" and additional **kwargs are provided, they are both applied in that
            order (style, then **kwargs).

    Returns:
        A list of mpl.artist.Artist objects, set as a direct child of the matplotlib Axes.
        The list can have one or two elements, depending on whether you are requesting to
        plot a network, a grouping, or both.
    """
    # Equivalence of node_labels and vertex_labels
    if node_labels is not None:
        vertex_labels = node_labels
    del node_labels

    stylecontext = context(style, **kwargs) if style or kwargs else nullcontext()

    with stylecontext:
        if (network is None) and (grouping is None):
            raise ValueError("At least one of network or grouping must be provided.")

        artists = []
        if network is not None:
            nwkart = NetworkArtist(
                network,
                layout,
                vertex_labels=vertex_labels,
                edge_labels=edge_labels,
                transform=mpl.transforms.IdentityTransform(),
            )
            artists.append(nwkart)
            layout = nwkart.get_layout()
        else:
            nwkart = None

        if grouping is not None:
            grpart = GroupingCollection(
                grouping,
                layout,
                network=network,
            )
            layout = grpart.get_layout()
            artists.append(grpart)
        else:
            grpart = None

        if (nwkart is not None) or (grpart is not None):
            ndim = layout.shape[1]
        else:
            ndim = None

        if ax is None:
            if ndim == 3:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection="3d")
            else:
                fig, ax = plt.subplots(figsize=figsize)
                ndim = 2
        else:
            # Check that the expected axis projection is used (3d for 3d layouts)
            if ndim == 3:
                assert isinstance(ax, Axes3D)
            elif ndim == 2:
                # NOTE: technically we probably want it to be cartesian (not polar, etc.)
                # but let's be flexible for now and let that request bubble up from users
                assert not isinstance(ax, Axes3D)

        # This is used in 3D for autoscaling
        had_data = ax.has_data()

        if nwkart is not None:
            # Set the figure, which itself sets the dpi scale for vertices, edges,
            # arrows, etc. Now data limits can be computed correctly
            nwkart.set_offset_transform(ax.transData)
            ax.add_artist(nwkart)
            nwkart.axes = ax

        if grpart is not None:
            grpart.set_transform(ax.transData)
            ax.add_artist(grpart)
            grpart.ax = ax

        if title is not None:
            ax.set_title(title)

        if aspect is not None:
            ax.set_aspect(aspect)

        _postprocess_axes(ax, artists, strip=strip_axes, had_data=had_data)

        if np.isscalar(margins):
            margins = [margins] * ndim
        if (margins[0] != 0) or (margins[1] != 0) or ((len(margins) == 3) and (margins[2] != 0)):
            ax.margins(*margins)

        return artists


# Aliases
plot = network
graph = network


def tree(
    tree: Optional[TreeType] = None,
    layout: str | LayoutType = "horizontal",
    directed: bool | str = False,
    vertex_labels: Optional[list | dict | pd.Series | bool] = None,
    node_labels: Optional[list | dict | pd.Series | bool] = None,
    leaf_labels: Optional[list | dict | pd.Series | bool] = None,
    show_support: bool = False,
    ax: Optional[mpl.axes.Axes] = None,
    style: str | dict | Sequence[str | dict] = "tree",
    title: Optional[str] = None,
    aspect: Optional[str | float] = None,
    margins: float | tuple[float, float] = 0,
    strip_axes: bool = True,
    figsize: Optional[tuple[float, float]] = None,
    **kwargs,
) -> TreeArtist:
    """Plot a tree using the specified layout.

    Parameters:
        tree: The tree to plot. Can be a BioPython.Phylo.Tree object.
        layout: The layout to use for plotting.
        directed: If False, do not draw arrows.
        vertex_labels: The labels for the vertices. If None or False, no vertex labels. Also
            read leaf_labels for leaf nodes.
        node_labels: Same meaning as vertex_labels. This is an alias to help users who prefer
            the word "node" over "vertex". If both vertex_labels and node_labels are specified,
            "node_labels" overrides "vertex_labels".
        leaf_labels: The labels for the leaf nodes. If None or False, no leaf labels are used
            except if vertex_labels are specified for leaf nodes. This argument and the
            previous vertex_labels provide somewhat redundant functionality but have quite
            different default behaviours for distinct use cases. This argument is typically
            useful for labels that are specific to leaf nodes only (e.g. species in a
            phylogenetic tree), whereas vertex_labels is typically used for labels that apply
            to internal nodes too (e.g. branch support values). This redundancy is left on
            purpose to allow for maximal style flexibility.
        show_support: If True, show the support values for the nodes (assumed to be from 0 to 100,
            rounded to nearest integer). If both this parameter and vertex_labels are set,
            show_support takes precedence and hides the vertex labels.
        ax: The axis to plot on. If None, a new figure and axis will be created. Defaults to
            None.
        style: Apply this style for the objects to plot. This can be a sequence (e.g. list)
            of styles and they will be applied in order.
        title: If not None, set the axes title to this value.
        aspect: If not None, set the aspect ratio of the axis to this value. The most common
            value is 1.0, which proportionates x- and y-axes.
        margins: How much margin to leave around the plot. A higher value (e.g. 0.1) can be
            used as a quick fix when some vertex shapes reach beyond the plot edge. This is
            a fraction of the data limits, so 0.1 means 10% of the data limits will be left
            as margin.
        strip_axes: If True, remove axis spines and ticks.
        figsize: If ax is None, a new matplotlib Figure is created. This argument specifies
            the (width, height) dimension of the figure in inches. If ax is not None, this
            argument is ignored. If None, the default matplotlib figure size is used.
        kwargs: Additional arguments are treated as an alternate way to specify style. If
            both "style" and additional **kwargs are provided, they are both applied in that
            order (style, then **kwargs).

    Returns:
        A TreeArtist object, set as a direct child of the matplotlib Axes.
    """
    # Equivalence of node_labels and vertex_labels
    if node_labels is not None:
        vertex_labels = node_labels
    del node_labels

    stylecontext = context(style, **kwargs) if style or kwargs else nullcontext()

    with stylecontext:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        artist = TreeArtist(
            tree=tree,
            layout=layout,
            directed=directed,
            transform=mpl.transforms.IdentityTransform(),
            offset_transform=ax.transData,
            vertex_labels=vertex_labels,
            leaf_labels=leaf_labels,
            show_support=show_support,
        )
        ax.add_artist(artist)
        artist.set_figure(ax.figure)

        if title is not None:
            ax.set_title(title)

        if aspect is not None:
            ax.set_aspect(aspect)

        _postprocess_axes(ax, [artist], strip=strip_axes)

        if np.isscalar(margins):
            margins = (margins, margins)
        if (margins[0] != 0) or (margins[1] != 0):
            ax.margins(*margins)

    return artist


# INTERNAL ROUTINES
def _postprocess_axes(ax, artists, strip=True, had_data=None):
    """Postprocess axis after plotting."""

    if strip:
        if not isinstance(ax, Axes3D):
            # Despine
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        if isinstance(ax, Axes3D):
            ax.set_zticks([])

    # NOTE: bboxes appear to be not that well defined in 3D axes
    # instead, there is a dedicated function that is a little
    # pedestrian
    if isinstance(ax, Axes3D):
        for art in artists:
            XYZ = art.get_layout().values.T
            if ax._zmargin < 0.05 and XYZ[0].size > 0:
                ax.set_zmargin(0.05)
            ax.auto_scale_xyz(
                *XYZ,
                had_data=had_data,
            )
            # NOTE: breaking is not needed, worst case it will
            # autoscale twice (for network and grouping), which
            # is better, at this stage of development, than
            # trying to be too clever by doing the math outselves
    else:
        # Set new data limits
        bboxes = []
        for art in artists:
            bboxes.append(art.get_datalim(ax.transData))
        bbox = mpl.transforms.Bbox.union(bboxes)
        ax.update_datalim(bbox)

        # Autoscale for x/y axis limits
        ax.autoscale_view()
