"""
Typing module for data/object ingestion. This module described the abstract data types that providers need to comply with to be compatible with iplotx.

Networkx and trees are treated separately for practical reasons: many tree analysis libraries rely heavily on recursive data structures, which do not
work as well on general networks.
"""

from typing import (
    NotRequired,
    TypedDict,
    Protocol,
    Optional,
    Sequence,
    Any,
    Iterable,
)
from collections.abc import Hashable
import numpy as np
import pandas as pd
from ..typing import (
    GraphType,
    LayoutType,
    TreeType,
)
from .heuristics import (
    normalise_tree_layout,
)


class NetworkData(TypedDict):
    """Network data structure for iplotx."""

    directed: bool
    vertex_df: pd.DataFrame
    edge_df: pd.DataFrame
    ndim: int
    network_library: NotRequired[str]


class NetworkDataProvider(Protocol):
    """Protocol for network data ingestion provider for iplotx."""

    def __init__(
        self,
        network: GraphType,
    ) -> None:
        """Initialise network data provider.

        Parameters:
            network: The network to ingest.
        """
        self.network = network

    def __call__(
        self,
        layout: Optional[LayoutType] = None,
        vertex_labels: Optional[Sequence[str] | dict[Hashable, str] | pd.Series] = None,
        edge_labels: Optional[Sequence[str] | dict] = None,
    ) -> NetworkData:
        """Create network data object for iplotx from any provider."""
        raise NotImplementedError("Network data providers must implement this method.")

    @staticmethod
    def check_dependencies():
        """Check whether the dependencies for this provider are installed."""
        raise NotImplementedError("Network data providers must implement this method.")

    @staticmethod
    def graph_type():
        """Return the graph type from this provider to check for instances."""
        raise NotImplementedError("Network data providers must implement this method.")

    def is_directed(self):
        """Check whether the network is directed."""
        raise NotImplementedError("Network data providers must implement this method.")

    def number_of_vertices(self):
        """The number of vertices/nodes in the network."""
        raise NotImplementedError("Network data providers must implement this method.")


class TreeData(TypedDict):
    """Tree data structure for iplotx."""

    rooted: bool
    directed: bool | str
    root: Optional[Hashable]
    leaf_df: pd.DataFrame
    vertex_df: dict[Hashable, tuple[float, float]]
    edge_df: dict[Hashable, Sequence[tuple[float, float]]]
    layout_coordinate_system: str
    layout_name: str
    orientation: str
    ndim: int
    tree_library: NotRequired[str]


class TreeDataProvider(Protocol):
    """Protocol for tree data ingestion provider for iplotx."""

    def __init__(
        self,
        tree: TreeType,
    ) -> None:
        """Initialize the provider with the tree type.

        Parameters:
            tree: The tree type that this provider will handle.
        """
        self.tree = tree

    @staticmethod
    def check_dependencies():
        """Check whether the dependencies for this provider are installed."""
        raise NotImplementedError("Tree data providers must implement this method.")

    @staticmethod
    def tree_type():
        """Return the tree type from this provider to check for instances."""
        raise NotImplementedError("Tree data providers must implement this method.")

    def is_rooted(self) -> bool:
        """Get whether the tree is rooted.

        Returns:
            A boolean indicating whether the tree is rooted.

        Note: This is a default implemntation that can be overridden by the provider
        if they support unrooted trees (e.g. Biopython).
        """
        return True

    def get_root(self) -> Any:
        """Get the tree root in a provider-specific data structure.

        Returns:
            The root of the tree.

        Note: This is a default implemntation that can be overridden by the provider.
        """
        root_attr = self.tree.root
        if callable(root_attr):
            return root_attr()
        else:
            return root_attr

    def get_leaves(self) -> Sequence[Any]:
        """Get the tree leaves/tips in a provider-specific data structure.

        Returns:
            The leaves or tips of the tree.
        """
        raise NotImplementedError("Tree data providers must implement this method.")

    def preorder(self) -> Iterable[Any]:
        """Preorder (DFS - parent first) iteration over the tree.

        Returns:
            An iterable of nodes in preorder traversal.
        """
        raise NotImplementedError("Tree data providers must implement this method.")

    def postorder(self) -> Iterable[Any]:
        """Postorder (DFS - child first) iteration over the tree.

        Returns:
            An iterable of nodes in preorder traversal.
        """
        raise NotImplementedError("Tree data providers must implement this method.")

    @staticmethod
    def get_children(
        node: Any,
    ) -> Sequence[Any]:
        """Get the children of a node.

        Parameters:
            node: The node to get the children from.
        Returns:
            A sequence of children nodes.
        """
        raise NotImplementedError("Tree data providers must implement this method.")

    @staticmethod
    def get_branch_length(
        node: Any,
    ) -> Optional[float]:
        """Get the length of the branch to this node.

        Parameters:
            node: The node to get the branch length from.
        Returns:
            The branch length to the node.
        """
        raise NotImplementedError("Tree data providers must implement this method.")

    def get_branch_length_default_to_one(
        self,
        node: Any,
    ) -> float:
        """Get the length of the branch to this node, defaulting to 1.0 if not available.

        Parameters:
            node: The node to get the branch length from.
        Returns:
            The branch length to the node, defaulting to 1.0 if not available.
        """
        branch_length = self.get_branch_length(node)
        return branch_length if branch_length is not None else 1.0

    def __call__(
        self,
        layout: str | LayoutType,
        orientation: Optional[str],
        layout_style: Optional[dict[str, int | float | str]] = None,
        directed: bool | str = False,
        vertex_labels: Optional[
            Sequence[str] | dict[Hashable, str] | pd.Series | bool | str
        ] = None,
        edge_labels: Optional[Sequence[str] | dict] = None,
        leaf_labels: Optional[
            Sequence[str] | dict[Hashable, str] | pd.Series | bool
        ] = None,
    ) -> TreeData:
        """Create tree data object for iplotx from ete4.core.tre.Tree classes.

        NOTE: This function needs NOT be implemented by individual providers.
        """

        if layout_style is None:
            layout_style = {}

        if orientation is None:
            if layout == "horizontal":
                orientation = "right"
            elif layout == "vertical":
                orientation = "descending"
            elif layout == "radial":
                orientation = "clockwise"

        tree_data = {
            "root": self.get_root(),
            "rooted": self.is_rooted(),
            "directed": directed,
            "ndim": 2,
            "layout_name": layout,
            "orientation": orientation,
        }

        # Add vertex_df including layout
        tree_data["vertex_df"] = normalise_tree_layout(
            layout,
            orientation=orientation,
            root=tree_data["root"],
            preorder_fun=self.preorder,
            postorder_fun=self.postorder,
            children_fun=self.get_children,
            branch_length_fun=self.get_branch_length_default_to_one,
            **layout_style,
        )
        if layout in ("radial",):
            tree_data["layout_coordinate_system"] = "polar"
        else:
            tree_data["layout_coordinate_system"] = "cartesian"

        # Add edge_df
        edge_data = {"_ipx_source": [], "_ipx_target": []}
        for node in self.preorder():
            for child in self.get_children(node):
                if directed == "parent":
                    edge_data["_ipx_source"].append(child)
                    edge_data["_ipx_target"].append(node)
                else:
                    edge_data["_ipx_source"].append(node)
                    edge_data["_ipx_target"].append(child)
        edge_df = pd.DataFrame(edge_data)
        tree_data["edge_df"] = edge_df

        # Add leaf_df
        # NOTE: Sometimes (e.g. cogent3) the leaves convert into a pd.Index
        # in a strange way, whereby their name disappears upon printing the
        # index but is actually visible (and kept) when inspecting the
        # individual elements (leaves). Seems ok functionally, though a little
        # awkward visually during debugging.
        tree_data["leaf_df"] = pd.DataFrame(index=self.get_leaves())
        leaf_name_attrs = ("name",)

        # Add vertex labels
        # If only leaf labels are chosen (at vertex positions), make a dict and
        # feed into the normal logic
        if isinstance(vertex_labels, str) and (vertex_labels == "leaves"):
            vertex_labels = {}
            for leaf in tree_data["leaf_df"].index:
                for name_attr in leaf_name_attrs:
                    if hasattr(leaf, name_attr):
                        vertex_labels[leaf] = getattr(leaf, name_attr)
                        break

        if vertex_labels is None:
            vertex_labels = False
        if np.isscalar(vertex_labels) and vertex_labels:
            tree_data["vertex_df"]["label"] = [
                x.name for x in tree_data["vertex_df"].index
            ]
        elif not np.isscalar(vertex_labels):
            # If a dict-like object is passed, it can be incomplete (e.g. only the leaves):
            # we fill the rest with empty strings which are not going to show up in the plot.
            if isinstance(vertex_labels, pd.Series):
                vertex_labels = dict(vertex_labels)
            if isinstance(vertex_labels, dict):
                for vertex in tree_data["vertex_df"].index:
                    if vertex not in vertex_labels:
                        vertex_labels[vertex] = ""
            tree_data["vertex_df"]["label"] = pd.Series(vertex_labels)

        # Add leaf labels
        if leaf_labels is None:
            leaf_labels = False
        if np.isscalar(leaf_labels) and leaf_labels:
            leaf_labels = []
            for leaf in tree_data["leaf_df"].index:
                for name_attr in leaf_name_attrs:
                    if hasattr(leaf, name_attr):
                        label = getattr(leaf, name_attr)
                        break
                else:
                    raise ValueError(
                        "Could not find leaf name attribute.",
                    )
                leaf_labels.append(label)
            tree_data["leaf_df"]["label"] = leaf_labels
        elif not np.isscalar(leaf_labels):
            # Leaves are already in the dataframe in a certain order, so sequences are allowed
            if isinstance(leaf_labels, (list, tuple, np.ndarray)):
                leaf_labels = {
                    leaf: label
                    for leaf, label in zip(tree_data["leaf_df"].index, leaf_labels)
                }
            # If a dict-like object is passed, it can be incomplete (e.g. only the leaves):
            # we fill the rest with empty strings which are not going to show up in the plot.
            if isinstance(leaf_labels, pd.Series):
                leaf_labels = dict(leaf_labels)
            if isinstance(leaf_labels, dict):
                for leaf in tree_data["leaf_df"].index:
                    if leaf not in leaf_labels:
                        leaf_labels[leaf] = ""
            tree_data["leaf_df"]["label"] = pd.Series(leaf_labels)

        return tree_data
