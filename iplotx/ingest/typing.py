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

    def __call__(
        self,
        network: GraphType,
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


class TreeData(TypedDict):
    """Tree data structure for iplotx."""

    rooted: bool
    directed: bool | str
    root: Optional[Hashable]
    leaves: list[Hashable]
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
        orientation: str = "horizontal",
        directed: bool | str = False,
        vertex_labels: Optional[
            Sequence[str] | dict[Hashable, str] | pd.Series | bool
        ] = None,
        edge_labels: Optional[Sequence[str] | dict] = None,
    ) -> TreeData:
        """Create tree data object for iplotx from ete4.core.tre.Tree classes."""

        tree_data = {
            "root": self.get_root(),
            "leaves": self.get_leaves(),
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

        # Add vertex labels
        if vertex_labels is None:
            vertex_labels = False
        if np.isscalar(vertex_labels) and vertex_labels:
            tree_data["vertex_df"]["label"] = [
                x.name for x in tree_data["vertices"].index
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

        return tree_data
