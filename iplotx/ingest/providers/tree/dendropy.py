from typing import (
    Any,
    Optional,
    Sequence,
)
import importlib

from ...typing import (
    TreeDataProvider,
)


class DendropyDataProvider(TreeDataProvider):
    def is_rooted(self) -> bool:
        return True

    def get_root(self) -> Any:
        """Get the root of the tree."""
        return next(self.preorder())

    def preorder(self) -> Any:
        """Preorder traversal of the tree.

        NOTE: This will work on both entire Trees and Nodes (which means a subtree including self).
        """
        if hasattr(self.tree, "preorder_node_iter"):
            return self.tree.preorder_node_iter()
        return self.tree.preorder_iter()

    def postorder(self) -> Any:
        """Preorder traversal of the tree.

        NOTE: This will work on both entire Trees and Nodes (which means a subtree including self).
        """
        if hasattr(self.tree, "postorder_node_iter"):
            return self.tree.postorder_node_iter()
        return self.tree.postorder_iter()

    def get_leaves(self) -> Sequence[Any]:
        """Get a list of leaves."""
        return self.tree.leaf_nodes()

    @staticmethod
    def get_children(node: Any) -> Sequence[Any]:
        return node.child_nodes()

    @staticmethod
    def get_branch_length(node: Any) -> Optional[float]:
        return node.edge.length

    @staticmethod
    def check_dependencies() -> bool:
        return importlib.util.find_spec("dendropy") is not None

    @staticmethod
    def tree_type():
        import dendropy

        return dendropy.Tree
