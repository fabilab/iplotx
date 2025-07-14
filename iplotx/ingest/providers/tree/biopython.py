from typing import (
    Any,
    Optional,
    Sequence,
)
import importlib
from functools import partialmethod

from ...typing import (
    TreeDataProvider,
)


class BiopythonDataProvider(TreeDataProvider):
    def is_rooted(self) -> bool:
        return self.tree.rooted

    def _traverse(self, order: str) -> Any:
        """Traverse the tree."""
        return self.tree.find_clades(order=order)

    preorder = partialmethod(_traverse, order="preorder")
    postorder = partialmethod(_traverse, order="postorder")

    def get_leaves(self) -> Sequence[Any]:
        return self.tree.get_terminals()

    @staticmethod
    def get_children(node: Any) -> Sequence[Any]:
        return node.clades

    @staticmethod
    def get_branch_length(node: Any) -> Optional[float]:
        return node.branch_length

    @staticmethod
    def check_dependencies() -> bool:
        return importlib.util.find_spec("Bio") is not None

    @staticmethod
    def tree_type():
        from Bio import Phylo

        return Phylo.BaseTree.Tree

    def get_support(self):
        """Get support/confidence values for all nodes."""
        support_dict = {}
        for node in self.preorder():
            if hasattr(node, "confidences"):
                support = node.confidences
            elif hasattr(node, "confidence"):
                support = node.confidence
            else:
                support = None
            support_dict[node] = support
        return support_dict
