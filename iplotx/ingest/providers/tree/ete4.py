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


class Ete4DataProvider(TreeDataProvider):
    def _traverse(self, order: str) -> Any:
        """Traverse the tree."""
        return self.tree.traverse(order)

    preorder = partialmethod(_traverse, order="preorder")
    postorder = partialmethod(_traverse, order="postorder")

    def get_leaves(self) -> Sequence[Any]:
        return self.tree.leaves()

    @staticmethod
    def get_children(node: Any) -> Sequence[Any]:
        return node.children

    @staticmethod
    def get_branch_length(node: Any) -> Optional[float]:
        return node.dist

    @staticmethod
    def check_dependencies() -> bool:
        return importlib.util.find_spec("ete4") is not None

    @staticmethod
    def tree_type():
        from ete4 import Tree

        return Tree
