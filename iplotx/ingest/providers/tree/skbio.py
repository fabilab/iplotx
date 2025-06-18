from typing import (
    Any,
    Optional,
    Sequence,
)
from ...typing import (
    TreeDataProvider,
)


class SkbioDataProvider(TreeDataProvider):
    def preorder(self) -> Sequence[Any]:
        return self.tree.preorder()

    def postorder(self) -> Sequence[Any]:
        return self.tree.postorder()

    def get_leaves(self) -> Sequence[Any]:
        return self.tree.tips()

    @staticmethod
    def get_children(node: Any) -> Sequence[Any]:
        return node.children

    @staticmethod
    def get_branch_length(node: Any) -> Optional[float]:
        return node.length

    @staticmethod
    def check_dependencies() -> bool:
        try:
            from skbio import TreeNode
        except ImportError:
            return False
        return True

    @staticmethod
    def tree_type():
        from skbio import TreeNode

        return TreeNode
