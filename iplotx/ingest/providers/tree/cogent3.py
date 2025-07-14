from typing import (
    Any,
    Optional,
    Sequence,
)
import importlib
from ...typing import (
    TreeDataProvider,
)


class Cogent3DataProvider(TreeDataProvider):
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
        return importlib.util.find_spec("cogent3") is not None

    @staticmethod
    def tree_type():
        from cogent3.core.tree import PhyloNode

        return PhyloNode

    def get_support(self):
        """Get support values for all nodes."""
        support_dict = {}
        for node in self.preorder():
            support_dict[node] = node.params.get("support", None)
        return support_dict
