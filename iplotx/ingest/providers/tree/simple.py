from typing import (
    Any,
    Optional,
    Sequence,
    Iterable,
    Self,
)

from ...typing import (
    TreeDataProvider,
)


class SimpleTree:
    """A simple tree class for educational purposes.

    Properties:
        children: Children SimpleTree objects.
        branch_length: Length of the branch leading to this node/tree.
    """

    children: Sequence[Self] = []
    branch_length: float = 1
    name: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a SimpleTree from a dictionary.

        Parameters:
            data: A dictionary representation of the tree, with "children" as a list offset_transform
                child nodes and an optional "branch_length" property (float).

        Returns:
            An instance of SimpleTree constructed from the provided dictionary.
        """
        tree = cls()
        tree.branch_length = data.get("branch_length", 1)
        tree.name = data.get("name", "")
        tree.children = [cls.from_dict(child) for child in data.get("children", [])]
        return tree


class SimpleTreeDataProvider(TreeDataProvider):
    def is_rooted(self) -> bool:
        return True

    def get_root(self) -> Any:
        """Get the root node of the tree."""
        return self.tree

    def preorder(self) -> Iterable[dict[dict | str, Any]]:
        def _recur(node):
            yield node
            for child in node.children:
                yield from _recur(child)

        yield from _recur(self.tree)

    def postorder(self) -> Iterable[dict[dict | str, Any]]:
        def _recur(node):
            for child in node.children:
                yield from _recur(child)
            yield node

        yield from _recur(self.tree)

    def get_leaves(self) -> Sequence[Any]:
        def _recur(node):
            if len(node.children) == 0:
                yield node
            else:
                for child in node.children:
                    yield from _recur(child)

        return list(_recur(self.tree))

    @staticmethod
    def get_children(node: Any) -> Sequence[Any]:
        return node.children

    @staticmethod
    def get_branch_length(node: Any) -> Optional[float]:
        return node.branch_length

    @staticmethod
    def check_dependencies() -> bool:
        return True

    @staticmethod
    def tree_type():
        return SimpleTree

    def get_support(self):
        """Get support/confidence values for all nodes."""
        support_dict = {}
        for node in self.preorder():
            support_dict[node] = None
        return support_dict
