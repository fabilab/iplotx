from typing import (
    Optional,
    Sequence,
    Hashable,
)
import numpy as np
import pandas as pd

from ...typing import (
    GraphType,
    LayoutType,
)
from ..common import (
    TreeDataProvider,
    TreeData,
    _make_layout_columns,
)


class BioPythonDataProvider(TreeDataProvider):
    def __call__(
        self,
        tree: TreeType,
        layout: str | LayoutType,
    ) -> TreeData:
        """Create tree data object for iplotx from BioPython.Phylo.Tree classes."""

        root = tree.root
        leaves = tree.get_terminals()

        layout = normalise_tree_layout(layout, tree=tree)
