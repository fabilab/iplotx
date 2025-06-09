from typing import (
    Optional,
    Sequence,
    Hashable,
)
import numpy as np
import pandas as pd
from Bio import Phylo

from ...typing import (
    TreeType,
    LayoutType,
)
from ..common import (
    TreeDataProvider,
    TreeData,
    _make_layout_columns,
)


class BiopythonDataProvider(TreeDataProvider):
    def __call__(
        self,
        tree: TreeType,
        layout: str | LayoutType,
    ) -> TreeData:
        """Create tree data object for iplotx from BioPython.Phylo.Tree classes."""

        tree_data = normalise_tree_layout(layout, tree=tree)
        tree_data.update(
            {
                "root": tree.root,
                "leaves": tree.get_terminals(),
                "rooted": tree.rooted,
            }
        )

        return tree_data
