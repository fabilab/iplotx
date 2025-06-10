from typing import (
    Optional,
    Sequence,
    Hashable,
)
import numpy as np
import pandas as pd

from ...typing import (
    TreeType,
    LayoutType,
)
from ..common import (
    TreeDataProvider,
    TreeData,
    _make_layout_columns,
)
from ..heuristics import (
    normalise_tree_layout,
)


class BiopythonDataProvider(TreeDataProvider):
    def __call__(
        self,
        tree: TreeType,
        layout: str | LayoutType,
        orientation: str = "horizontal",
        directed: bool | str = False,
        vertex_labels: Optional[
            Sequence[str] | dict[Hashable, str] | pd.Series | bool
        ] = None,
        edge_labels: Optional[Sequence[str] | dict] = None,
    ) -> TreeData:
        """Create tree data object for iplotx from BioPython.Phylo.Tree classes."""

        tree_data = {
            "root": tree.root,
            "leaves": tree.get_terminals(),
            "rooted": tree.rooted,
            "directed": directed,
            "ndim": 2,
        }

        # Add vertex_df including layout
        tree_data["vertex_df"] = normalise_tree_layout(
            layout,
            tree=tree,
            orientation=orientation,
        )

        # Add edge_df
        edge_data = {"_ipx_source": [], "_ipx_target": []}
        for node in tree.find_clades(order="preorder"):
            for child in node.clades:
                if directed == "parent":
                    edge_data["_ipx_source"].append(child)
                    edge_data["_ipx_target"].append(node)
                else:
                    edge_data["_ipx_source"].append(node)
                    edge_data["_ipx_target"].append(child)
        edge_df = pd.DataFrame(edge_data)
        edge_df.set_index(["_ipx_source", "_ipx_target"], inplace=True, drop=False)
        tree_data["edge_df"] = edge_df

        # Add vertex labels
        if vertex_labels is None:
            vertex_labels = False
        if np.isscalar(vertex_labels) and vertex_labels:
            tree_data["vertex_df"]["label"] = [
                x.name for x in tree_data["vertices"].index
            ]
        elif not np.isscalar(vertex_labels):
            tree_data["vertex_df"]["label"] = vertex_labels

        return tree_data

    def check_dependencies(self) -> bool:
        try:
            from Bio import Phylo
        except ImportError:
            return False
        return True
