from typing import (
    Optional,
    Sequence,
)
from collections.abc import Hashable
from operator import attrgetter
import numpy as np
import pandas as pd

from ....typing import (
    TreeType,
    LayoutType,
)
from ...typing import (
    TreeDataProvider,
    TreeData,
)
from ...heuristics import (
    normalise_tree_layout,
)


class Ete4DataProvider(TreeDataProvider):
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
        """Create tree data object for iplotx from ete4.core.tre.Tree classes."""

        root_fun = attrgetter("root")
        preorder_fun = lambda tree: tree.traverse("preorder")
        postorder_fun = lambda tree: tree.traverse("postorder")
        children_fun = attrgetter("children")
        branch_length_fun = lambda node: node.dist if node.dist is not None else 1.0
        leaves_fun = lambda tree: tree.leaves()

        tree_data = {
            "root": tree.root,
            "leaves": leaves_fun(tree),
            "rooted": True,
            "directed": directed,
            "ndim": 2,
            "layout_name": layout,
        }

        # Add vertex_df including layout
        tree_data["vertex_df"] = normalise_tree_layout(
            layout,
            tree=tree,
            orientation=orientation,
            root_fun=root_fun,
            preorder_fun=preorder_fun,
            postorder_fun=postorder_fun,
            children_fun=children_fun,
            branch_length_fun=branch_length_fun,
        )
        if layout in ("radial",):
            tree_data["layout_coordinate_system"] = "polar"
        else:
            tree_data["layout_coordinate_system"] = "cartesian"

        # Add edge_df
        edge_data = {"_ipx_source": [], "_ipx_target": []}
        for node in preorder_fun(tree):
            for child in children_fun(node):
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

    def check_dependencies(self) -> bool:
        try:
            from ete4 import Tree
        except ImportError:
            return False
        return True

    def tree_type(self):
        from ete4 import Tree

        return Tree
