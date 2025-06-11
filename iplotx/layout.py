import numpy as np
import pandas as pd
from .typing import TreeType


def compute_tree_layout(
    tree: TreeType,
    layout: str,
    orientation: str,
):

    if layout == "radial":
        layout_dict = _circular_tree_layout(tree, orientation=orientation)
    elif layout == "horizontal":
        layout_dict = _horizontal_tree_layout(tree, orientation=orientation)
    elif layout == "vertical":
        layout_dict = _vertical_tree_layout(tree, orientation=orientation)
    else:
        raise ValueError(f"Tree layout not available: {layout}")

    return layout_dict


def _horizontal_tree_layout_right(tree):
    """Build a tree layout horizontally, left to right.

    The strategy is the usual one:
    1. Compute the y values for the leaves, from 0 upwards.
    2. Compute the y values for the internal nodes, bubbling up (postorder).
    3. Set the x value for the root as 0.
    4. Compute the x value of all nodes, trickling down (BFS/preorder).
    5. Compute the edges from the end nodes.
    """
    layout = {}

    # Set the y values for vertices
    for i, leaf in enumerate(tree.get_terminals()):
        layout[leaf] = [None, i]
    for node in tree.get_nonterminals(order="postorder"):
        layout[node] = [
            None,
            np.mean([layout[child][1] for child in node.clades]),
        ]

    # Set the x values for vertices
    layout[tree.root][0] = 0
    for node in tree.find_clades(order="level"):
        x0, y0 = layout[node]
        for child in node.clades:
            bl = child.branch_length if child.branch_length is not None else 1.0
            layout[child][0] = layout[node][0] + bl

    return layout


def _horizontal_tree_layout(tree, orientation="right"):
    if orientation not in ("right", "left"):
        raise ValueError("Orientation must be 'right' or 'left'.")

    layout = _horizontal_tree_layout_right(tree)

    if orientation == "left":
        for key, value in layout.items():
            layout[key].values[:] = value.values * np.array([-1, 1])
    return layout


def _vertical_tree_layout(tree, orientation="descending"):
    """Vertical tree layout."""
    sign = 1 if orientation == "descending" else -1
    layout = _horizontal_tree_layout(tree)
    # FIXME: the data structure might end up differing
    for key, value in layout.items():
        value = value.values @ np.array([[0, sign], [-sign, 0]])
        if orientation == "descending":
            value = value * np.array([-1, 1])
        layout[key].values[:] = value
        # TODO: we might need to fix right/left if descending
    return layout


def _circular_tree_layout(tree, orientation="right", starting_angle=0):
    """Circular tree layout."""
    # Short form
    th = starting_angle
    sign = 1 if orientation == "right" else -1

    layout = _horizontal_tree_layout_right(tree)
    ymax = max(point[1] for point in layout.values())
    for key, (x, y) in layout.items():
        r = x
        theta = sign * 2 * np.pi * y / (ymax + 1) + th
        # We export r and theta to ensure theta does not
        # modulo 2pi if we take the tan and then arctan later.
        layout[key] = (r, theta)

    return layout
