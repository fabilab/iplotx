import numpy as np
import pandas as pd
from .typing import TreeType


def compute_tree_layout(
    layout: str,
    tree: TreeType,
):

    if layout == "radial":
        layout_dict = _circular_tree_layout(tree)
    elif layout == "horizontal":
        layout_dict = _horizontal_tree_layout(tree)
    elif layout == "vertical":
        layout_dict = _vertical_tree_layout(tree)
    else:
        raise ValueError(f"Tree layout not available: {layout}")

    layout_dict["name"] = layout
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
    layout = {
        "vertices": {},
        "edges": {},
    }

    # Set the y values for vertices
    for i, leaf in tree.get_terminals():
        layout["vertices"][leaf] = (None, i)
    for node in tree.get_nonterminals():
        layout["vertices"][node] = (
            None,
            np.mean([layout["vertices"][child][1] for child in node.clades]),
        )

    # Set the x values for vertices and the edge values
    layout["vertices"][tree.root][0] = 0
    for node in tree.find_clades(order="level"):
        x0, y0 = layout["vertices"][node]
        for child in node.clades:
            bl = child.branch_length if child.branch_length is not None else 1.0
            layout["vertices"][child][0] = layout["vertices"][node][0] + bl

            # Edges can be referred to by their end vertex since that's the property of (rooted) trees
            x1, y1 = layout["vertices"][child]
            layout["edges"][child] = [
                (x0, y0),
                (x0, y1),
                (x1, y1),
            ]

    return layout


def _horizontal_tree_layout(tree, direction="right"):
    if direction not in ("right", "left"):
        raise ValueError("Direction must be 'right' or 'left'.")

    layout = _horizontal_tree_layout_right(tree)

    if direction == "left":
        for key, value in layout.items():
            layout[key].values[:] = value.values * np.array([-1, 1])
    return layout


def _vertical_tree_layout(tree, direction="descending"):
    """Vertical tree layout."""
    sign = 1 if direction == "descending" else -1
    layout = _horizontal_tree_layout(tree)
    # FIXME: the data structure might end up differing
    for key, value in layout.items():
        value = value.values @ np.array([[0, sign], [-sign, 0]])
        if direction == "descending":
            value = value * np.array([-1, 1])
        layout[key].values[:] = value
        # TODO: we might need to fix right/left if descending
    return layout


def _circular_tree_layout(tree, direction="clockwise", starting_angle=0):
    """Circular tree layout."""
    # Short form
    th = starting_angle

    layout = _horizontal_tree_layout_right(tree)
    for key, (x, y) in layout["vertices"].items():
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) + th
        layout["vertices"][key] = (r * np.cos(theta), r * np.sin(theta))
    for key, coords in layout["edges"].items():
        for i, (x, y) in enumerate(coords):
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x) + th
            layout["edges"][key][i] = (r * np.cos(theta), r * np.sin(theta))

    return layout
