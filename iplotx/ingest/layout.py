import numpy as np
import pandas as pd
from .typing import TreeType


def compute_tree_layout(
    layout: str,
    tree: TreeType,
):

    if layout == "radial":
        return _circular_tree_layout(tree)
    if layout == "horizontal":
        return _horizontal_tree_layout(tree)
    if layout == "vertical":
        return _vertical_tree_layout(tree)


def _horizontal_tree_layout_right(tree):
    """Build a tree layout horizontally, left to right."""
    layout = {
        "vertices": [],
        "edges": [],
    }
    for i, leaf in enumerate(tree.get_terminals()):


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
