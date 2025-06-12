import numpy as np
import pandas as pd
from .typing import TreeType


def compute_tree_layout(
    tree: TreeType,
    layout: str,
    orientation: str,
    **kwargs,
):

    if layout == "radial":
        layout_dict = _circular_tree_layout(tree, orientation=orientation, **kwargs)
    elif layout == "horizontal":
        layout_dict = _horizontal_tree_layout(tree, orientation=orientation, **kwargs)
    elif layout == "vertical":
        layout_dict = _vertical_tree_layout(tree, orientation=orientation, **kwargs)
    else:
        raise ValueError(f"Tree layout not available: {layout}")

    return layout_dict


def _horizontal_tree_layout_right(
    tree,
    root_fun: callable,
    preorder_fun: callable,
    postorder_fun: callable,
    children_fun: callable,
    branch_length_fun: callable,
):
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
    i = 0
    for node in postorder_fun(tree):
        children = children_fun(node)
        if len(children) == 0:
            layout[node] = [None, i]
            i += 1
        else:
            layout[node] = [
                None,
                np.mean([layout[child][1] for child in children]),
            ]

    # Set the x values for vertices
    layout[root_fun(tree)][0] = 0
    for node in preorder_fun(tree):
        x0, y0 = layout[node]
        for child in children_fun(node):
            bl = branch_length_fun(child)
            if bl is None:
                bl = 1.0
            layout[child][0] = layout[node][0] + bl

    return layout


def _horizontal_tree_layout(tree, orientation="right", **kwargs):
    if orientation not in ("right", "left"):
        raise ValueError("Orientation must be 'right' or 'left'.")

    layout = _horizontal_tree_layout_right(tree, **kwargs)

    if orientation == "left":
        for key, value in layout.items():
            layout[key].values[:] = value.values * np.array([-1, 1])
    return layout


def _vertical_tree_layout(tree, orientation="descending", **kwargs):
    """Vertical tree layout."""
    sign = 1 if orientation == "descending" else -1
    layout = _horizontal_tree_layout(tree, **kwargs)
    # FIXME: the data structure might end up differing
    for key, value in layout.items():
        value = value.values @ np.array([[0, sign], [-sign, 0]])
        if orientation == "descending":
            value = value * np.array([-1, 1])
        layout[key].values[:] = value
        # TODO: we might need to fix right/left if descending
    return layout


def _circular_tree_layout(
    tree,
    orientation="right",
    starting_angle=0,
    angular_span=360,
    **kwargs,
):
    """Circular tree layout."""
    # Short form
    th = starting_angle * np.pi / 180
    th_span = angular_span * np.pi / 180
    sign = 1 if orientation == "right" else -1

    layout = _horizontal_tree_layout_right(tree, **kwargs)
    ymax = max(point[1] for point in layout.values())
    for key, (x, y) in layout.items():
        r = x
        theta = sign * th_span * y / (ymax + 1) + th
        # We export r and theta to ensure theta does not
        # modulo 2pi if we take the tan and then arctan later.
        layout[key] = (r, theta)

    return layout
