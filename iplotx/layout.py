"""
Layout functions, currently limited to trees.
"""

from typing import Any
from collections.abc import (
    Hashable,
    Callable,
)

import numpy as np


def compute_tree_layout(
    layout: str,
    orientation: str,
    root: Any,
    preorder_fun: Callable,
    postorder_fun: Callable,
    children_fun: Callable,
    branch_length_fun: Callable,
    **kwargs,
) -> dict[Hashable, list[float]]:
    """Compute the layout for a tree.

    Parameters:
        layout: The name of the layout, e.g. "horizontal", "vertical", or "radial".
        orientation: The orientation of the layout, e.g. "right", "left", "descending",
            "ascending", "clockwise", "anticlockwise".

    Returns:
        A layout dictionary with node positions.
    """
    kwargs["root"] = root
    kwargs["preorder_fun"] = preorder_fun
    kwargs["postorder_fun"] = postorder_fun
    kwargs["children_fun"] = children_fun
    kwargs["branch_length_fun"] = branch_length_fun
    kwargs["orientation"] = orientation

    # Angular or not, the vertex layout is unchanged. Since we do not
    # currently compute an edge layout here, we can ignore the option.
    kwargs.pop("angular", None)

    if layout == "radial":
        layout_dict = _radial_tree_layout(**kwargs)
    elif layout == "horizontal":
        layout_dict = _horizontal_tree_layout(**kwargs)
    elif layout == "vertical":
        layout_dict = _vertical_tree_layout(**kwargs)
    else:
        raise ValueError(f"Tree layout not available: {layout}")

    return layout_dict


def _horizontal_tree_layout_right(
    root: Any,
    preorder_fun: Callable,
    postorder_fun: Callable,
    children_fun: Callable,
    branch_length_fun: Callable,
) -> dict[Hashable, list[float]]:
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
    for node in postorder_fun():
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
    layout[root][0] = 0
    for node in preorder_fun():
        for child in children_fun(node):
            bl = branch_length_fun(child)
            if bl is None:
                bl = 1.0
            layout[child][0] = layout[node][0] + bl

    return layout


def _horizontal_tree_layout(
    orientation="right",
    **kwargs,
) -> dict[Hashable, list[float]]:
    """Horizontal tree layout."""
    if orientation not in ("right", "left"):
        raise ValueError("Orientation must be 'right' or 'left'.")

    layout = _horizontal_tree_layout_right(**kwargs)

    if orientation == "left":
        for key in layout:
            layout[key][0] *= -1
    return layout


def _vertical_tree_layout(
    orientation="descending",
    **kwargs,
) -> dict[Hashable, list[float]]:
    """Vertical tree layout."""
    sign = -1 if orientation == "descending" else 1
    layout = _horizontal_tree_layout(**kwargs)
    for key, value in layout.items():
        # Invert x and y
        layout[key] = value[::-1]
        # Orient vertically
        layout[key][1] *= sign
    return layout


def _radial_tree_layout(
    orientation: str = "right",
    start: float = 180,
    span: float = 360,
    **kwargs,
) -> dict[Hashable, tuple[float, float]]:
    """Radial tree layout.

    Parameters:
        orientation: Whether the layout fans out towards the right (clockwise) or left
            (anticlockwise).
        start: The starting angle in degrees, default is -180 (left).
        span: The angular span in degrees, default is 360 (full circle). When this is
            360, it leaves a small gap at the end to ensure the first and last leaf
            are not overlapping.
    Returns:
        A dictionary with the radial layout.
    """
    # Short form
    th = start * np.pi / 180
    th_span = span * np.pi / 180
    pad = int(span == 360)
    sign = -1 if orientation in ("right", "clockwise") else 1

    layout = _horizontal_tree_layout_right(**kwargs)
    ymax = max(point[1] for point in layout.values())
    for key, (x, y) in layout.items():
        r = x
        theta = sign * th_span * y / (ymax + pad) + th
        # We export r and theta to ensure theta does not
        # modulo 2pi if we take the tan and then arctan later.
        layout[key] = (r, theta)

    return layout
