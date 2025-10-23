"""
Tree layout algorithms.
"""

from typing import (
    Any,
)
from collections.abc import (
    Hashable,
    Callable,
)

from .rooted import (
    _horizontal_tree_layout,
    _vertical_tree_layout,
    _radial_tree_layout,
)
from .unrooted import (
    _equalangle_tree_layout,
    _daylight_tree_layout,
)


def compute_tree_layout(
    layout: str,
    orientation: str,
    root: Any,
    preorder_fun: Callable,
    postorder_fun: Callable,
    levelorder_fun: Callable,
    children_fun: Callable,
    branch_length_fun: Callable,
    leaves_fun: Callable,
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
    kwargs["levelorder_fun"] = levelorder_fun
    kwargs["children_fun"] = children_fun
    kwargs["orientation"] = orientation
    kwargs["branch_length_fun"] = branch_length_fun
    kwargs["leaves_fun"] = leaves_fun

    # Angular or not, the vertex layout is unchanged. Since we do not
    # currently compute an edge layout here, we can ignore the option.
    kwargs.pop("angular", None)

    if layout == "radial":
        layout_dict = _radial_tree_layout(**kwargs)
    elif layout == "horizontal":
        layout_dict = _horizontal_tree_layout(**kwargs)
    elif layout == "vertical":
        layout_dict = _vertical_tree_layout(**kwargs)
    elif layout == "equalangle":
        layout_dict = _equalangle_tree_layout(**kwargs)
    elif layout == "daylight":
        layout_dict = _daylight_tree_layout(**kwargs)
    else:
        raise ValueError(f"Tree layout not available: {layout}")

    return layout_dict
