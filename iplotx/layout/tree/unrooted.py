"""
Unrooted tree layout for iplotx.
"""

from typing import (
    Any,
)
from collections.abc import (
    Hashable,
    Callable,
)
import numpy as np


def _equalangle_tree_layout(
    root: Any,
    preorder_fun: Callable,
    postorder_fun: Callable,
    children_fun: Callable,
    branch_length_fun: Callable,
    leaves_fun: Callable,
    orientation: str = "right",
    start: float = 180,
    span: float = 360,
    **kwargs,
) -> dict[Hashable, list[float]]:
    """Equal angle unrooted tree layout.

    Parameters:
        orientation: Whether the layout fans out towards the right (clockwise) or left
            (anticlockwise).
        start: The starting angle in degrees, default is -180 (left).
        span: The angular span in degrees, default is 360 (full circle). When this is
            360, it leaves a small gap at the end to ensure the first and last leaf
            are not overlapping.
    Returns:
        A dictionary with the layout.

    Reference: "Inferring Phylogenies" by Joseph Felsenstein, ggtree.
    """

    props = {
        "layout": {},
        "nleaves": {},
        "start": {},
        "end": {},
        "angle": {},
    }

    props["layout"][root] = [0.0, 0.0]
    props["start"][root] = 0.0
    props["end"][root] = 360.0
    props["angle"][root] = 0.0

    # Count the number of leaves in each subtree
    for node in postorder_fun():
        props["nleaves"][node] = sum(props["nleaves"][child] for child in children_fun(node)) or 1

    # Set the layout of everyone except the root
    # NOTE: In ggtree, it says "postorder", but I cannot quite imagine how that would work,
    # given that in postorder the root is visited last but it's also the only node about
    # which we know anything at this point.
    for node in preorder_fun():
        nleaves = props["nleaves"][node]
        children = children_fun(node)

        # Get current node props
        start = props["start"].get(node, 0)
        end = props["end"].get(node, 0)
        cur_x, cur_y = props["layout"].get(node, [0.0, 0.0])

        total_angle = end - start

        for child in children:
            nleaves_child = props["nleaves"][child]
            alpha = nleaves_child / nleaves * total_angle
            beta = start + alpha / 2

            props["layout"][child] = [
                cur_x + branch_length_fun(child) * np.cos(np.radians(beta)),
                cur_y + branch_length_fun(child) * np.sin(np.radians(beta)),
            ]
            props["angle"][child] = -90 - beta * np.sign(beta - 180)
            props["start"][child] = start
            props["end"][child] = start + alpha
            start += alpha

    # FIXME: figure out how to tell the caller about "angle"
    return props["layout"]


def _daylight_tree_layout(
    root: Any,
    preorder_fun: Callable,
    postorder_fun: Callable,
    levelorder_fun: Callable,
    children_fun: Callable,
    branch_length_fun: Callable,
    leaves_fun: Callable,
    orientation: str = "right",
    start: float = 180,
    span: float = 360,
    max_iter: int = 5,
    **kwargs,
) -> dict[Hashable, list[float]]:
    """Daylight unrooted tree layout.

    Parameters:
        orientation: Whether the layout fans out towards the right (clockwise) or left
            (anticlockwise).
        start: The starting angle in degrees, default is -180 (left).
        span: The angular span in degrees, default is 360 (full circle). When this is
            360, it leaves a small gap at the end to ensure the first and last leaf
            are not overlapping.
        max_iter: Maximum number of iterations to perform.
    Returns:
        A dictionary with the layout.

    Reference: "Inferring Phylogenies" by Joseph Felsenstein, ggtree.
    """

    delta_angle_min = 9.0

    layout = _equalangle_tree_layout(
        root,
        preorder_fun,
        postorder_fun,
        children_fun,
        branch_length_fun,
        leaves_fun,
        orientation,
        start,
        span,
        **kwargs,
    )

    change_avg = 1.0
    for it in range(max_iter):
        change_sum = 0
        ninternal = 0
        for node in levelorder_fun():
            # Ignore leaves
            if len(children_fun(node)) == 0:
                continue

            res = _apply_daylight_single_node(node, layout)
            change_sum += res
            ninternal += 1

        change_avg = change_sum / ninternal
        if change_avg < delta_angle_min:
            break

    return layout


def _apply_daylight_single_node(node: Any, layout: dict[Hashable, list[float]]) -> float:
    """Apply daylight adjustment to a single internal node.

    Parameters:
        node: The internal node to adjust.
    Returns:
        The total change in angle applied.

    NOTE: The layout is also changed in place.
    """
    change = 0.0

    # TODO: implement

    return change
