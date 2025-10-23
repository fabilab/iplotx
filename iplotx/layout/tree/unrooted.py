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
            for child in children_fun(node):
                res = _apply_daylight_single_node(
                    child,
                    node,
                    layout,
                    leaves_fun,
                    children_fun,
                )
                change_sum += res
                ninternal += 1

        change_avg = change_sum / ninternal
        if change_avg < delta_angle_min:
            break

    return layout


def _apply_daylight_single_node(
    node: Any,
    parent: Any,
    layout: dict[Hashable, list[float]],
    leaves_fun: Callable,
    children_fun: Callable,
) -> float:
    """Apply daylight adjustment to a single internal node.

    Parameters:
        node: The internal node to adjust.
    Returns:
        The total change in angle applied.

    NOTE: The layout is also changed in place.
    """
    # Three steps:
    # 1. Find subtrees
    # 2. Find angles of subtrees
    # 3. Adjust angles of (rotate) subtrees
    # (return angular change)

    # https://github.com/thomasp85/ggraph/blob/6c4ce81e460c50a16f9cd97e0b3a089f36901316/src/unrooted.cpp#L122
    # This algo needs to look at all leaves, not only the subtree. In other words, because
    # it's an unrooted tree algo, it treats the node as a split and collects both leaves *below*
    # this node and *above* this node. The latter are just the leaves below all siblings of this node.
    leaves_below = list(leaves_fun(node))
    leaves_above = sum(
        (list(leaves_fun(sibling)) for sibling in children_fun(parent) if sibling != node),
        [],
    )
    leaves = leaves_below + leaves_above
    nl_below = len(leaves_below)

    children = children_fun(node)

    bounds_lower = {}
    bounds_upper = {}

    x0, y0 = layout[node]
    for il, leaf in enumerate(leaves):
        if il < nl_below:
            dv = layout[leaf][0] - x0, layout[leaf][1] - y0
        else:
            dv = layout[parent][0] - x0, layout[parent][1] - y0
        lower, upper = 0, 0
        lower_angle, upper_angle = 2 * np.pi, -2 * np.pi
        for leaf2 in ...:
            dv2 = layout[leaf2][0] - x0, layout[leaf2][1] - y0

            # Anticlockwise angle between dv and dv2
            angle = _angle_between_vectors(dv, dv2)

            if angle < lower_angle:
                lower_angle = angle
                lower = leaf2
            if angle > upper_angle:
                upper_angle = angle
                upper = leaf2

        bounds_lower[leaf] = lower
        bounds_upper[leaf] = upper

    daylight = {}
    daylight_sum = 0.0
    old_leaf = leaves[-1]
    for leaf in bounds_lower:
        dx, dy = layout[old_leaf][0] - x0, layout[old_leaf][1] - y0
        dx2, dy2 = layout[leaf][0] - x0, layout[leaf][1] - y0
        dot = dx * dx2 + dy * dy2
        determinant = dx * dy2 - dy * dx2
        angle = np.arctan2(determinant, dot)

        daylight[leaf] = angle
        daylight_sum += angle

        old_leaf = leaf

    daylight_avg = daylight_sum / len(leaves)
    deylight_cumsum = 0.0
    daylight_changes = 0
    for leaf in leaves:
        daylight_cumsum += daylight_avg - daylight[leaf]
        daylight_changes += abs(daylight_cumsum)
        layout[child] = _rotate_around_point(
            layout[child],
            daylight_cumsum,
            (x0, y0),
        )

    return daylight_changes / len(leaves)


# see: https://stackoverflow.com/questions/14066933/direct-way-of-computing-the-clockwise-angle-between-two-vectors
def _anticlockwise_angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    determinant = v1[0] * v2[1] - v1[1] * v2[0]
    return np.arctan2(determinant, dot)
