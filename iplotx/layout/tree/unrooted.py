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

    # Inspired from:
    # https://github.com/thomasp85/ggraph/blob/6c4ce81e460c50a16f9cd97e0b3a089f36901316/src/unrooted.cpp#L122
    # This algo needs to look at all leaves, not only the subtree. In other words, because
    # it's an unrooted tree algo, it treats the node as a split and collects both leaves *below*
    # this node and *above* this node. The latter are just the leaves below all siblings of this node.
    leaves_above = sum(
        (list(leaves_fun(sibling)) for sibling in children_fun(parent) if sibling != node),
        [],
    )

    children = children_fun(node)

    # 1. Find boundary leaves for each child and for the parent
    x0, y0 = layout[node]
    bounds_lower = {}
    bounds_upper = {}

    # Check the parent first...
    vec1 = layout[parent][0] - x0, layout[parent][1] - y0
    lower, upper = 0, 0
    lower_angle, upper_angle = 2 * np.pi, -2 * np.pi
    for leaf in leaves_above:
        vec2 = layout[leaf][0] - x0, layout[leaf][1] - y0
        angle = _anticlockwise_angle(vec1, vec2)
        if angle < lower_angle:
            lower_angle = angle
            lower = leaf
        if angle > upper_angle:
            upper_angle = angle
            upper = leaf
    bounds_lower[parent] = lower
    bounds_upper[parent] = upper

    # ... and now repeat the exact same thing for each child rather than the parent
    for child in children:
        vec1 = layout[child][0] - x0, layout[child][1] - y0
        lower, upper = 0, 0
        lower_angle, upper_angle = 2 * np.pi, -2 * np.pi

        for leaf in leaves_fun(child):
            vec2 = layout[leaf][0] - x0, layout[leaf][1] - y0
            angle = _anticlockwise_angle(vec1, vec2)
            if angle < lower_angle:
                lower_angle = angle
                lower = leaf
            if angle > upper_angle:
                upper_angle = angle
                upper = leaf
        bounds_lower[child] = lower
        bounds_upper[child] = upper

    # 2. Compute daylight angles
    # NOTE: Since Python 3.6, python keys are ordered by insertion order.
    daylight = {}
    daylight_sum = 0.0
    # TODO: Mayvbe optimise this by avoiding creating all these lists
    prev_leaves = list(bounds_upper.keys())
    leaves = list(bounds_lower.keys())[1:] + [parent]
    for prev_leaf, leaf in zip(prev_leaves, leaves):
        vec1 = layout[prev_leaf][0] - x0, layout[prev_leaf][1] - y0
        vec2 = layout[leaf][0] - x0, layout[leaf][1] - y0
        angle = _anticlockwise_angle(vec1, vec2)
        daylight[leaf] = angle
        daylight_sum += angle

    # 3. Compute *excess* daylight, and correct it
    # NOTE: There seems to be this notion that you rotate a node by the accumulated daylight
    # correction (to fill the space on the left) plus its own correction (to fill the space on the right).
    # It reads funny and this is within a BFS iteration anyway so it's probably ok, but it's
    # curious.
    daylight_avg = daylight_sum / (len(children) + 1)
    daylight_cum_corr = 0.0
    daylight_changes = 0
    for child in children:
        daylight_cum_corr += daylight_avg - daylight[leaf]
        layout[child] = _rotate_around_point(
            layout[child],
            (x0, y0),
            daylight_cum_corr,
        )
        daylight_changes += abs(daylight_cum_corr)

    return daylight_changes / len(leaves)


# see: https://stackoverflow.com/questions/14066933/direct-way-of-computing-the-clockwise-angle-between-two-vectors
def _anticlockwise_angle(v1, v2):
    """Compute the anticlockwise angle between two 2D vectors.

    Parameters:
        v1: First vector.
        v2: Second vector.
    Returns:
        The angle in radians.
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    determinant = v1[0] * v2[1] - v1[1] * v2[0]
    return np.arctan2(determinant, dot)


def _rotate_around_point(
    point,
    piviot,
    angle,
):
    """Rotate a point around a piviot by angle (in radians).

    Parameters:
        point: The point to rotate.
        piviot: The piviot point.
        angle: The angle in radians.
    Returns:
        The rotated point.
    """
    vec = point[0] - piviot[0], point[1] - piviot[1]
    cos = np.cos(angle)
    sin = np.sin(angle)

    x = piviot[0] + vec[0] * cos - vec[1] * sin
    y = piviot[1] + vec[0] * sin + vec[1] * cos
    return [x, y]
