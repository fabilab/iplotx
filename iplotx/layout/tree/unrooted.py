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
            if orientation in ("left", "counterclockwise"):
                alpha = -alpha
            beta = start + alpha / 2

            props["layout"][child] = [
                cur_x + branch_length_fun(child) * np.cos(np.radians(beta)),
                cur_y + branch_length_fun(child) * np.sin(np.radians(beta)),
            ]
            # props["angle"][child] = -90 - beta * np.sign(beta - 180)
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
    max_iter: int = 15,
    dampening: float = 0.8,
    max_correction: float = 20.0,
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
        dampening: Dampening factor for angle adjustments. 1.0 means full adjustment.
            The number must be strictily positive (usually between 0 excluded and 1
            included).
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

    if len(layout) <= 2:
        return layout

    # Make all arrays for easier manipulation
    orig_class = next(iter(layout.values())).__class__
    for key, value in layout.items():
        layout[key] = np.asarray(value)

    all_leaves = list(leaves_fun(root))

    change_avg = 1.0
    for it in range(max_iter):
        change_sum = 0
        ninternal = 0
        parents = [None] + list(levelorder_fun())
        for parent in parents:
            children = children_fun(parent) if parent is not None else [root]
            for node in children:
                grandchildren = children_fun(node)
                # Exclude leaves, since they have no children subtrees
                # that can be adjusted. Exclude also passthrough nodes with
                # a single child, because they are rotating rigidly when their
                # parent does so or tells them to do so.
                if len(grandchildren) < 2:
                    continue
                if parent is None and len(children) < 3:
                    continue
                res = _apply_daylight_single_node(
                    node,
                    parent,
                    grandchildren,
                    all_leaves,
                    layout,
                    leaves_fun,
                    children_fun,
                    dampening,
                    max_correction,
                )
                change_sum += res
                ninternal += 1

        change_avg = change_sum / ninternal
        if change_avg < delta_angle_min:
            break

    # Make all lists again
    for key, value in layout.items():
        layout[key] = orig_class(value)

    return layout


def _apply_daylight_single_node(
    node: Any,
    parent: Any,
    children: list[Any],
    all_leaves: list[Any],
    layout: dict[Hashable, np.ndarray],
    leaves_fun: Callable,
    children_fun: Callable,
    dampening: float,
    max_correction: float,
) -> float:
    """Apply daylight adjustment to a single internal node.

    Parameters:
        node: The internal node to adjust.
    Returns:
        The total change in angle applied.

    NOTE: The layout is also changed in place.

    # Inspired from:
    # https://github.com/thomasp85/ggraph/blob/6c4ce81e460c50a16f9cd97e0b3a089f36901316/src/unrooted.cpp#L122
    """

    import os
    from builtins import print as _print

    DEBUG_DAYLIGHT = os.getenv("IPLOTX_DEBUG_DAYLIGHT", "0") == "1"

    print = _print if DEBUG_DAYLIGHT else lambda *a, **k: None

    # 1. Find daylight boundary leaves for each subtree. There are always at least two subtrees,
    # the first child and the parent (this function is not called for leaves, and the root hopefully
    # has at least two children).
    p0 = layout[node]
    bounds = {}

    print("node")
    print(node)
    print(float(p0[0]), float(p0[1]))

    # To find the parent side leaves, we take all leaves and skip the ones
    # downstream of this node
    leaves_below = leaves_fun(node)

    # Check the parent first if there is one
    if parent is not None:
        leaves_parent_subtree = [leaf for leaf in all_leaves if leaf not in leaves_below]
        vec1 = layout[parent] - p0
        print("parent side leaves:")
        print(parent)
        print(
            f"  node to parent vector: {vec1[0]:.2f}, {vec1[1]:.2f}, angle: {np.degrees(np.arctan2(vec1[1], vec1[0])):.2f}"
        )
        lower_angle, upper_angle = 2 * np.pi, -2 * np.pi
        for leaf in leaves_parent_subtree:
            vec2 = layout[leaf] - p0
            angle = -_clockwise_angle(vec1, vec2)
            print("  parent side leaf:")
            print(leaf)
            print(
                f"  node to leaf vector: {vec2[0]:.2f}, {vec2[1]:.2f}, angle: {np.degrees(np.arctan2(vec2[1], vec2[0])):.2f}"
            )
            print(f"  angle: {np.degrees(angle):.2f}")
            if angle < lower_angle:
                print("lowering lower angle")
                lower_angle = angle
                lower = leaf
            else:
                print("not lowering lower angle")
            if angle > upper_angle:
                print("raising upper angle")
                upper_angle = angle
                upper = leaf
            else:
                print("not raising upper angle")
        bounds[parent] = (lower, upper, lower_angle, upper_angle)

    # Repeat the exact same thing for each child rather than the parent
    print("subtree leaves:")
    for child in children:
        vec1 = layout[child] - p0
        print(
            f"  node to child vector: {vec1[0]:.2f}, {vec1[1]:.2f}, angle: {np.degrees(np.arctan2(vec1[1], vec1[0])):.2f}"
        )
        lower_angle, upper_angle = 2 * np.pi, -2 * np.pi

        for leaf in leaves_fun(child):
            vec2 = layout[leaf] - p0
            angle = -_clockwise_angle(vec1, vec2)
            print(
                f"  node to leaf vector: {vec2[0]:.2f}, {vec2[1]:.2f}, angle: {np.degrees(np.arctan2(vec2[1], vec2[0])):.2f}"
            )
            print(leaf)
            print(f"  angle: {np.degrees(angle):.2f}")
            if angle < lower_angle:
                print("lowering lower angle")
                lower_angle = angle
                lower = leaf
            else:
                print("not lowering lower angle")
            if angle > upper_angle:
                print("raising upper angle")
                upper_angle = angle
                upper = leaf
            else:
                print("not raising upper angle")
        bounds[child] = (lower, upper, lower_angle, upper_angle)

    print("final boundary leaves:")
    print(bounds)

    for subtree, bound in bounds.items():
        vec1 = layout[bound[0]] - p0
        vec2 = layout[bound[1]] - p0
        angle = -_clockwise_angle(vec1, vec2)
        print("subtree angles:")
        print(f"  lower {np.degrees(np.arctan2(vec1[1], vec1[0])):.2f}")
        print(f"  upper {np.degrees(np.arctan2(vec2[1], vec2[0])):.2f}")
        print(f"  angle {np.degrees(angle):.2f}")

    # 2. Compute daylight angles
    # NOTE: Since Python 3.6, python keys are ordered by insertion order.
    daylight = {}
    daylight_sum = 0.0
    # TODO: Mayvbe optimise this by avoiding creating all these lists
    prev_upper = [bound[1] for bound in bounds.values()]
    new_lower = [bound[0] for bound in bounds.values()]
    new_lower = new_lower[1:] + [new_lower[0]]  # cycle left
    subtrees = children + ([parent] if parent is not None else [])
    print("daylight computation:")
    for subtree, prev_upp, new_low in zip(subtrees, prev_upper, new_lower):
        vec1 = layout[prev_upp] - p0
        vec2 = layout[new_low] - p0
        angle = -_clockwise_angle(vec1, vec2)
        print("daylight angle:")
        print(f"  previous upper {np.degrees(np.arctan2(vec1[1], vec1[0])):.2f}")
        print(f"  new lower      {np.degrees(np.arctan2(vec2[1], vec2[0])):.2f}")
        print(f"  angle: {np.degrees(angle):.2f}")

        daylight_sum += angle
        daylight[subtree] = float(angle)

    print("final daylight")
    print(daylight)
    print(f"daylight sum: {daylight_sum:.2f}")

    # 3. Compute *excess* daylight, and correct it
    # NOTE: There seems to be this notion that you rotate a node by the accumulated daylight
    # correction (to fill the space on the left) plus its own correction (to fill the space on the right).
    # It reads funny and this is within a BFS iteration anyway so it's probably ok, but it's
    # curious.
    # NOTE: The average adjustment is divided by children + 1 (the parent) bc this is unrooted.
    daylight_avg = daylight_sum / len(daylight)
    print(f"Average daylight angle: {np.degrees(daylight_avg):.2f}")
    daylight_cum_corr = 0.0
    daylight_changes = 0
    for child in children:
        daylight_excess = daylight[child] - daylight_avg
        daylight_excess = np.clip(
            daylight_excess, np.radians(-max_correction), np.radians(max_correction)
        )
        print(f"Applying daylight excess to child {child}: {np.degrees(daylight_excess):.2f}")
        # daylight_cum_corr += daylight_avg - daylight[leaf]
        _rotate_subtree_around_point(
            leaf,
            children_fun,
            layout,
            p0,
            dampening * daylight_excess,
            # dampening * daylight_cum_corr,
            recur=True,
        )
        daylight_changes += abs(dampening * daylight_cum_corr)
    # __import__("ipdb").set_trace()

    # Caller wants degrees
    return np.degrees(daylight_changes / len(children))


# see: https://stackoverflow.com/questions/14066933/direct-way-of-computing-the-clockwise-angle-between-two-vectors
def _clockwise_angle(v1, v2):
    """Compute the anticlockwise angle between two 2D vectors.

    Parameters:
        v1: First vector.
        v2: Second vector.
    Returns:
        The angle in radians.
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    determinant = v1[0] * v2[1] - v1[1] * v2[0]
    return -np.arctan2(determinant, dot)


def _rotate_subtree_around_point(
    node,
    children_fun: Callable,
    layout: dict[Hashable, list[float]],
    pivot,
    angle,
    recur: bool = True,
):
    point = np.asarray(layout[node])
    pivot = np.asarray(pivot)
    layout[node] = _rotate_around_point(
        point,
        pivot,
        angle,
    )
    if not recur:
        return
    for child in children_fun(node):
        _rotate_subtree_around_point(
            child,
            children_fun,
            layout,
            pivot,
            angle,
        )


def _rotate_around_point(
    point,
    pivot,
    angle,
):
    """Rotate a point around a piviot by angle (in radians).

    Parameters:
        point: The point to rotate.
        pivot: The piviot point.
        angle: The angle in radians.
    Returns:
        The rotated point.
    """
    point = np.asarray(point)
    pivot = np.asarray(pivot)
    cos = np.cos(angle)
    sin = np.sin(angle)
    rot = np.array([[cos, sin], [-sin, cos]])
    return pivot + (point - pivot) @ rot
