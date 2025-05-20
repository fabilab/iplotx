from math import tan, atan2
import numpy as np


# See also this link for the general answer (using scipy to compute coefficients):
# https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
def _evaluate_squared_bezier(points, t):
    """Evaluate a squared Bezier curve at t."""
    p0, p1, p2 = points
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def _evaluate_cubic_bezier(points, t):
    """Evaluate a cubic Bezier curve at t."""
    p0, p1, p2, p3 = points
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )


def convex_hull(points):
    """Compute the convex hull of a set of 2D points."""
    from ..importing import igraph

    points = np.asarray(points)

    # igraph's should be faster in 2D
    if igraph is not None:
        hull_idx = igraph.convex_hull(list(points))
    else:
        try:
            from scipy.spatial import ConvexHull

            hull_idx = ConvexHull(points).vertices
        except ImportError:
            hull_idx = _convex_hull_Graham_scan(points)

    return hull_idx


# see also: https://github.com/igraph/igraph/blob/075be76c92b99ca4c95ad9207bcc1af6d471c85e/src/misc/other.c#L116
# Compared to that C implementation, this is a bit more vectorised and messes less with memory as usual when
# optimising Python/numpy code
def _convex_hull_Graham_scan(points):
    """Compute the indices for the convex hull of a set of 2D points using Graham's scan algorithm."""
    if len(points) < 4:
        # NOTE: for an exact triangle, this does not guarantee chirality. Should be ok anyway
        return np.arange(len(points))

    points = np.asarray(points)

    # Find pivot (bottom left corner)
    miny_idx = np.flatnonzero(points[:, 1] == points[:, 1].min())
    pivot_idx = miny_idx[points[miny_idx, 0].argmin()]

    # Compute angles against that pivot, ensuring the pivot itself last
    angles = np.arctan2(
        points[:, 1] - points[pivot_idx, 1], points[:, 0] - points[pivot_idx, 0]
    )
    angles[pivot_idx] = np.inf

    # Sort points by angle
    order = np.argsort(angles)

    # Whenever two points have the same angle, keep the furthest one from the pivot
    # whenever an index is discarded from "order", set it to -1
    j = 0
    last_idx = order[0]
    pivot_idx = order[-1]
    for i in range(1, len(order)):
        next_idx = order[i]
        if angles[last_idx] == angles[next_idx]:
            dlast = np.linalg.norm(points[last_idx] - points[pivot_idx])
            dnext = np.linalg.norm(points[next_idx] - points[pivot_idx])
            # Ignore the new point, it's inside
            if dlast > dnext:
                order[i] = -1
            # Ignore the old point, it's inside
            # The new one has a chance (depending on who comes next)
            else:
                order[j] = -1
                last_idx = next_idx
                j = i
        # New angle found: this point automatically gets a chance
        # (depending on who comes next). This also means that the
        # last point (last_idx before reassignment) will make it into
        # the hull
        else:
            last_idx = next_idx
            j += 1

    # Construct the hull from all indices that are not -1
    order = order[order != -1]
    jorder = len(order) - 1
    stack = []
    j = 0
    last_idx = -1
    before_last_idx = -1
    while jorder > -1:
        next_idx = order[jorder]

        # If doing a correct turn (right), add the point to the hull
        # if doing a wrong turn (left), backtrack and skip

        # At the beginning, assume it's a good turn to start collecting points
        if j < 2:
            cp = -1
        else:
            cp = (points[last_idx, 0] - points[before_last_idx, 0]) * (
                points[next_idx, 1] - points[before_last_idx, 1]
            ) - (points[next_idx, 0] - points[before_last_idx, 0]) * (
                points[last_idx, 1] - points[before_last_idx, 1]
            )

        # turning correctly or accumulating: add to the stack
        if cp < 0:
            jorder -= 1
            stack.append(next_idx)
            j += 1
            before_last_idx = last_idx
            last_idx = next_idx

            # wrong turn: backtrack, excise wrong point and move to next vertex
        else:
            del stack[-1]
            j -= 1
            last_idx = before_last_idx
            before_last_idx = stack[j - 2] if j >= 2 else -1

    stack = np.asarray(stack)

    return stack


def _compute_group_path_with_vertex_padding(
    points,
    transform,
    vertexpadding=10,
):
    """Offset path for a group based on vertex padding.

    At the input, the structure is [v1, v1, v1, ..., vn, vn, vn, v1]

    # NOTE: this would look better as a cubic Bezier, but ok for now.
    """

    # Transform into figure coordinates
    trans = transform.transform
    trans_inv = transform.inverted().transform
    points = trans(points)

    # Find the vertex centers, to recompute the offsets from scratch
    # Independent whether this is a first call or a later draw,
    # finding the vertex center can be done at once
    # 0.         .->.vcenter
    #  |         |  ^
    #  |         |  |
    # 1.--.2     .--.
    # singleton group
    if len(points) == 5:
        points[:] = 0.5 * (points[0] + points[2])
        points[0] -= np.array([0, 1]) * vertexpadding
        points[1] -= np.array([1, 0]) * 2 * vertexpadding
        points[2] += np.array([0, 1]) * vertexpadding
        points[3] += np.array([1, 0]) * 2 * vertexpadding

    else:
        # doublet group
        if len(points) == 7:
            points[:-1:3] = 0.5 * (points[:-1:3] + points[2:-1:3])
        # triangle+ groups
        else:
            points[:-1:3] = points[:-1:3] + points[2:-1:3] - points[1:-1:3]
        points[1:-1:3] = points[:-1:3]
        points[2:-1:3] = points[:-1:3]
        points[-1] = points[0]

        # Compute all shift vectors by diff, arctan2, then add 90 degrees, tan, norm
        # This maintains chirality
        # NOTE: the last point is just going back to the beginning, this
        # is a quirk or how mpl's closed paths work

        # Diff
        vpoints = points[:-1:3].copy()
        vpoints[0] -= points[-2]
        vpoints[1:] -= points[:-4:3]

        # Normalise vpoints to 1
        norm = np.sqrt((vpoints**2).sum(axis=1))
        vpoints_norm = (vpoints.T / norm).T

        # Rotate vpoints by 90 degrees
        vpads = vpoints_norm @ np.array([[0, 1], [-1, 0]])
        vpads_perm = np.zeros_like(vpads)
        vpads_perm[:-1] = vpads[1:]
        vpads_perm[-1] = vpads[0]

        # Shift the points
        points[:-1:3] += vpads * vertexpadding
        points[2:-1:3] += vpads_perm * vertexpadding
        if len(vpoints) == 2:
            points[1:-1:3] += vpoints_norm * 2 * vertexpadding
        else:
            points[1:-1:3] += (vpads + vpads_perm) * vertexpadding

    # mpl's quirky closed-path thing
    points[-1] = points[0]

    # Transform back to data coordinates
    points = trans_inv(points)

    return points
