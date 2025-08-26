from typing import (
    Sequence,
)
from math import atan2
import numpy as np
import matplotlib as mpl


# See also this link for the general answer (using scipy to compute coefficients):
# https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
def _evaluate_squared_bezier(points, t):
    """Evaluate a squared Bezier curve at t."""
    p0, p1, p2 = points
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def _evaluate_cubic_bezier(points, t):
    """Evaluate a cubic Bezier curve at t."""
    p0, p1, p2, p3 = points
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


def convex_hull(points):
    """Compute the convex hull of a set of 2D points.

    This is guaranteed to return the vertices clockwise.

    (Therefore, (v[i+1] - v[i]) rotated *left* by pi/2 points *outwards* of the convex hull.)
    """
    points = np.asarray(points)
    if len(points) < 3:
        return np.arange(len(points))

    hull_idx = None

    # igraph's should be faster in 2D
    try:
        import igraph

        hull_idx = igraph.convex_hull(list(points))
    except ImportError:
        pass

    # Otherwise, try scipy
    if hull_idx is None:
        try:
            from scipy.spatial import ConvexHull

            # NOTE: scipy guarantees counterclockwise ordering in 2D
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
            hull_idx = ConvexHull(points).vertices[::-1]
        except ImportError:
            pass

    # Last resort: our own Graham scan
    if hull_idx is None:
        hull_idx = _convex_hull_Graham_scan(points)

    return hull_idx


# see also: https://github.com/igraph/igraph/blob/075be76c92b99ca4c95ad9207bcc1af6d471c85e/src/misc/other.c#L116
# Compared to that C implementation, this is a bit more vectorised and messes less with memory as usual when
# optimising Python/numpy code
def _convex_hull_Graham_scan(points):
    """Compute the indices for the convex hull of a set of 2D points using Graham's scan algorithm.

    NOTE: This works from 3 points upwards, guaranteed clockwise.
    """
    points = np.asarray(points)

    # Find pivot (bottom left corner)
    miny_idx = np.flatnonzero(points[:, 1] == points[:, 1].min())
    pivot_idx = miny_idx[points[miny_idx, 0].argmin()]

    # Compute angles against that pivot, ensuring the pivot itself last
    angles = np.arctan2(points[:, 1] - points[pivot_idx, 1], points[:, 0] - points[pivot_idx, 0])
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
    hull: np.ndarray | Sequence[int],
    points: np.ndarray,
    transform: mpl.transforms.Transform,
    vertexpadding: int = 10,
    # TODO: check how dpi affects this
    dpi: float = 72.0,
) -> np.ndarray:
    """Offset path for a group based on vertex padding.

    Parameters:
        hull: The coordinates (not indices!) of the convex hull.
        points: This is the np.ndarray where the coordinates will be written to (output).
            The length is some integer ppc * len(hull) + 1 because for each vertex, this
            function wraps around it using a certain fixed ppc number of points, plus the
            final point for CLOSEPOLY.
        transform: The transform of the hull points.
        vertexpadding: The padding to apply to the vertices, in figure coordinates.
        dpi (WIP): The dpi of the figure renderer.

    Returns:
        None. The output is written to the `points` array in place. This ensures that the
        length of this array is unchanged, which is important to ensure that the vertices
        and SVG codes are in sync.
    """
    if len(hull) == 0:
        return

    # Short form for point per curve
    ppc = (len(points) - 1) // len(hull)
    assert len(points) % ppc == 1

    # No padding, set degenerate path
    if vertexpadding == 0:
        for j, point in enumerate(hull):
            points[ppc * j : ppc * (j + 1)] = point
        points[-1] = points[0]
        return points

    # Transform into figure coordinates
    trans = transform.transform
    trans_inv = transform.inverted().transform

    # Singleton: draw a circle around it
    if len(hull) == 1:
        # NOTE: linspace is double inclusive, which covers CLOSEPOLY
        thetas = np.linspace(
            -np.pi,
            np.pi,
            len(points),
        )
        # NOTE: dpi scaling might need to happen here
        perimeter = vertexpadding * np.vstack([np.cos(thetas), np.sin(thetas)]).T
        return trans_inv(trans(hull[0]) + perimeter)

    # Doublet: draw two semicircles
    if len(hull) == 2:
        # Unit vector connecting the two points
        dv = trans(hull[0]) - trans(hull[1])
        dv = dv / np.sqrt((dv**2).sum())

        # Draw a semicircle
        angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 30)
        vs = np.array([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        vs = vs.T.reshape((len(angles), 2, 2))
        vs = np.matmul(dv, vs)

        # NOTE: dpi scaling might need to happen here
        semicircle1 = vertexpadding * vs
        semicircle2 = vertexpadding * np.matmul(vs, -np.diag((1, 1)))

        # Put it together
        vs1 = trans_inv(trans(hull[0]) + semicircle1)
        vs2 = trans_inv(trans(hull[1]) + semicircle2)
        points[:ppc] = vs1
        points[ppc:-1] = vs2
        points[-1] = points[0]
        return points

    # At least three points, i.e. a nondegenerate convex hull
    nsides = len(hull)
    for i, point1 in enumerate(hull):
        point0 = hull[i - 1]
        point2 = hull[(i + 1) % nsides]

        # NOTE: this can be optimised by computing things once
        # unit vector to previous point
        dv0 = trans(point1) - trans(point0)
        dv0 = dv0 / np.sqrt((dv0**2).sum())

        # unit vector to next point
        dv2 = trans(point2) - trans(point1)
        dv2 = dv2 / np.sqrt((dv2**2).sum())

        # span the angles
        theta0 = atan2(dv0[1], dv0[0])
        theta2 = atan2(dv2[1], dv2[0])

        # The worst that can happen is that we go exactly backwards, i.e. theta2 == theta0 + np.pi
        # if it's more than that, we are on the inside of the convex hull due to the periodicity of atan2
        if theta2 - theta0 > np.pi:
            theta2 -= 2 * np.pi

        # angles is from the point of view of the first vector, dv0
        angles = np.linspace(theta0 + np.pi / 2, theta2 + np.pi / 2, ppc)
        vs = np.array([np.cos(angles), np.sin(angles)]).T

        # NOTE: dpi scaling might need to happen here
        chunkcircle = vertexpadding * vs

        vs1 = trans_inv(trans(point1) + chunkcircle)
        points[i * ppc : (i + 1) * ppc] = vs1

    points[-1] = points[0]
    return points
