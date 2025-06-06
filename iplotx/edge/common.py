from math import pi, atan2, tan
import numpy as np


def _compute_loops_per_angle(nloops, angles):
    if len(angles) == 0:
        return [(0, 2 * pi, nloops)]

    angles_sorted_closed = list(sorted(angles))
    angles_sorted_closed.append(angles_sorted_closed[0] + 2 * pi)
    deltas = np.diff(angles_sorted_closed)

    # Now we have the deltas and the total number of loops
    # 1. Assign all loops to the largest wedge
    idx_dmax = deltas.argmax()
    if nloops == 1:
        return [
            (angles_sorted_closed[idx_dmax], angles_sorted_closed[idx_dmax + 1], nloops)
        ]

    # 2. Check if any other wedges are larger than this
    # If not, we are done (this is the algo in igraph)
    dsplit = deltas[idx_dmax] / nloops
    if (deltas > dsplit).sum() < 2:
        return [
            (angles_sorted_closed[idx_dmax], angles_sorted_closed[idx_dmax + 1], nloops)
        ]

    # 3. Check how small the second-largest wedge would become
    idx_dsort = np.argsort(deltas)
    return [
        (
            angles_sorted_closed[idx_dmax],
            angles_sorted_closed[idx_dmax + 1],
            nloops - 1,
        ),
        (
            angles_sorted_closed[idx_dsort[-2]],
            angles_sorted_closed[idx_dsort[-2] + 1],
            1,
        ),
    ]


def _get_shorter_edge_coords(vpath, vsize, theta):
    # Bound theta from -pi to pi (why is that not guaranteed?)
    theta = (theta + pi) % (2 * pi) - pi

    # Size zero vertices need no shortening
    if vsize == 0:
        return np.array([0, 0])

    for i in range(len(vpath)):
        v1 = vpath.vertices[i]
        v2 = vpath.vertices[(i + 1) % len(vpath)]
        theta1 = atan2(*((v1)[::-1]))
        theta2 = atan2(*((v2)[::-1]))

        # atan2 ranges ]-3.14, 3.14]
        # so it can be that theta1 is -3 and theta2 is +3
        # therefore we need two separate cases, one that cuts at pi and one at 0
        cond1 = theta1 <= theta <= theta2
        cond2 = (
            (theta1 + 2 * pi) % (2 * pi)
            <= (theta + 2 * pi) % (2 * pi)
            <= (theta2 + 2 * pi) % (2 * pi)
        )
        if cond1 or cond2:
            break
    else:
        raise ValueError("Angle for patch not found")

    # The edge meets the patch of the vertex on the v1-v2 size,
    # at angle theta from the center
    mtheta = tan(theta)
    if v2[0] == v1[0]:
        xe = v1[0]
    else:
        m12 = (v2[1] - v1[1]) / (v2[0] - v1[0])
        xe = (v1[1] - m12 * v1[0]) / (mtheta - m12)
    ye = mtheta * xe
    ve = np.array([xe, ye])
    return ve * vsize
