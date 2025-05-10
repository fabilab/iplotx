from math import pi
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
    dsplit_new = dsplit * nloops / (nloops - 1)
    dsplit2_new = deltas[idx_sort[-2]]
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

    # TODO: we should greedily iterate from this
