"""
Module for handling edge ports in iplotx.
"""

from collections.abc import Callable
import numpy as np

sq2 = np.sqrt(2) / 2

port_dict = {
    "s": (0, -1),
    "w": (-1, 0),
    "n": (0, 1),
    "e": (1, 0),
    "sw": (-sq2, -sq2),
    "nw": (-sq2, sq2),
    "ne": (sq2, sq2),
    "se": (sq2, -sq2),
}


def _get_port_unit_vector(
    portstring: str,
    trans_inv: Callable,
):
    """Get the tangent unit vector from a port string."""
    # The only tricky bit is if the port says e.g. north but the y axis is inverted, in which
    # case the port should go south. We can figure it out by checking the sign of the monotonic
    # trans_inv from figure to data coordinates.
    v12 = trans_inv(
        np.array(
            [
                [0, 0],
                [1, 1],
            ]
        )
    )
    invertx = v12[1, 0] - v12[0, 0] < 0
    inverty = v12[1, 1] - v12[0, 1] < 0

    if invertx:
        portstring = portstring.replace("w", "x").replace("e", "w").replace("x", "e")
    if inverty:
        portstring = portstring.replace("n", "x").replace("s", "n").replace("x", "s")

    if portstring not in port_dict:
        raise KeyError(f"Port not found: {portstring}")
    return np.array(port_dict[portstring])
