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
    portstring,
):
    """Get the tangent unit vector from a port string."""
    if portstring not in port_dict:
        raise KeyError(f"Port not found: {portstring}")
    return np.array(port_dict[portstring])
