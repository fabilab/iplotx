"""
iplotx is a library for interactive plotting of networks and trees in
matplotlib.

It guarantees the visualisation will look exactly the same no matter what
library was used to construct the network.
"""

from .version import __version__
from .plotting import (
    network,
    graph,
    tree,
    plot,
)
import iplotx.artists as artists
import iplotx.style as style


__all__ = [
    "network",
    "tree",
    "plot",
    "graph",
    "artists",
    "style",
    "__version__",
]
