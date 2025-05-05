from .heuristics import (
    normalise_layout,
    detect_directedness,
)

class NetworkPlot:
    def __init__(self, network, layout=None, ax=None):
        print("NetworkPlot initialized")
