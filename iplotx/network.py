import matplotlib as mpl

from .heuristics import (
    normalise_layout,
    detect_directedness,
)
from .tools.matplotlib import (
    _stale_wrapper,
    _forwarder,
    _set_additional_methods,
)



@_forwarder(
    (
        "set_clip_path",
        "set_clip_box",
        "set_transform",
        "set_snap",
        "set_sketch_params",
        "set_figure",
        "set_animated",
        "set_picker",
    )
)
class NetworkArtist(mpl.artist.Artist):
    def __init__(self, network, layout=None):
        super().__init__()

        self.network = network
        self._internal_network_data = _create_internal_data(network, layout)

        print("NetworkArtist initialized")



# INTERNAL ROUTINES
def _create_internal_data(network, layout=None):
    """Create internal data for the network."""
    layout = normalise_layout(network)
    directed = detect_directedness(network)


    return {
        "layout": layout,
        "directed": directed,
        "network": network,
    }
