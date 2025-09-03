"""
Zero-dependency networks
========================

This example uses `iplotx`'s internal `SimpleNetworkDataProvider` to show how to visualise
networkx without using any external analysis library (e.g. `igraph`, `networkx`).
"""

import iplotx as ipx

network = {
    "edges": [
        ("alice", "bob"),
        ("bob", "jago"),
        ("alice", "jago"),
    ]
}
layout = {
    "alice": (0, 0),
    "bob": (1, 1),
    "jago": (1, 0),
}
ipx.plot(
    network,
    layout,
    vertex_labels=True,
    style="hollow",
)
