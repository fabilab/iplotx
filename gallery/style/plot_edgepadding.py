"""
Edge shrink
===========

This example illustrates how to shrink edges, i.e. leave a bit of empty space between
edge cap (end) and the border of its source/target vertices.
"""

import iplotx as ipx

graph = {
    "edges": [
        ("A", "B"),
        ("B", "C"),
        ("C", "D"),
        ("C", "E"),
        ("E", "Bingo"),
    ],
    "directed": True,
}
layout = {
    "A": (0, 0),
    "B": (1, 0),
    "C": (1, 1),
    "D": (-1, 1),
    "E": (-0.5, 2),
    "Bingo": (0, 3),
}

ipx.network(
    graph,
    layout,
    style="rededge",
    edge_shrink=5,
    vertex_labels=True,
    vertex_size={"A": 20, "B": 20, "C": 20, "D": 20, "E": 20, "Bingo": 50},
)
