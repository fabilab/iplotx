"""
Directed Acyclic Graph
======================

This example demonstrates how to create a random directed acyclic graph (DAG) and how to use `igraph` layouts based on strings.
"""

import random
import igraph as ig
import iplotx as ipx


# First, we set a random seed for reproducibility.
random.seed(0)

# We generate a random undirected graph with a fixed number of edges, without loops.
g = ig.Graph.Erdos_Renyi(n=15, m=30, directed=False, loops=False)

# Then we convert it to a DAG *in place* and replot it:
g.to_directed(mode="acyclic")

ipx.plot(
    g,
    layout="sugiyama",
    style={
        "vertex": {
            "size": 15,
            "facecolor": "grey",
            "edgecolor": "black",
        },
        "edge": {
            "color": "#222",
            "linewidth": 1,
        },
    },
)
