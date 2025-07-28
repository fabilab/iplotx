import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import patches
import iplotx as ipx


def vertex_collection(**kwargs):
    if "layout" not in kwargs:
        kwargs["layout"] = pd.DataFrame(
            data=np.array([[0, 0], [1, 1]], np.float64),
            index=["A", "B"],
        )
    return ipx.artists.VertexCollection(**kwargs)


def edge_collection(**kwargs):
    if vertex_collection not in kwargs:
        kwargs["vertex_collection"] = vertex_collection()
    if "vertex_ids" not in kwargs:
        vids = kwargs["vertex_collection"].get_index()
        nedges = 4
        kwargs["vertex_ids"] = [
            [vids[i % len(vids)], vids[(i + 1) % len(vids)]] for i in range(nedges)
        ]
        kwargs["patches"] = [
            patches.PathPatch(mpl.path.Path(np.array([[0, 0], [1, 1]]))),
        ] * len(kwargs["vertex_ids"])

    return ipx.artists.EdgeCollection(**kwargs)


def test_props():
    edges = edge_collection()
    edges.directed = True
    assert edges._directed is True
    assert edges.get_curved() is False


def test_curved():
    edges = edge_collection()
    edges.set_curved(True)


def test_split():
    with ipx.style.context("dashdepth"):
        style = ipx.style.get_style(".edge")
        edge_collection(style=style)


def test_ports():
    edges = edge_collection()
    edges.set_ports(None)
    edges.set_ports(("n", "w"))


def test_tension():
    edges = edge_collection()
    edges.set_tension(None)
    edges.set_tension(0.5)


def test_looptension():
    edges = edge_collection()
    edges.set_looptension(None)
    edges.set_looptension(0.5)


def test_loopmaxangle():
    edges = edge_collection()
    edges.set_loopmaxangle(45)


def test_offset():
    edges = edge_collection()
    edges.set_offset(None)
    edges.set_offset(2)


def test_arrow_marker():
    for marker in [
        "|>",
        "|/",
        "|\\",
        ">",
        "<",
        ">>",
        ")>",
        ")",
        "(",
        "]",
        "[",
        "|",
        "x",
        "s",
        "d",
        "p",
        "q",
    ]:
        with ipx.style.context(edge_arrow_marker=marker):
            edge_collection(directed=True, style=ipx.style.get_style(".edge"))
