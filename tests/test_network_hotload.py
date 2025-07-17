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


def test_hotload():
    edges = edge_collection()
    res = ipx.artists.NetworkArtist.from_edgecollection(
        edges,
    )
    assert isinstance(res, ipx.artists.NetworkArtist)

    res2 = ipx.artists.NetworkArtist.from_other(res)
    assert isinstance(res2, ipx.artists.NetworkArtist)
