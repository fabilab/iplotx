import numpy as np
import pandas as pd
import iplotx as ipx


def vertex_collection(**kwargs):
    if "layout" not in kwargs:
        kwargs["layout"] = pd.DataFrame(
            data=np.array([[0, 0], [1, 1]], np.float64),
            index=["A", "B"],
        )
    return ipx.artists.VertexCollection(**kwargs)


def edge_collection(**kwargs):
    patches = []
    if vertex_collection not in kwargs:
        kwargs["vertex_collection"] = vertex_collection()
    if "vertex_ids" not in vertex_collection:
        vids = kwargs["vertex_collection"].get_index()
        kwargs["vertex_ids"] = [
            [vids[i % len(vids)], vids[(i + 1) % len(vids)]] for i in range(len(patches))
        ]
    return ipx.artists.EdgeCollection(**kwargs)
