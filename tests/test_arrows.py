import pytest
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import patches
import matplotlib.pyplot as plt
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


def test_arrows_init():
    edges = edge_collection()
    res = ipx.artists.EdgeArrowCollection(
        edges,
    )
    assert isinstance(res, ipx.artists.EdgeArrowCollection)


def test_cmap():
    edges = edge_collection(
        style={
            "cmap": mpl.colormaps.get_cmap("viridis"),
            "norm": mpl.colors.Normalize(vmin=0, vmax=1),
        }
    )
    res = ipx.artists.EdgeArrowCollection(
        edges,
    )
    assert isinstance(res, ipx.artists.EdgeArrowCollection)


def test_properties():
    res = ipx.artists.EdgeArrowCollection(
        edge_collection(),
    )
    np.testing.assert_array_almost_equal(res.get_sizes(), np.array([10.4] * 4))
    np.testing.assert_array_almost_equal(res.get_sizes_dpi(), np.array([10.4] * 4))
    res.set_sizes(None)

    fig = plt.figure()
    try:
        res.set_figure(fig)
        assert res.get_figure() == fig
    finally:
        plt.close(fig)

    with pytest.raises(ValueError):
        res.set_array(None)

    res = ipx.artists.EdgeArrowCollection(
        edge_collection(),
    )
    res.set_colors(np.asarray([(0, 0, 1, 1)] * 4))


def test_make_arrow_patch():
    markers = [
        "|>",
        "|/",
        "|\\",
        ">",
        ">>",
        ")>",
        ")",
        "|",
        "s",
        "d",
        "p",
        "q",
        "none",
    ]
    for marker in markers:
        patch, size = ipx.edge.arrow.make_arrow_patch(
            marker,
            width=10,
            height="width",
            color="blue",
        )
        assert isinstance(patch, patches.Patch)
        assert size == 10

    with pytest.raises(ValueError):
        ipx.edge.arrow.make_arrow_patch(
            "unsupported_marker",
        )
