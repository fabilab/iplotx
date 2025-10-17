import pytest
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import iplotx as ipx

from utils import image_comparison

gt = pytest.importorskip("graph_tool")


@pytest.fixture
def layout_small_ring():
    coords = [
        [1.015318095035966, 0.03435580194714975],
        [0.29010409851547664, 1.0184451153265959],
        [-0.8699239050738742, 0.6328259400443561],
        [-0.8616466426732888, -0.5895891303732176],
        [0.30349699041342515, -0.9594640169691343],
    ]
    return coords


@image_comparison(baseline_images=["graph_basic"], remove_text=True)
def test_basic(layout_small_ring):
    g = gt.Graph(directed=False)
    vertices = list(g.add_vertex(5))
    for i in range(5):
        g.add_edge(vertices[i], vertices[(i + 1) % 5])

    fig, ax = plt.subplots(figsize=(3, 3))
    ipx.graph(g, ax=ax, layout=layout_small_ring)


@image_comparison(baseline_images=["graph_directed"], remove_text=True)
def test_directed(layout_small_ring):
    g = gt.Graph(directed=True)
    vertices = list(g.add_vertex(5))
    for i in range(5):
        g.add_edge(vertices[i], vertices[(i + 1) % 5])

    fig, ax = plt.subplots(figsize=(3, 3))
    ipx.graph(g, ax=ax, layout=layout_small_ring)
