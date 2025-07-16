import unittest
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import iplotx as ipx

from utils import image_comparison


class GraphTestRunner(unittest.TestCase):
    @property
    def layout_small_ring(self):
        coords = [
            [1.015318095035966, 0.03435580194714975],
            [0.29010409851547664, 1.0184451153265959],
            [-0.8699239050738742, 0.6328259400443561],
            [-0.8616466426732888, -0.5895891303732176],
            [0.30349699041342515, -0.9594640169691343],
        ]
        return coords

    def make_ring(self):
        g = {
            "edges": [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 0),
            ]
        }
        return g

    @image_comparison(baseline_images=["graph_basic"], remove_text=True)
    def test_basic(self):
        g = self.make_ring()
        fig, ax = plt.subplots(figsize=(3, 3))
        ipx.plot(g, ax=ax, layout=self.layout_small_ring)

    @image_comparison(baseline_images=["graph_directed"], remove_text=True)
    def test_directed(self):
        g = self.make_ring()
        g["directed"] = True
        fig, ax = plt.subplots(figsize=(3, 3))
        ipx.plot(g, ax=ax, layout=self.layout_small_ring)

    @image_comparison(baseline_images=["graph_labels"], remove_text=True)
    def test_labels(self):
        g = self.make_ring()
        fig, ax = plt.subplots(figsize=(3, 3))
        ipx.plot(
            network=g,
            ax=ax,
            layout=self.layout_small_ring,
            vertex_labels=["1", "2", "3", "4", "5"],
            style={
                "vertex": {
                    "size": 20,
                    "label": {
                        "color": "white",
                        "size": 10,
                    },
                }
            },
        )
