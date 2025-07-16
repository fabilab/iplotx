import unittest
import pytest
import numpy as np
import pandas as pd
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

    def test_with_nodes(self):
        g = self.make_ring()
        g["nodes"] = [0, 1, 2, 3, 4]
        ipx.ingest.ingest_network_data(
            g,
        )

    def test_with_False_labels(self):
        g = self.make_ring()
        ipx.ingest.ingest_network_data(
            g,
            vertex_labels=False,
        )

    def test_with_True_labels(self):
        g = self.make_ring()
        ipx.ingest.ingest_network_data(
            g,
            vertex_labels=True,
        )

    def test_with_invalid_labels(self):
        g = self.make_ring()
        with pytest.raises(ValueError):
            ipx.ingest.ingest_network_data(
                g,
                vertex_labels=["x", "y"],
            )

    def test_with_no_edges(self):
        g = {
            "nodes": [0, 1],
        }
        ipx.ingest.ingest_network_data(
            g,
        )

    def test_with_layout_dataframe(self):
        g = self.make_ring()
        ipx.ingest.ingest_network_data(
            g,
            layout=pd.DataFrame(
                np.zeros((5, 2), np.float64),
            ),
        )

    def test_with_layout_dict(self):
        g = self.make_ring()
        ipx.ingest.ingest_network_data(
            g,
            layout=pd.DataFrame(
                np.zeros((5, 2), np.float64),
            ).T.to_dict(),
        )

    def test_count_nodes(self):
        g = self.make_ring()
        network_data = ipx.ingest.ingest_network_data(g)
        ptype = network_data["network_library"]
        provider = ipx.ingest.data_providers["network"][ptype]
        nvertices = provider(g).number_of_vertices()
        assert nvertices == 5

        g["nodes"] = [0, 1, 2, 3, 4]
        nvertices = provider(g).number_of_vertices()
        assert nvertices == 5

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
