import os
import unittest
import pytest
import numpy as np
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt

import iplotx as ipx
from iplotx.importing import networkx as nx

# FIXME: find a better way to do this that works for both direct call and module
# import e.g. tox
try:
    from .utils import image_comparison
except ImportError:
    from utils import image_comparison


class GraphTestRunner(unittest.TestCase):
    def setUp(self):
        if nx is None:
            raise unittest.SkipTest("networkx not found, skipping tests")

    @image_comparison(baseline_images=["simple_graph"], remove_text=True)
    def test_simple_graph(self):
        plt.close("all")

        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(1, 5)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(4, 5)

        # explicitly set positions
        pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

        ipx.plot(
            G,
            layout=pos,
            vertex_labels=True,
            style={
                "vertex": {
                    "size": "label",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 5,
                    "label": {
                        "size": 36,
                        "color": "black",
                    },
                },
                "edge": {
                    "linewidth": 5,
                },
            },
        )

        # Set margins for the axes so that nodes aren't clipped
        ax = plt.gca()
        ax.margins(0.20)

    @image_comparison(baseline_images=["cluster-layout"], remove_text=True)
    def test_cluster_layout(self):
        plt.close("all")

        G = nx.davis_southern_women_graph()
        communities = nx.community.greedy_modularity_communities(G)

        # Layout does not appear deterministic, so we use a fixed layout from one run of spring_layout
        pos = {
            "Evelyn Jefferson": np.array([-25.68131066, -10.80854424]),
            "Laura Mandeville": np.array([-25.55280383, -10.99674201]),
            "Theresa Anderson": np.array([-25.38187247, -10.86875164]),
            "Brenda Rogers": np.array([-25.67901346, -11.17009239]),
            "Charlotte McDowd": np.array([-25.85576192, -11.09635554]),
            "Frances Anderson": np.array([-24.97100665, -10.80341997]),
            "Eleanor Nye": np.array([-25.22178561, -11.61473984]),
            "Pearl Oglethorpe": np.array([-24.43628739, -11.25006292]),
            "Ruth DeSand": np.array([-25.15041064, -11.82194973]),
            "E1": np.array([-26.10000875, -11.13170858]),
            "E2": np.array([-25.52047415, -10.41822091]),
            "E3": np.array([-25.48205106, -10.7056092]),
            "E4": np.array([-25.96271954, -10.75283759]),
            "E5": np.array([-25.35150057, -11.23805354]),
            "E6": np.array([-25.1129681, -11.11829674]),
            "E7": np.array([-25.5206234, -11.4515497]),
            "E13": np.array([-29.38952624, -9.82903937]),
            "Nora Fayette": np.array([-30.14866675, -9.89241409]),
            "Olivia Carleton": np.array([-30.44992786, -8.93390847]),
            "Katherina Rogers": np.array([-29.87511453, -10.05207037]),
            "Helen Lloyd": np.array([-30.70287724, -10.18706022]),
            "E12": np.array([-29.97554058, -10.44474811]),
            "E14": np.array([-30.32816285, -10.28626239]),
            "Sylvia Avondale": np.array([-29.68181361, -10.1960916]),
            "Myra Liddel": np.array([-29.50794269, -10.44247286]),
            "E9": np.array([-29.86346377, -9.54678644]),
            "Flora Price": np.array([-30.07123751, -8.8956997]),
            "E10": np.array([-30.15355559, -10.54517655]),
            "E11": np.array([-30.59292037, -9.39236595]),
            "E8": np.array([-19.26284314, -9.57050676]),
            "Dorothy Murchison": np.array([-19.48148306, -10.57263599]),
            "Verne Sanderson": np.array([-19.0455969, -8.57476521]),
        }

        # Nodes colored by cluster
        node_color = {}
        for nodes, clr in zip(communities, ("tab:blue", "tab:orange", "tab:green")):
            for node in nodes:
                node_color[node] = clr

        fig, ax = plt.subplots(figsize=(6, 6))
        ipx.plot(
            G,
            layout=pos,
            style={
                "vertex": {
                    "facecolor": node_color,
                    "linewidth": 0,
                    "size": 25,
                }
            },
            ax=ax,
        )

    @image_comparison(baseline_images=["directed_graph"], remove_text=True)
    def test_directed_graph(self):
        plt.close("all")

        seed = 13648  # Seed random number generators for reproducibility
        G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
        pos = nx.spring_layout(G, seed=seed)

        node_sizes = [3 + 1.5 * i for i in range(len(G))]
        M = G.number_of_edges()
        edge_colors = list(range(2, M + 2))
        edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
        cmap = plt.cm.plasma

        fig, ax = plt.subplots(figsize=(6, 4))

        ipx.plot(
            network=G,
            ax=ax,
            layout=pos,
            style={
                "vertex": {
                    "size": node_sizes,
                    "facecolor": "indigo",
                    "edgecolor": "none",
                },
                "edge": {
                    "color": edge_colors,
                    "alpha": edge_alphas,
                    "cmap": cmap,
                    "linewidth": 2,
                    "offset": 0,
                },
                "arrow": {
                    "marker": ">",
                    "width": 5,
                },
            },
        )

        fig.colorbar(
            plt.cm.ScalarMappable(
                cmap=cmap,
                norm=mpl.colors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)),
            ),
            ax=ax,
        )

    # @image_comparison(baseline_images=["igraph_layout_object"], remove_text=True)
    # def test_layout_attribute(self):
    #    plt.close("all")
    #    g = ig.Graph.Ring(5)
    #    layout = g.layout("circle")
    #    fig, ax = plt.subplots(figsize=(3, 3))
    #    ipx.plot(g, layout=layout, ax=ax)

    # @image_comparison(baseline_images=["graph_layout_attribute"], remove_text=True)
    # def test_layout_attribute(self):
    #    plt.close("all")
    #    g = ig.Graph.Ring(5)
    #    g["layout"] = ig.Layout([(x, x) for x in range(g.vcount())])
    #    fig, ax = plt.subplots(figsize=(3, 3))
    #    ipx.plot(g, layout="layout", ax=ax)

    # @pytest.mark.skip(reason="curved edges are currently a little off-center")
    # @image_comparison(baseline_images=["graph_directed_curved_loops"], remove_text=True)
    # def test_directed_curved_loops(self):
    #    plt.close("all")
    #    g = ig.Graph.Ring(5, directed=True)
    #    g.add_edge(0, 0)
    #    g.add_edge(0, 0)
    #    g.add_edge(2, 2)
    #    fig, ax = plt.subplots(figsize=(4, 4))
    #    ax.set_xlim(-1.2, 1.2)
    #    ax.set_ylim(-1.2, 1.2)
    #    ipx.plot(
    #        g,
    #        ax=ax,
    #        layout=self.layout_small_ring,
    #        style={
    #            "edge": {
    #                "curved": True,
    #                "tension": 1.7,
    #                "loop_tension": 5,
    #            }
    #        },
    #    )

    # @image_comparison(baseline_images=["graph_squares_directed"], remove_text=True)
    # def test_mark_groups_squares(self):
    #    plt.close("all")
    #    g = ig.Graph.Ring(5, directed=True)
    #    fig, ax = plt.subplots(figsize=(3, 3))
    #    ipx.plot(
    #        g,
    #        ax=ax,
    #        layout=self.layout_small_ring,
    #        style={
    #            "vertex": {"marker": "s"},
    #        },
    #    )

    # @image_comparison(baseline_images=["graph_edit_children"], remove_text=True)
    # def test_edit_children(self):
    #    plt.close("all")
    #    g = ig.Graph.Ring(5)
    #    fig, ax = plt.subplots(figsize=(4, 4))
    #    ipx.plot(
    #        g,
    #        ax=ax,
    #        style={"vertex": {"marker": "o"}},
    #        layout=self.layout_small_ring,
    #    )
    #    graph_artist = ax.get_children()[0]

    #    dots = graph_artist.get_vertices()
    #    dots.set_facecolors(["blue"] + list(dots.get_facecolors()[1:]))
    #    new_sizes = dots.get_sizes()
    #    new_sizes[1] = 30
    #    dots.set_sizes(new_sizes)

    #    lines = graph_artist.get_edges()
    #    lines.set_edgecolor("green")

    # @pytest.mark.skip(reason="curved edges are currently a little off-center")
    # @image_comparison(baseline_images=["graph_with_curved_edges"])
    # def test_graph_with_curved_edges(self):
    #    plt.close("all")
    #    g = ig.Graph.Ring(24, directed=True, mutual=True)
    #    fig, ax = plt.subplots()
    #    lo = g.layout("circle")
    #    lo.scale(3)
    #    ipx.plot(
    #        g,
    #        ax=ax,
    #        layout=lo,
    #        style={
    #            "vertex": {
    #                "size": 15,
    #            },
    #            "edge": {
    #                "offset": 8,
    #                "curved": True,
    #                "tension": 0.5,
    #            },
    #            "arrow": {
    #                "height": 5,
    #                "width": 5,
    #            },
    #        },
    #    )
    #    ax.set_aspect(1.0)

    # @pytest.mark.skip(
    #    reason="curved edges are currently a little off-center and not offset properly"
    # )
    # @image_comparison(baseline_images=["multigraph_with_curved_edges_undirected"])
    # def test_graph_with_curved_edges(self):
    #    plt.close("all")
    #    g = ig.Graph.Ring(24, directed=False)
    #    g.add_edges([(0, 1), (1, 2)])
    #    fig, ax = plt.subplots()
    #    lo = g.layout("circle")
    #    lo.scale(3)
    #    ipx.plot(
    #        g,
    #        ax=ax,
    #        layout=lo,
    #        style={
    #            "vertex": {
    #                "size": 15,
    #            },
    #            "edge": {
    #                "offset": 8,
    #                "curved": True,
    #                "tension": 0.5,
    #            },
    #            "arrow": {
    #                "height": 5,
    #                "width": 5,
    #            },
    #        },
    #    )
    #    ax.set_aspect(1.0)

    # @image_comparison(baseline_images=["graph_null"])
    # def test_null_graph(self):
    #    plt.close("all")
    #    g = ig.Graph()
    #    fig, ax = plt.subplots()
    #    ipx.plot(g, ax=ax)
    #    ax.set_aspect(1.0)


def suite():
    return unittest.TestSuite(
        [
            unittest.defaultTestLoader.loadTestsFromTestCase(GraphTestRunner),
        ]
    )


def test():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == "__main__":
    test()
