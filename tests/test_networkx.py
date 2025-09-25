import unittest
import importlib
import numpy as np
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import iplotx as ipx
from utils import image_comparison

if importlib.util.find_spec("networkx") is None:
    raise unittest.SkipTest("networkx not found, skipping tests")
else:
    import networkx as nx


class GraphTestRunner(unittest.TestCase):
    @image_comparison(baseline_images=["flat_style"], remove_text=True)
    def test_flat_style(self):
        G = nx.Graph(
            [
                (4, 1),
                (1, 2),
                (5, 3),
                (1, 3),
            ]
        )
        pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}
        ipx.network(
            G,
            layout=pos,
            node_labels=False,
            node_size=5,
            node_facecolor="white",
            node_edgecolor="black",
            node_linewidth=3,
            edge_linewidth=3,
            edge_color=["grey", "black"],
            margins=0.1,
        )

    @image_comparison(baseline_images=["simple_graph"], remove_text=True)
    def test_simple_graph(self):
        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(1, 3)
        G.add_edge(1, 5)
        G.add_edge(2, 3)
        G.add_edge(3, 4)
        G.add_edge(4, 5)

        # explicitly set positions
        pos = {1: (0, 0), 2: (-1, 0.3), 3: (2, 0.17), 4: (4, 0.255), 5: (5, 0.03)}

        ipx.network(
            G,
            layout=pos,
            node_labels=True,
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
            margins=0.2,
        )

    @image_comparison(baseline_images=["cluster-layout"], remove_text=True)
    def test_cluster_layout(self):
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
                "node": {
                    "facecolor": node_color,
                    "linewidth": 0,
                    "size": 21,
                }
            },
            ax=ax,
        )

    @image_comparison(baseline_images=["directed_graph_with_colorbar"], remove_text=True)
    def test_directed_graph(self):
        seed = 13648  # Seed random number generators for reproducibility
        G = nx.random_k_out_graph(10, 3, 0.5, seed=seed)
        pos = nx.spring_layout(G, seed=seed)

        node_sizes = [3 + 1.5 * i for i in range(len(G))]
        M = G.number_of_edges()
        edge_colors = list(range(2, M + 2))
        edge_alphas = [(5 + i) / (M + 4) for i in range(M)]
        cmap = plt.cm.plasma

        fig, ax = plt.subplots(figsize=(6, 4))

        arts = ipx.network(
            network=G,
            ax=ax,
            layout=pos,
            style={
                "node": {
                    "size": node_sizes,
                    "facecolor": "indigo",
                    "edgecolor": "none",
                },
                "edge": {
                    "color": edge_colors,
                    "alpha": edge_alphas,
                    "cmap": cmap,
                    "linewidth": 2,
                    "paralleloffset": 0,
                    "arrow": {
                        "marker": ">",
                        "width": 5,
                    },
                },
            },
        )
        fig.colorbar(
            arts[0].get_edges(),
            ax=ax,
        )

    @image_comparison(baseline_images=["empty_graph"], remove_text=True)
    def test_display_empty_graph(self):
        G = nx.empty_graph()
        fig, ax = plt.subplots()
        ipx.plot(G, ax=ax)

    @image_comparison(baseline_images=["shortest_path"], remove_text=True)
    def test_display_shortest_path(self):
        G = nx.Graph()
        G.add_nodes_from(["A", "B", "C", "D", "E", "F", "G", "H"])
        G.add_edge("A", "B", weight=4)
        G.add_edge("A", "H", weight=8)
        G.add_edge("B", "C", weight=8)
        G.add_edge("B", "H", weight=11)
        G.add_edge("C", "D", weight=7)
        G.add_edge("C", "F", weight=4)
        G.add_edge("C", "I", weight=2)
        G.add_edge("D", "E", weight=9)
        G.add_edge("D", "F", weight=14)
        G.add_edge("E", "F", weight=10)
        G.add_edge("F", "G", weight=2)
        G.add_edge("G", "H", weight=1)
        G.add_edge("G", "I", weight=6)
        G.add_edge("H", "I", weight=7)

        # Find the shortest path from node A to node E
        path = nx.shortest_path(G, "A", "E", weight="weight")

        # Create a list of edges in the shortest path
        path_edges = list(zip(path, path[1:]))
        nx.set_node_attributes(G, nx.spring_layout(G, seed=37), "pos")
        nx.set_edge_attributes(
            G,
            {
                (u, v): {
                    "color": (
                        "red"
                        if (u, v) in path_edges or tuple(reversed((u, v))) in path_edges
                        else "black"
                    ),
                    "label": d["weight"],
                }
                for u, v, d in G.edges(data=True)
            },
        )

        fig, ax = plt.subplots()
        ipx.network(
            G,
            ax=ax,
            layout="pos",
            node_labels=True,
            edge_labels=True,
        )

    @image_comparison(baseline_images=["house_with_colors"], remove_text=True)
    def test_display_house_with_colors(self):
        G = nx.house_graph()
        fig, ax = plt.subplots(figsize=(4, 4))
        nx.set_node_attributes(
            G, {0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1), 4: (0.5, 2.0)}, "pos"
        )
        nx.set_node_attributes(
            G,
            {
                n: {
                    "size": 40 if n != 4 else 30,
                    "color": "tab:blue" if n != 4 else "tab:orange",
                }
                for n in G.nodes()
            },
        )
        ipx.plot(
            G,
            ax=ax,
            layout="pos",
            style={
                "edge": {
                    "alpha": 0.5,
                    "linewidth": 6,
                },
                "vertex": {
                    "size": G.nodes.data("size"),
                    "facecolor": G.nodes.data("color"),
                    "edgecolor": "k",
                },
            },
            margins=0.1,
        )
        plt.tight_layout()

    @image_comparison(baseline_images=["labels_and_colors"], remove_text=True)
    def test_labels_and_colors(self):
        """Test complex labels and colors."""
        G = nx.cubical_graph()
        pos = nx.spring_layout(G, seed=3113794652)  # positions for all nodes
        nx.set_node_attributes(G, pos, "pos")  # Will not be needed after PR 7571
        labels = iter(
            [
                r"$a$",
                r"$b$",
                r"$c$",
                r"$d$",
                r"$\alpha$",
                r"$\beta$",
                r"$\gamma$",
                r"$\delta$",
            ]
        )
        nx.set_node_attributes(
            G,
            {
                n: {
                    "color": "tab:red" if n < 4 else "tab:blue",
                    "label": next(labels),
                }
                for n in G.nodes()
            },
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        ipx.network(
            G,
            ax=ax,
            layout="pos",
            node_labels=True,
            style={
                "vertex": {
                    "facecolor": G.nodes.data("color"),
                    "alpha": G.nodes.data("alpha"),
                    "size": 30,
                    "label": {"size": 22, "color": "whitesmoke"},
                    "zorder": 7,
                },
                "edge": {
                    "color": "tab:grey",
                },
            },
        )
        for i in range(2):
            subG = G.subgraph(range(i * 4, (i + 1) * 4))
            ipx.plot(
                subG,
                layout="pos",
                ax=ax,
                style={
                    "edge": {
                        "color": ["tab:red", "tab:blue"][i],
                        "alpha": 0.5,
                        "linewidth": 8,
                        "zorder": 0,
                    },
                    "vertex": {
                        "size": 25,
                        "alpha": 0,
                    },
                },
                margins=0.1,
            )
        plt.tight_layout()

    @image_comparison(baseline_images=["complex"], remove_text=True)
    def test_complex(self):
        import itertools as it

        nodes = "ABC"
        prod = list(it.product(nodes, repeat=2)) * 4
        G = nx.MultiDiGraph()
        for i, (u, v) in enumerate(prod):
            G.add_edge(u, v, w=round(i / 3, 2))
        nx.set_node_attributes(G, nx.spring_layout(G, seed=3113794652), "pos")
        csi = it.cycle([5 * r for r in it.accumulate([0.15] * 4)])
        nx.set_edge_attributes(G, {e: next(csi) for e in G.edges(keys=True)}, "tension")
        nx.set_edge_attributes(
            G,
            {tuple(e): w for *e, w in G.edges(keys=True, data="w")},
            "label",
        )

        fig, ax = plt.subplots()
        ipx.plot(
            G,
            ax=ax,
            layout="pos",
            edge_labels=True,
            margins=0.03,
            style={
                "edge": {
                    "curved": True,
                    "tension": G.edges.data("tension"),
                    "color": G.edges.data("w"),
                    "cmap": mpl.colormaps["inferno"],
                    "linewidth": 1,
                    "looptension": 7.5,
                    "label": {
                        "rotate": True,
                        "color": "black",
                        "bbox": {
                            "facecolor": "none",
                        },
                    },
                },
            },
        )

    @image_comparison(baseline_images=["complex_rotatelabels"], remove_text=True)
    def test_complex_rotatelabels(self):
        import itertools as it

        nodes = "ABC"
        prod = list(it.product(nodes, repeat=2)) * 4
        G = nx.MultiDiGraph()
        for i, (u, v) in enumerate(prod):
            G.add_edge(u, v, w=round(i / 3, 2))
        nx.set_node_attributes(G, nx.spring_layout(G, seed=3113794652), "pos")
        csi = it.cycle([5 * r for r in it.accumulate([0.15] * 4)])
        nx.set_edge_attributes(G, {e: next(csi) for e in G.edges(keys=True)}, "tension")
        nx.set_edge_attributes(
            G,
            {tuple(e): w for *e, w in G.edges(keys=True, data="w")},
            "label",
        )

        fig, ax = plt.subplots()
        ipx.plot(
            G,
            ax=ax,
            layout="pos",
            edge_labels=True,
            margins=0.03,
            style={
                "edge": {
                    "curved": True,
                    "tension": G.edges.data("tension"),
                    "color": G.edges.data("w"),
                    "cmap": mpl.colormaps["inferno"],
                    "linewidth": 1,
                    "looptension": 7.5,
                    "label": {
                        "rotate": False,
                        "color": "black",
                        "bbox": {
                            "facecolor": "none",
                        },
                    },
                },
            },
        )


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
