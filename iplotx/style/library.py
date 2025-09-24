style_library = {
    "default": {
        "vertex": {
            "size": 20,
            "facecolor": "black",
            "marker": "o",
            "label": {
                "color": "white",
                "horizontalalignment": "center",
                "verticalalignment": "center",
                "hpadding": 18,
                "vpadding": 12,
            },
        },
        "edge": {
            "linewidth": 1.5,
            "linestyle": "-",
            "color": "black",
            "curved": False,
            "tension": 1,
            "looptension": 4,
            "loopmaxangle": 60,
            "paralleloffset": 3,
            "label": {
                "horizontalalignment": "center",
                "verticalalignment": "center",
                "rotate": False,
                "bbox": {
                    "boxstyle": "round",
                    "facecolor": "white",
                    "edgecolor": "none",
                },
            },
            "arrow": {
                "marker": "|>",
                "width": 8,
            },
        },
        "grouping": {
            "facecolor": ["grey", "steelblue", "tomato"],
            "edgecolor": "black",
            "linewidth": 1.5,
            "alpha": 0.5,
            "vertexpadding": 18,
        },
    },
    # Hollow style for organization charts et similar
    "hollow": {
        "vertex": {
            "color": None,
            "facecolor": "none",
            "edgecolor": "black",
            "linewidth": 1.5,
            "marker": "r",
            "size": "label",
            "label": {
                "color": "black",
            },
        }
    },
    # Feedback, for regulatory networks
    "feedback": {
        "edge": {
            "linewidth": 4,
            "shrink": 10,
            "arrow": {
                "marker": ")>",
                "width": 20,
                "height": 28,
            },
        },
        "vertex": {
            "size": 35,
            "color": None,
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 4,
            "label": {
                "color": "black",
            },
        },
    },
    # Tree style, with zero-size vertices
    "tree": {
        "vertex": {
            "size": 0,
            "label": {
                "color": "black",
                "size": 12,
                "verticalalignment": "center",
                "hmargin": 10,
                "bbox": {
                    "boxstyle": "square,pad=0.5",
                    "facecolor": "none",
                    "edgecolor": "none",
                },
            },
        },
        "edge": {
            "linewidth": 2.5,
        },
    },
    # Cogent3 tree style
    "cogent3": {
        "vertex": {
            "size": 3,
            "label": {
                "color": "black",
                "size": 10,
                "verticalalignment": "center",
                "bbox": {
                    "facecolor": "none",
                    "edgecolor": "none",
                },
            },
        },
        "edge": {
            "linewidth": 1.5,
        },
        "leaf": {
            "deep": False,
        },
    },
    # Dashed depth tree branches
    "dashdepth": {
        "vertex": {
            "size": 2,
            "label": {
                "color": "black",
                "size": 10,
                "verticalalignment": "center",
                "bbox": {
                    "facecolor": "none",
                    "edgecolor": "none",
                },
            },
        },
        "edge": {
            "linewidth": 1.5,
            "split": {
                "linestyle": ":",
            },
        },
        "leaf": {
            "deep": False,
        },
    },
    # Dashed depth tree branches
    "dashwidth": {
        "vertex": {
            "size": 2,
            "label": {
                "color": "black",
                "size": 10,
                "verticalalignment": "center",
                "bbox": {
                    "facecolor": "none",
                    "edgecolor": "none",
                },
            },
        },
        "edge": {
            "linewidth": 1.5,
            "linestyle": ":",
            "split": {
                "linestyle": "-",
            },
        },
        "leaf": {
            "deep": False,
        },
    },
    # Greyscale style
    "greyscale": {
        "vertex": {
            "color": None,
            "facecolor": "lightgrey",
            "edgecolor": "#111",
            "marker": ">",
            "size": 15,
            "linewidth": 0.75,
            "label": {
                "color": "black",
            },
        },
        "edge": {
            "color": "#111",
            "linewidth": 0.75,
            "arrow": {
                "marker": ">",
            },
            "label": {
                "rotate": True,
                "color": "black",
            },
        },
    },
    # Edge highlight
    "rededge": {
        "vertex": {
            "size": 13,
            "color": None,
            "facecolor": "none",
            "edgecolor": "#111",
            "linewidth": 2,
            "label": {
                "color": "#111",
            },
        },
        "edge": {
            "color": "tomato",
            "linewidth": 2.5,
        },
    },
    # Vertex highlight
    "rednode": {
        "vertex": {
            "size": 22,
            "color": None,
            "facecolor": "tomato",
            "edgecolor": "firebrick",
            "linewidth": 2,
        },
        "edge": {
            "color": "#333",
            "linewidth": 1,
        },
    },
    # Eerie style
    "eerie": {
        "vertex": {
            "color": None,
            "facecolor": "white",
            "edgecolor": "#111",
            "linestyle": "--",
            "linewidth": 2,
        },
        "edge": {
            "color": "#111",
            "linestyle": "--",
            "linewidth": 2,
            "arrow": {
                "marker": ")",
            },
        },
    },
    # Emulate a little networkx
    "networkx": {
        "vertex": {
            "color": None,
            "facecolor": "steelblue",
            "edgecolor": "none",
            "label": {
                "color": "black",
            },
        },
        "edge": {
            "color": "black",
            "linewidth": 1.5,
            "arrow": {
                "width": 5,
            },
        },
    },
    "igraph": {
        "vertex": {
            "color": None,
            "facecolor": "lightblue",
            "edgecolor": "black",
            "linewidth": 2,
            "label": {
                "color": "black",
            },
        },
        "edge": {
            "color": "black",
            "linewidth": 2,
            "arrow": {
                "marker": "|>",
                "width": 6.5,
            },
        },
    },
    # Colorful style for general use
    "unicorn": {
        "vertex": {
            "size": 23,
            "color": None,
            "facecolor": [
                "darkorchid",
                "slateblue",
                "seagreen",
                "lime",
                "gold",
                "orange",
                "sandybrown",
                "tomato",
                "deeppink",
            ],
            "edgecolor": "black",
            "linewidth": 1,
            "marker": "*",
            "label": {
                "color": [
                    "white",
                    "white",
                    "white",
                    "black",
                    "black",
                    "black",
                    "black",
                    "white",
                    "white",
                ],
            },
        },
        "edge": {
            "color": [
                "darkorchid",
                "slateblue",
                "seagreen",
                "lime",
                "gold",
                "orange",
                "sandybrown",
                "tomato",
                "deeppink",
            ],
            "linewidth": 1.5,
            "label": {
                "rotate": False,
                "color": [
                    "white",
                    "white",
                    "white",
                    "black",
                    "black",
                    "black",
                    "black",
                    "white",
                    "white",
                ],
            },
        },
    },
}
