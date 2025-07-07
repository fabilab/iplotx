style_library = {
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
    # Tree style, with zero-size vertices
    "tree": {
        "vertex": {
            "size": 0,
            "alpha": 0,
            "label": {
                "color": "black",
                "size": 12,
                "verticalalignment": "center",
                "hmargin": 10,
                "bbox": {
                    "boxstyle": "square,pad=0.5",
                    "facecolor": "white",
                    "edgecolor": "none",
                },
            },
        },
        "edge": {
            "linewidth": 2.5,
        },
    },
    # Greyscale style
    "greyscale": {
        "vertex": {
            "color": None,
            "facecolor": "lightgrey",
            "edgecolor": "#111",
            "marker": ">",
            "size": 10,
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
    # Colorful style for general use
    "unicorn": {
        "vertex": {
            "size": 30,
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
                    "black",
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
                    "black",
                ],
            },
        },
    },
}
