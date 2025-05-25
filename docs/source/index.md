# iplotx documentation

`iplotx` is a Python library to display graphs or networks using `matplotlib` as a backend. It supports multiple network analysis libraries (currently `networkx` and `igraph`).

## Installation
```
pip install iplotx
```


## Quick Start
::::{tab-set}

:::{tab-item} igraph

```
import igraph as ig
import matplotlib.pyplot as plt
import iplotx as ipx

g = ig.Graph.Ring(5)
layout = g.layout("circle").coords
fig, ax = plt.subplots(figsize=(3, 3))
ipx.plot(g, ax=ax, layout=layout)
```



:::

:::{tab-item} networkx
```
import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

g = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
layout = nx.layout.circular_layout(g)
fig, ax = plt.subplots(figsize=(3, 3))
ipx.plot(g, ax=ax, layout=layout)
```

:::

::::

Either way, the result is the same:

![graph_basic](_static/graph_basic.png)

## Gallery
See <project:gallery/index.rst>.

## Styles
`iplotx` is designed to produce publication-quality figures using **styles**, which are dictionaries specifying the visual properties of each graph element, including vertices, edges, arrows, labels, and groupings.

The default style is shown here below as a reference:

```python
default = {
    "vertex": {
        "size": 20,
        "facecolor": "black",
        "marker": "o",
        "label": {
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
        "offset": 3,
        "tension": 1,
        "label": {
            "horizontalalignment": "center",
            "verticalalignment": "center",
        },
    },
    "arrow": {
        "marker": "|>",
        "width": 8,
        "color": "black",
    },
    "grouping": {
        "facecolor": ["grey", "steelblue", "tomato"],
        "edgecolor": "black",
        "linewidth": 1.5,
        "alpha": 0.5,
        "vertexpadding": 25,
    },
}
```

All properties listed in the default style can be modified. When **leaf properties** are set as lists-like objects, they are applied to the graph elements in a cyclic manner. For example, if you set `facecolor` to `["red", "blue"]`, the first vertex will be red, the second blue, the third red, and so on.

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

gallery/index
```
