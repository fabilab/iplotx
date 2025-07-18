# iplotx documentation

`iplotx` is a Python library to display graphs, networks and trees using `matplotlib` as a backend. It supports multiple network analysis libraries including `networkx` and `igraph` for networks and `biopython`, `scikit-bio`, `cogent3`, and `ete4` for trees.

`iplotx` guarantees the **exact same visual appearance** independently of what library you used to construct the network/tree.

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
See <project:gallery/index.rst> for examples of plots made with `iplotx`. Feel free to suggest new examples on GitHub by opening a new issue or pull request!

## Features
```{important}
    If you are the maintainer of a network/graph/tree analysis library and would like
    to propose improvements or see support for it, please reach out with an issue/PR
    on GitHub!
```

`iplotx`'s features' include:
- per-edge and per-vertex styling using sequences or dictionaries
- labels
- arrows
- tunable offset for parallel (i.e. multi-) edges
- ports (a la Graphviz)
- curved edges and self-loops with controllable tension
- tree layouts
- label-based vertex autoscaling
- node label margins and padding
- export to many formats (e.g. PNG, SVG, PDF, EPS) thanks to `matplotlib`
- compatibility with many GUI frameworks (e.g. Qt, GTK, Tkinter) thanks to `matplotlib`
- data-driven axes autoscaling
- consistent behaviour upon zooming and panning
- correct HiDPI scaling (e.g. retina screens) including for vertex sizes, arrow sizes, and edge offsets
- a consistent `matplotlib` artist hierarchy
- post-plot editability (e.g. for animations)
- plays well with other charting tools (e.g. `seaborn`)
- chainable style contexts
- vertex clusterings and covers with convex hulls and rounding
- a plugin mechanism for additional libraries (WIP)
- animations (see <project:gallery/plot_animation.rst>)
- mouse/keyboard interaction and events (e.g. hover, click, see <project:gallery/plot_mouse_hover.rst>)
- ... and probably more by the time you read this.

## Styles
`iplotx` is designed to produce publication-quality figures using **styles**, which are dictionaries specifying the visual properties of each graph element, including vertices, edges, arrows, labels, and groupings.

An example of style specification is as follows:

```python
import iplotx as ipx
with ipx.style.context({
    'vertex': {
        'size': 30,
        'facecolor': 'red',
        'edgecolor': 'none',
    }
}):
    ipx.plot(...)
```

See <project:style.md> for more info.

## Reference API
See <project:api.md> for reference documentation of all functions and classes in `iplotx`.

## Rationale
We believe graph **analysis**, graph **layouting**, and graph **visualisation** to be three separate tasks. `iplotx` currently focuses on visualisation. It can also compute simple tree layouts and might expand towards network layouts in the future.

```{toctree}
:maxdepth: 2
:caption: Contents

gallery/index
style
Data providers <providers>
API <api>
Plotting API <api/plotting>
Styling API <api/style>
Complete style specification <api/complete_style_specification>
Artist hierarchy <api/artists>
Data provider protocols <api/providers>
```
