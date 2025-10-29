# Installing
```
pip install iplotx
```


## Quick Start
::::{tab-set}

:::{tab-item} igraph

```
import igraph as ig
import iplotx as ipx

g = ig.Graph.Ring(5)
layout = g.layout("circle").coords
ipx.network(g, layout)
```



:::

:::{tab-item} networkx
```
import networkx as nx
import iplotx as ipx

g = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
layout = nx.layout.circular_layout(g)
ipx.network(g, layout)
```

:::

::::

Either way, the result is the same:

![graph_basic](_static/graph_basic.png)

## Features
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
- interoperability with other charting tools (e.g. `seaborn`)
- chainable style contexts
- vertex clusterings and covers with convex hulls and rounding
- a plugin mechanism for additional libraries
- animations (see <project:gallery/other/plot_animation.rst>)
- 3D visualisations
- mouse/keyboard interaction and events (e.g. hover, click, see <project:gallery/other/plot_mouse_hover.rst>)
- ... and many more.

## Rationale
We believe graph **analysis**, graph **layouting**, and graph **visualisation** to be three separate tasks. `iplotx` currently focuses on visualisation. It can also compute simple tree layouts and might expand towards network layouts in the future.

## Citation
If you use `iplotx` for publication figures, please cite the [zenodo preprint](https://doi.org/10.5281/zenodo.16599333):

```
F. Zanini. (2025). Unified network visualisation in Python. Zenodo [PREPRINT]. https://doi.org/10.5281/zenodo.16599333
```

## Contributing
Open an [issue on GitHub](https://github.com/fabilab/iplotx/issues) to request features, report bugs, or show intention in contributing. Pull requests are very welcome.

```{important}
  If you are the maintainer of a network/graph/tree analysis library and would like
  to propose improvements or see support for it, please reach out with an issue/PR
  on GitHub!
```
