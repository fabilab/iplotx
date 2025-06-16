![Github Actions](https://github.com/fabilab/iplotx/actions/workflows/test.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/iplotx)
![RTD](https://readthedocs.org/projects/iplotx/badge/?version=latest)
![pylint](assets/pylint.svg)

# iplotx
Plotting networks from igraph and networkx.

**NOTE**: This is currently beta quality software. The API and functionality are settling in and might break occasionally.

## Installation
```bash
pip install iplotx
```

## Quick Start
```python
import networkx as nx
import matplotlib.pyplot as plt
import iplotx as ipx

g = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
layout = nx.layout.circular_layout(g)
fig, ax = plt.subplots(figsize=(3, 3))
ipx.plot(g, ax=ax, layout=layout)
```

![Quick start image](docs/source/_static/graph_basic.png)

## Documentation
See [readthedocs](https://iplotx.readthedocs.io/en/latest/) for the full documentation.

## Gallery
See [gallery](https://iplotx.readthedocs.io/en/latest/gallery/index.html).

## Roadmap
- Plot networks from igraph and networkx interchangeably, using matplotlib as a backend. ✅
- Support interactive plotting, e.g. zooming and panning after the plot is created. ✅
- Support storing the plot to disk thanks to the many matplotlib backends (SVG, PNG, PDF, etc.). ✅
- Support flexible yet easy styling. ✅
- Efficient plotting of large graphs using matplotlib's collection functionality. ✅
- Support editing plotting elements after the plot is created, e.g. changing node colors, labels, etc. ✅
- Support animations, e.g. showing the evolution of a network over time. ✅
- Support mouse interaction, e.g. hovering over or clicking on nodes and edges to get information about them. ✅
- Support trees from special libraries such as ete3, biopython, etc. This will need a dedicated function and layouting. ✅
- Support uni- and bi-directional communication between graph object and plot object.🏗️

**NOTE:** The last item can probably be achieved already by using `matplotlib`'s existing callback functionality. It is currently untested, but if you manage to get it to work on your graph let me know and I'll add it to the examples (with credit).

## Authors
Fabio Zanini (https://fabilab.org)
