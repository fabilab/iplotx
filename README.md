![Github Actions](https://github.com/fabilab/iplotx/actions/workflows/test.yml/badge.svg)
![PyPI - Version](https://img.shields.io/pypi/v/iplotx)
![RTD](https://readthedocs.org/projects/iplotx/badge/?version=latest)

# iplotx
Plotting networks from igraph and networkx.

**NOTE**: This is currently alpha quality software. The API and functionality will break constantly, so use at your own risk. That said, if you have things you would like to see improved, please open a GitHub issue.

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
- Support trees from special libraries such as ete3, biopython, etc. This will need a dedicated function and layouting. ✅
- Support animations, e.g. showing the evolution of a network over time. 🏗️
- Support uni- and bi-directional communication between graph object and plot object.🏗️

## Authors
Fabio Zanini (https://fabilab.org)
