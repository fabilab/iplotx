# Reference API
This is the reference documentation for `iplotx`.

## Main functions
The main user-facing function is `iplotx.network`, which can be used to plot graphs and graph groupings (covers and clusterings), and `iplotx.tree` to plot trees. `iplotx.plot` is a synonym for `iplotx.network`.

```{eval-rst}
.. autofunction:: iplotx.network
    :noindex:

.. autofunction:: iplotx.tree
    :noindex:
```

## Styling
See also the <project:style.md> for an introduction to styles in `iplotx`.


```{eval-rst}
.. autofunction:: iplotx.style.context

.. autofunction:: iplotx.style.use

.. autofunction:: iplotx.style.reset

.. autofunction:: iplotx.style.get_style
```

The following functions are reported for completeness but are rarely used by users directly:

```{eval-rst}
.. autofunction:: iplotx.style.unflatten_style

.. autofunction:: iplotx.style.rotate_style
```

## Artist hierarchy
`iplotx.plot` return a list of `matplotlib` artists (1 or 2). When a network is plotted, the first artist is an instance of `iplotx.NetworkArtist`. This class contains the visual elements representing vertices, edges, labels, arrows, etc. and can be used to further edit the plot after `iplotx.plot` returned.

A `NetworkArtist` instance has two notable properties: vertices and edges, which are instances of `VertexCollection` and `EdgeCollection`, respectively. These collections are `matplotlib` artists that can be used to modify the appearance of vertices and edges after the plot has been created.

In turn, a `VertexCollection` or `EdgeCollection` instance **may** contain a `LabelCollection` instance if the plot includes labels. Moreover, an `EdgeCollection` instance **may** contain an `EdgeArrowCollection` instance if the graph is directed.

```{eval-rst}
.. autoclass:: iplotx.network.NetworkArtist
    :members:


.. autoclass:: iplotx.vertex.VertexCollection
    :members:

.. autoclass:: iplotx.edge.EdgeCollection
    :members:

.. autoclass:: iplotx.label.LabelCollection
    :members:

.. autoclass:: iplotx.edge.arrow.EdgeArrowCollection
    :members:
```
