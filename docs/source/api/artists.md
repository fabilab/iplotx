# Artist hierarchy
`iplotx.network` return a list of `matplotlib` artists (1 or 2). When a network is plotted, the first artist is an instance of `iplotx.NetworkArtist`. This class contains the visual elements representing vertices, edges, labels, arrows, etc. and can be used to further edit the plot after `iplotx.plot` returned.

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


## 3D artists
The {py:class}`iplotx.network.NetworkArtist` class is also used for 3D plots. In that case, the `VertexCollection` and `EdgeCollection` instances are substituted by the following:

```{eval-rst}
.. autoclass:: iplotx.art3d.vertex.Vertex3DCollection
    :members:

.. autoclass:: iplotx.art3d.edge.Edge3DCollection
    :members:

```
