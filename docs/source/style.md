# Styles
Network plots can be customised using styles.

## What is a style?
Formally, a style is a **nested dictionary** specifying the visual properties of each graph element. The three top-level keys for a style dictionary are `vertex`, `edge`, and `grouping`. A typical style specification looks like this:

```python
mystyle = {
    "vertex": {
        "size": 20,
        "facecolor": "red",
        "edgecolor": "black",
        "linewidth": 1,
    },
    "edge": {
        "color": "steelblue",
        "linewidth": 2,
    }
}
```


`iplotx` has a default style that you can inspect as follows:

```python
from iplotx.style import default as default_style
print(default_style)
```

When a custom style is specified for a plot, it is applied **on top of** the current style, which is usually the default style.

```{warning}
  Multiple styling at once might be necessary. For example, the default style has white labels onto black vertex faces. If you want to have white vertices,
  you should also at the same time specify another color (e.g. black) for the labels, otherwise labels will be drawn white on white and therefore invisible.
```

`iplotx` also has a dictionary of predefined styles to serve as basis in different contexts. You can access these styles as follows:

```python
from iplotx.style import styles
print(styles)
```

For example, the `hollow` style uses vertices with no face color, black edges, black vertex labels, square vertices, and autosizes vertices to fit their text labels. This style is designed to be useful when label boxes are important to visualise the graph (e.g. company tree structures, or block-type diagrams).

## Applying styles
There are a few different ways to use a style in `iplotx` (the mechanism is similar to styles in `matplotlib`).

### Function argument
To apply a style to a single plot, you can pass it to the `iplotx.plot` function as a keyword argument:

```python
import iplotx as ipx
ipx.plot(
  ...,
  style={
    "vertex": {'size': 20},
  },
)
```

In addition to the method above, the `iplotx.plot` function accepts an additional syntax for styles, which is similar to how plotting styles work in `networkx` and `igraph`. You can specify each style change as a keyword argument, with underscores `_` used to mean a subdictionary. For instance, you can specify to have vertices with a red face and size 30 as follows:

```python
ipx.plot(
    ...,
    vertex_facecolor="red",
    vertex_size=30,
)
```

If both `style` and these custom arguments are used in the function, styles are applied first and these additional arguments are applied at the end.

````{important}
  This last syntax is designed to make short but deep style specifications easier. For instance, to change the background of edge labels to green, you can do:

  ```python
  ipx.plot(
      ...,
      edge_label_bbox_facecolor="grey",
  )
  ```

  which is more readable than the equivalent style dictionary:
  ```python
  ipx.plot(
      ...,
      style={'edge': {'label': {'bbox': {'facecolor': "grey"}}}},
  )
  ```
  However, they can become confusing if many details of the same element are styled at once. Do not do the following:
  ```python
  iplotx.plot(
      ...,
      vertex_size=20,
      vertex_facecolor="red",
      vertex_edgecolor="grey",
      vertex_linewidth=2,
      vertex_marker="d",
      vertex_label_color="black",
  )
  ```
  It is correct syntax, but obviously not very readable; use a dictionary instead.
````

### Style context
If you want a style to be applied beyond a single function call, you can use a style context:

```python
import iplotx as ipx
with iplotx.style.context(
    style={
        "vertex": {'size': 20},
    }
):
    # First plot uses this style
    ipx.plot(...)
    # Second plot ALSO uses the same style
    ipx.plot(...)
```

```{note}
  You can also pass the same `style` argument to all functions instead. Both achieve the same effect in practice, though the context is slightly more Pythonic.
```

### Permanent style
To apply a style permanently (in this Python session), you can use the `iplotx.style.use` function:

```python
import iplotx as ipx
ipx.style.use({
    "vertex": {"size": 20},
})

# From now on all plots will default to 20-point sized vertices unless specified otherwise
...
```

To specify a predefined style, you can just use its name as a string:

```python
ipx.style.use("hollow")
```

### Chaining styles
All style specifications can include one style (as shown in the examples above) or a list of styles, which will be applied in order on top of the current style (usually default). For instance, to use a hollow style customised to have red edges, you can do:

```python
with iplotx.style.context([
    "hollow",
    {"edge": {"color": "red"}},
]):
    ipx.plot(...)
```

This will take the current style (usually default), apply the "hollow" style on top, and then apply the red edge color on top of that. The style will revert when the context exists.

```{note}
  The same works for the `iplotx.plot` function, where you can pass a list of styles as the `style` argument.
```

### Rotating style leaves
All properties listed in the default style can be modified.

When **leaf properties** are set as list-like objects, they are applied to the graph elements in a cyclic manner (a similar mechanism is in place in `matplotlib` and `seaborn` for color palettes). For example, if you set `facecolor` to `["red", "blue"]`, the first vertex will be red, the second blue, the third red, and so on. This is called **style leaf rotation**.

To see all leaf properties, you can type:
```python
print(ipx.styles.style_leaves)
```

Style leaves can be rotated also using a dictionary instead of a list. In that case, vertex and/or edge IDs are used to match each element to their appearance. Here's an example:

```python
import networkx as nx
import iplotx as ipx

G = nx.Graph([(0, 1)])
ipx.plot(
    G,
    vertex_size={0: 20, 1: 30},
)
```

This is likely to be useful when plotting objects from `networkx`, which used nested dictionaries to store graph/vertex/edge properties.
