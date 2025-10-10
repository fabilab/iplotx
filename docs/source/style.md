# Styles
Visualisations can be customised using styles.

## What is a style?
Formally, a style is a **nested dictionary** specifying the visual properties of each graph element. The main top-level keys for a style dictionary are `vertex` and `edge`. A typical style specification looks like this:

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

Additional top-level keys exist for networks (e.g. `grouping`) and trees (e.g. `leaf`, `clade`, `internal`). The complete style specification is documented in the [API reference](api/complete_style_specification.md).

`iplotx` has a default style that you can inspect as follows:

```python
from iplotx.style import default as default_style
print(default_style)
```

When a custom style is specified for a plot, it is applied **on top of** the current style, which is usually the default style.

```{warning}
  The default style has black vertices with white vertex labels. If you change the vertex face color, you might want to change the vertex label color as well to ensure readability.
```

`iplotx` also has an internal [library of styles](gallery/style/plot_multistyle.rst) to serve as basis in different contexts. You can access these styles as follows:

```python
from iplotx.style import styles
print(styles)
```

For example, the `hollow` style uses vertices with no face color, black edges, black vertex labels, square vertices, and autosizes vertices to fit their text labels. This style is designed to be useful when label boxes are important to visualise the graph (e.g. company tree structures, or block-type diagrams).

## Applying styles
There are a few different ways to use a style in `iplotx` (the mechanism is similar to styles in `Matplotlib`).

### Single function calls
To apply a style to a single plot, you can pass it to the {func}`.network` and {func}`.tree` functions as a keyword argument:

```python
import iplotx as ipx
ipx.network(
  ...,
  style={
    "vertex": {'size': 20},
  },
)
```

These functions also accept individual element styling via keyword arguments, with underscores `_` meant for splitting levels. For instance, you can specify to have vertices with a red face and size 30 as follows:

```python
ipx.network(
    ...,
    vertex_facecolor="red",
    vertex_size=30,
)
```

If both `style` and these custom arguments are used in the function, styles are applied first and individual keyword arguments are applied at the end, e.g.:

```python
ipx.network(
    ...,
    style="unicorn",
    vertex_facecolor="red",
    vertex_size=30,
)
```

### Style contexts
If you want a style to be applied beyond a single function call, you can use a {func}`.style.context`:

```python
import iplotx as ipx
with iplotx.style.context(
    style={
        "vertex": {'size': 20},
    }
):
    # First plot uses this style
    ipx.network(...)
    # Second plot ALSO uses the same style
    ipx.network(...)
```

```{note}
  You can also pass the same `style` argument to all functions instead. Both achieve the same effect in practice, though the context is slightly more Pythonic.
```

### Permanent styles
To apply a style permanently (in this Python session), you can use the {func}`.style.use` function:

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

## Reverting to default style
To reset `iplotx`'s style to the default one, you can use the {func}`.style.reset` function:

```python
ipx.style.reset()
```

## Chaining styles
All three style specifications methods accept both a single style or a list of styles. Multiple styles, if present, are applied in order on top of the current style (usually default). For instance, to use a hollow style customised to have red edges, you can do:

```python
with iplotx.style.context([
    "hollow",
    {"edge": {"color": "red"}},
]):
    ipx.network(...)
```

This will take the current style (usually default), apply the "hollow" style on top, and then apply the red edge color on top of that. The style will revert when the context exists.

```{note}
  The same works for the {func}`.network` and {func}`.tree` functions, where you can pass a list of styles as the `style` argument.
```

## Rotating style leaves
All properties listed in the default style can be modified.

When **leaf properties** are set as list-like objects, they are applied to the graph elements in a cyclic manner (a similar mechanism is in place in `Matplotlib` and `seaborn` for color palettes). For example, if you set `facecolor` to `["red", "blue"]`, the first vertex will be red, the second blue, the third red, and so on. This is called **style leaf rotation**.

Style leaves can be rotated also using a dictionary instead of a list. In that case, vertex and/or edge IDs are used to match each element to their appearance. Here's an example:

```python
import networkx as nx
import iplotx as ipx

G = nx.Graph([(0, 1)])
ipx.network(
    G,
    vertex_size={0: 20, 1: 30},
)
```

These dictionaries (or dict-like, e.g. `defaultdict`) can be partial, i.e. only specify a custom styling for some elements within a class (e.g. vertex color). The current style will be applied as fallback for elements not specified in the dictionary.

```{note}
  When using dictionaries for style leaves, make sure that the keys match the vertex/edge IDs exactly. For instance, if your graph has string vertex IDs, using integers as keys might not work.
```


To see all leaf properties, you can type:

```python
print(ipx.styles.style_leaves)
```

To see properties that *cannot* be rotated, you can type:

```python
print(ipx.styles.nonrotating_leaves)
```


Please open a [GitHub issue](https://github.com/fabilab/iplotx/issues) if you would like a property listed in `nonrotating_leaves` to be rotated.
