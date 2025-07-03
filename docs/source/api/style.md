# Styling
See also the <project:../style.md> page for an introduction to styles in `iplotx`.

## Functions


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

## Commented style specification
The complete style dictionary specification is as follows:

```python
{
    # Vertex style
    "vertex": {
        # Size of the vertex in points. If a pair, it indicates width and height
        # of the marker.
        "size": float | tuple[float, float],

        # Marker style. Currently supported markers:
        # o, c, circle: circle
    f   # s, square, r, rectangle: rectangle (square if only one size specified)
        # ^, triangle: upward triangle
        # v, triangle_down: downward triangle
        # <, triangle_left: leftward triangle
        # >, triangle_right: rightward triangle
        # d, diamond: diamond
        # e, ellipse: ellipse
        # p, pentagon: pentagon
        # h, hexagon: hexagon
        # 8, octagon: octagon
        # *, star: 5-point star, upright
        "marker": str,
        
        "facecolor": str | Any,  # Color of the vertex face (e.g. 'red', '#FF0000')
        "edgecolor": str | Any,  # Color of the vertex edge (e.g. 'black', '#000000')
        "alpha": float,  # Opacity of the vertex (0.0 for fully transparent, 1.0 for fully opaque)

        # Vertex label style
        "label": {
            "color": str | Any,  # Color of the vertex label (e.g. 'white', '#FFFFFF')
            "horizontalalignment": str,  # Horizontal alignment of the label ('left', 'center', 'right')
            "verticalalignment": str,  # Vertical alignment of the label ('top', 'center', 'bottom', 'baseline', 'center_baseline')
            "hpadding": float,  # Horizontal padding around the label
            "vpadding": float,  # Vertical padding around the label

            # Bounding box properties for the label
            "bbox": {
                "boxstyle": str,  # Style of the bounding box ('round', 'square', etc.)
                "facecolor": str | Any,  # Color of the bounding box (e.g. 'yellow', '#FFFF00')
                "edgecolor": str | Any,  # Color of the bounding box edge (e.g. 'black', '#000000')
                ...  # Any other bounding box property

            },
        },
    },

    # Edge style
    "edge": {
        "linewidth": float,  # Width of the edge line in points
        "linestyle": str,  # Style of the edge line ('-' for solid, '--' for dashed, etc.)
        "color": str | Any,  # Color of the edge line (e.g. 'blue', '#0000FF')
        "alpha": float,  # Opacity of the vertex (0.0 for fully transparent, 1.0 for fully opaque)

        "curved": bool,  # Whether the edge is curved (True) or straight (False)


        # Tension for curved edges (0.0 for straight, higher values position the
        # Bezier control points further away from the nodes, creating more wiggly lines)
        # Negative values bend the curve on the other side of the straight line.
        "tension": float,

        # Tension for self-loops (higher values create more bigger loops).
        # This is typically a strictly positive value.
        "looptension": float,

        # The maximum angle for self-loops (in degrees). This is typically a positive
        # number quite a bit less than 180 degrees to avoid funny-looking self-loops.
        "loopmaxangle": float,

         # Edge label style
        "label": {
            "color": str | Any,  # Color of the vertex label (e.g. 'white', '#FFFFFF')
            "horizontalalignment": str,  # Horizontal alignment of the label ('left', 'center', 'right')
            "verticalalignment": str,  # Vertical alignment of the label ('top', 'center', 'bottom', 'baseline', 'center_baseline')
            "hpadding": float,  # Horizontal padding around the label
            "vpadding": float,  # Vertical padding around the label
            "bbox": dict,  # Bounding box properties for the label (see vertex labels)
        },

        # Edge arrow style for directed graphs
        "arrow": {
            # Arrow marker style. Currently supported:
            # |>
            # >
            # >>
            # )>
            # )
            # |
            # s
            # d
            # p
            # q
            "marker": str,
            "width": float,  # Width of the arrow in points

            # Height of the arrow in points. Defaults to 1.3 times the width if not
            # specified. If the string "width" is used, it is set to the width of
            # the arrow.
            "height": float | str,

            # Color of the arrow (e.g. 'black', '#000000'). This means both the
            # facecolor and edgecolor for full arrows (e.g. "|>"), only the edgecolor
            # for hollow arrows (e.g. ">"). This specification, like in matplotlib,
            # takes higher precedence than the "edgecolor" and "facecolor" properties.
            # If this property is not specified, the color of the edge to which the
            # arrowhead belongs to is used.
            "color": str | Any,

        },
    },

    # The following entry is used by networks ONLY (not trees as plotted by `iplotx.tree`)
    "grouping": {
        "facecolor": str | Any,  # Color of the grouping face (e.g. 'lightgray', '#D3D3D3')
        "edgecolor": str | Any,  # Color of the grouping edge (e.g. 'gray', '#808080')
        "linewidth": float,  # Width of the grouping edge in points
        "vertexpadding": float,  # Padding around the vertices in the grouping in points
    },

    # The following entries are used by trees ONLY (not by networks as plotted by `iplotx.network`)
    "cascade": {

    },

    "leaf": {

    },
}
```
