# Complete style specification
```{note}
  There might be additional matplotlib options not mentioned below (e.g. rotation,
  font family). Any options not mentioned below are passed directly to matplotlib.
```


```python
{
    # Vertex/node style
    # NOTE: you can use "node" or "vertex" interchangeably. If both are specified
    # at the SAME time, these styles are merged, with conflicts resolved in favour
    # of "node". In other words, "vertex" is applied first, then "node" on top of it.

    "vertex" | "node": {
        # Size of the vertex in points. If a pair, it indicates width and height
        # of the marker. If "label", set the size dynamically based on the vertex
        # label (a label is needed in this case)
        "size": float | tuple[float, float] | str,

        # Marker style. Currently supported markers:
        # o, c, circle: circle
        # s, square, r, rectangle: rectangle (square if only one size specified)
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

        "depthshade": bool,  # Whether to shade the color based on depth (3D only)

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
            "hmargin": float,  # Horizontal margin around (usually left of) the label
            "vmargin": float,  # Vertical margin around (usually below) the label. Rarely used.
        },
    },

    # Edge style
    "edge": {
        "linewidth": float,  # Width of the edge line in points

        # Style of the edge line ('-' for solid, '--' for dashed, etc.). Matplotlib
        # allows for quite flexible syntax here.
        "linestyle": str | Any,

        # Color of the edge line (e.g. 'blue', '#0000FF'). This can be a floating
        # number (usually set on a per-edge basis) to use color mapping. In that case
        # the "cmap" property needs to be set too (see next option). In that case,
        # the min/max values of the floats are used as extremes for the colormap,
        # unless the "norm" option is used.
        "color": str | float | Any,

        # How to cap the edge line. Should be "butt" (default), "round", or "projecting".
        "capstyle": str,

        # Whether to leave any space between edge cap and vertex border. This is
        # in figure points and autoscales correctly with dpi.
        # DEPRECATED: "padding" is a synonym for this option, but it is deprecated.
        "shrink": float,

        # Matplotlib color map used to map floating numbers into RGBA colors. Only
        # used when the previous option "color" is set to floats.
        "cmap": str | matplotlib.colors.Colormap,

        # Matplotlib norm to connect color map and edge floating-point color values.
        # This is only used when colors are mapped through a color map. In that case,
        # this option lets the user finely control what floating number will be mapped
        # onto what color.
        "norm": tuple[float, float] | matplotlib.colors.Normalize,

        # Opacity of the edge (0.0 for fully transparent, 1.0 for fully opaque).
        # If a colormap is used and this option is also set, this opacity takes
        # priority and finally determines the transparency of the edges.
        "alpha": float,

        "depthshade": bool,  # Whether to shade the color based on depth (3D only)

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

        # xy offset of the edge coordinate in figure points. This is usually set on a
        # per-edge basis if specific edges are to be shifted laterally slightly. Large
        # offsets tend to not play well with the rest of the visualisation.
        "offset": tuple[float, float],

        # How much (in figure points) to offset parallel straight edges (i.e. in a
        # multigraph only) along the orthogonal direction to make them more visible.
        # To obtain a double-headed arrow effect, set this to zero. On the flip side,
        # you will not know how many parallel edges there are between those two nodes.
        "paralleloffset": float,

        # Edge ports a la Graphviz, which indicate the attack angle of an edge into
        # its origin (first element) and target (second element) node. "w" (for
        # "west") means exiting from/entering into the left side, "e" is the right
        # side, "n" the top, "s" the bottom. "nw" and similar two-letter combinations
        # are accepted and indicate 45 degree angles (in figure space). None means
        # that edge side is free (i.e. no special instruction there). This is almost
        # universally either unset or set on a per-edge basis to finely control the
        # appearance of the network (e.g. in organisational charts).
        "ports": tuple[str | None, str | None],

        # Edge waypoints. This option is usually set together with "ports" to
        # finely control the appearance of edges. For trees, this option is used
        # internally to create piecewise-straight connections. This option is
        # currently experimental, but you can try the following settings:
        # - xmidy0,xmidy1: Two waypoints with the mid-x and the 1st and 2nd y values.
        # - ymidx0,ymidx1: The xy swap of the previous option.
        # - x0y1: One waypoint, with x of the first point and y of the second point.
        # - x1y0: The xy swap of the previous option.
        # We are looking into ways to generalise this idea.
        "waypoints": str,

        # Edge arrow style for directed graphs
        "arrow": {
            # Arrow marker style. Currently supported:
            # |>
            # |/
            # |\\ (double slash needed to avoid character escaping)
            # >
            # <
            # >>
            # )>
            # )
            # (
            # ]
            # [
            # |
            # x (or X)
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

        # Edge label style
        "label": {

            # Whether to rotate edge labels to be horizontal (True) or to leave them
            # parallel to their edge (False). Some users find this boolean
            # unintuitive and interpret it the other way around, so think carefully.
            "rotate": bool,

            "color": str | Any,  # Color of the edge label (e.g. 'white', '#FFFFFF')
            "horizontalalignment": str,  # Horizontal alignment of the label ('left', 'center', 'right')
            "verticalalignment": str,  # Vertical alignment of the label ('top', 'center', 'bottom', 'baseline', 'center_baseline')
            "hpadding": float,  # Horizontal padding around the label
            "vpadding": float,  # Vertical padding around the label
            "bbox": dict,  # Bounding box properties for the label (see vertex labels)

        },

        ############################################################################
        # The following edge properties are only valid for trees via `iplotx.tree`
        "split": {
            # NOTE: This takes any properties of "edge" except for itself (i.e. no
            # "nested" split) and applies it to the last segment of split edges.
            "color": str | Any,  # Color of the split edge

            # ...
        },

    },

    # The following entry is used by networks ONLY (not trees as plotted by `iplotx.tree`)
    "grouping": {
        "facecolor": str | Any,  # Color of the grouping face (e.g. 'lightgray', '#D3D3D3')
        "edgecolor": str | Any,  # Color of the grouping edge (e.g. 'gray', '#808080')
        "linewidth": float,  # Width of the grouping edge in points
        "vertexpadding": float,  # Padding around the vertices in the grouping in points
    },

    ############################################################################
    # The following entries are used by trees ONLY (as plotted by `iplotx.tree`)
    "cascade": {
        # Whether to limit the cascade to the deepest descendant (False),
        # to the deepest leaf overall (True), or to include the leaf labels
        # as well "leaf_labels"
        "extend": bool | str,

        "facecolor": str | Any,  # Color of the cascade face
        "edgecolor": str | Any,  # Color of the cascade edge
        "alpha": float,  # Opacity of the cascade patch
    },

    # Leaf styles are currently only used for their labels
    "leaf": {
        # Whether leaf nodes and labels will be aligned to the deepest leaf.
        # If False, each leaf label will be at the depth of its leaf.
        "deep": bool,
        "label": {
            "color": str | Any,  # Color of the label text
            "hmargin": float,  # Horizontal offset (before rotation) off the leaf node
        }
    },

    # Leaf edge styles are used for the (usually dashed) lines connecting leaves
    # at less-than-max depth to deep labels (if used).
    "leafedge": {
        "color": str | Any,  # Color of the leaf edge (e.g. 'gray', '#808080')
        "linewidth": float,  # Width of the leaf edge in points
        "linestyle": str | Any,  # Style of the leaf edge line ('--' for dashed etc.)

        # Leaf edge labels.
        # NOTE: These lines will NOT exist for leaves that are set at the max depth
        # (at least one leaf will always be skipped). Therefore, using labels here
        # can be a little trickier than you might expect.
        "label": {
            "color": str | Any,  # Color of the leaf edge label (e.g. 'black')
            "hmargin": float,  # Horizontal offset (before rotation) off the leaf edge
            "vmargin": float,  # Vertical offset (before rotation) off the leaf edge
            # Whether to rotate the label to be horizontal (True) or parallel to the edge (False)
            "rotate": bool,
        },
    },

    # Layout options for trees
    "layout": {
        # Orientation of the tree. Accepted values depend on the tree layout:
        # - "horizontal" layout: "left" or "right"
        # - "vertical" layout: "ascending" or "descending"
        # - "radial" layout: "clockwise" or "counterclockwise"
        "orientation": str,
        "angular": bool,  # Whether to use an angular or waypoint-based layout
        "start": float,  # Starting angle for radial layouts (in degrees)
        "span": float,  # Angle span for radial layouts (in degrees)
    },
    ############################################################################
}
```

```{tip}
Most options can be "leaf rotated" to obtain a per-vertex, per-edge, per-label, or per-cascade look. See <project:../style.md> for more info.
```
