# These properties are at the bottom of a style dictionary. Values corresponding
# to these keys are rotatable.
rotating_leaves = (
    "cmap",
    "color",
    "size",
    "edgecolor",
    "facecolor",
    "linewidth",
    "linestyle",
    "alpha",
    "zorder",
    "tension",
    "offset",
    "rotate",
    "marker",
    "waypoints",
    "horizontalalignment",
    "verticalalignment",
    "boxstyle",
    "hpadding",
    "vpadding",
    "hmargin",
    "vmargin",
    "ports",
    "width",
    "height",
    "shrink",
    # DEPRECATED
    "padding",
)

# These properties are also terminal style properties, but they cannot be rotated.
# This might change in the future as the API improves.
nonrotating_leaves = (
    "paralleloffset",
    "looptension",
    "loopmaxangle",
    "vertexpadding",
    "extend",
    "deep",
    "angular",
    "curved",
    "capstyle",
    "depthshade",
)

# Union of all style leaves (rotating and nonrotating)
style_leaves = tuple(list(rotating_leaves) + list(nonrotating_leaves))
