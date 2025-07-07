# These properties are at the bottom of a style dictionary. Values corresponding
# to these keys are generally rotatable, with the exceptions below.
style_leaves = (
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
    "looptension",
    "loopmaxangle",
    "paralleloffset",
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
    "extend",
)

# These properties are not allowed to be rotated (global throughout the graph).
# This might change in the future as the API improves.
nonrotating_leaves = (
    "paralleloffset",
    "looptension",
    "loopmaxangle",
    "vertexpadding",
    "extend",
)
