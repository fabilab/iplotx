from copy import deepcopy

default = {
    "vertex": {
        "size": 30,
        "color": "black",
        "marker": "o",
    },
    "edge": {
        "linewidth": 1.5,
        "color": "black",
    },
}

styles = {
    "default": default,
}


stylename = "default"


def get_stylename():
    """Return the name of the current iplotx style."""
    return str(stylename)


def get_style(name : str = "default"):
    namelist = name.split(".")
    style = styles
    for namei in namelist:
        try:
            style = style[namei]
        except KeyError:
            raise KeyError(f"Style not found: {name}")

    style = deepcopy(style)
    return style
