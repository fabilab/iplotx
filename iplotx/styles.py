from typing import Union, Sequence
from copy import deepcopy
from contextlib import contextmanager
import numpy as np
import pandas as pd


default = {
    "vertex": {
        "size": 20,
        "facecolor": "black",
        "marker": "o",
        "label": {
            "horizontalalignment": "center",
            "verticalalignment": "center",
            "hpadding": 18,
            "vpadding": 12,
        },
    },
    "edge": {
        "linewidth": 1.5,
        "linestyle": "-",
        "color": "black",
        "curved": False,
        "offset": 3,
        "tension": 1,
        "label": {
            "horizontalalignment": "center",
            "verticalalignment": "center",
        },
    },
    "arrow": {
        "marker": "|>",
        "width": 8,
        "color": "black",
    },
    "grouping": {
        "facecolor": ["grey", "steelblue", "tomato"],
        "edgecolor": "black",
        "linewidth": 1.5,
        "alpha": 0.5,
        "vertexpadding": 25,
    },
}

hollow = deepcopy(default)
hollow["vertex"]["color"] = None
hollow["vertex"]["facecolor"] = "none"
hollow["vertex"]["edgecolor"] = "black"
hollow["vertex"]["linewidth"] = 1.5
hollow["vertex"]["marker"] = "r"
hollow["vertex"]["size"] = "label"


styles = {
    "default": default,
    "hollow": hollow,
}


stylename = "default"


current = deepcopy(styles["default"])


def get_stylename():
    """Return the name of the current iplotx style."""
    return str(stylename)


def get_style(name: str = ""):
    namelist = name.split(".")
    style = styles
    for i, namei in enumerate(namelist):
        if (i == 0) and (namei == ""):
            style = current
        else:
            try:
                style = style[namei]
            except KeyError:
                raise KeyError(f"Style not found: {name}")

    style = deepcopy(style)
    return style


# The following is inspired by matplotlib's style library
# https://github.com/matplotlib/matplotlib/blob/v3.10.3/lib/matplotlib/style/core.py#L45
def use(style: Union[str, dict, Sequence]):
    """Use iplotx style setting for a style specification.

    The style name of 'default' is reserved for reverting back to
    the default style settings.

    Parameters:
        style: A style specification, currently either a name of an existing style
            or a dict with specific parts of the style to override. The string
            "default" resets the style to the default one. If this is a sequence,
            each style is applied in order.
    """
    global current

    def _update(style: dict, current: dict):
        for key, value in style.items():
            if key not in current:
                current[key] = value
                continue

            if isinstance(value, dict):
                _update(value, current[key])
            elif value is None:
                del current[key]
            else:
                current[key] = value

    if isinstance(style, (dict, str)):
        styles = [style]
    else:
        styles = style

    for style in styles:
        if style == "default":
            reset()
        else:
            if isinstance(style, str):
                current = get_style(style)
            else:
                _update(style, current)


def reset():
    """Reset to default style."""
    global current
    current = deepcopy(styles["default"])


@contextmanager
def stylecontext(style: Union[str, dict, Sequence]):
    current = get_style()
    try:
        use(style)
        yield
    finally:
        use(current)


def rotate_style(
    style,
    i,
    props=("edgecolor", "facecolor", "linewidth", "linestyle", "alpha", "zorder"),
):
    style = deepcopy(style)

    for prop in props:
        val = style.get(prop, None)
        if val is None:
            continue
        if isinstance(val, (tuple, list, np.ndarray, pd.Index, pd.Series)):
            style[prop] = val[i % len(val)]

    return style
