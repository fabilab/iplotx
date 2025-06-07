from typing import Optional, Sequence, Hashable
from copy import deepcopy
from contextlib import contextmanager
import numpy as np
import pandas as pd

from .importing import igraph, networkx


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
    "lloopmaxangle",
    "rotate",
    "marker",
)


default = {
    "vertex": {
        "size": 20,
        "facecolor": "black",
        "marker": "o",
        "label": {
            "color": "white",
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
        "looptension": 4,
        "loopmaxangle": 60,
        "label": {
            "horizontalalignment": "center",
            "verticalalignment": "center",
            "rotate": False,
            "bbox": {
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "none",
            },
        },
    },
    "arrow": {
        "marker": "|>",
        "width": 8,
    },
    "grouping": {
        "facecolor": ["grey", "steelblue", "tomato"],
        "edgecolor": "black",
        "linewidth": 1.5,
        "alpha": 0.5,
        "vertexpadding": 18,
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
            if namei in style:
                style = style[namei]
            # NOTE: if asking for a nonexistent, non-leaf style
            # give the benefit of the doubt and set an empty dict
            # which will not fail unless the uder tries to enter it
            elif namei not in style_leaves:
                style = {}
            else:
                raise KeyError(f"Style not found: {name}")

    style = deepcopy(style)
    return style


# The following is inspired by matplotlib's style library
# https://github.com/matplotlib/matplotlib/blob/v3.10.3/lib/matplotlib/style/core.py#L45
def use(style: Optional[str | dict | Sequence] = None, **kwargs):
    """Use iplotx style setting for a style specification.

    The style name of 'default' is reserved for reverting back to
    the default style settings.

    Parameters:
        style: A style specification, currently either a name of an existing style
            or a dict with specific parts of the style to override. The string
            "default" resets the style to the default one. If this is a sequence,
            each style is applied in order.
        **kwargs: Additional style changes to be applied at the end of any style.
    """
    global current

    def _sanitize_leaves(style: dict):
        for key, value in style.items():
            if key in style_leaves:
                if networkx is not None:
                    if isinstance(value, networkx.classes.reportviews.NodeView):
                        style[key] = dict(value)
                    elif isinstance(value, networkx.classes.reportviews.EdgeViewABC):
                        style[key] = [v for *e, v in value]
            elif isinstance(value, dict):
                _sanitize_leaves(value)

    def _update(style: dict, current: dict):
        for key, value in style.items():
            if key not in current:
                current[key] = value
                continue

            # Style leaves are by definition not to be recurred into
            if isinstance(value, dict) and (key not in style_leaves):
                _update(value, current[key])
            elif value is None:
                del current[key]
            else:
                current[key] = value

    old_style = deepcopy(current)

    try:
        if style is None:
            styles = []
        elif isinstance(style, (dict, str)):
            styles = [style]
        else:
            styles = list(style)

        if kwargs:
            styles.append(kwargs)

        for style in styles:
            if style == "default":
                reset()
            else:
                if isinstance(style, str):
                    current = get_style(style)
                else:
                    _sanitize_leaves(style)
                    unflatten_style(style)
                    _update(style, current)
    except:
        current = old_style
        raise


def reset():
    """Reset to default style."""
    global current
    current = deepcopy(styles["default"])


@contextmanager
def context(style: Optional[str | dict | Sequence] = None, **kwargs):
    current = get_style()
    try:
        use(style, **kwargs)
        yield
    finally:
        use(["default", current])


def unflatten_style(
    style_flat: dict[str, str | dict | int | float],
):
    """Convert a flat or semi-flat style into a fully structured dict."""

    def _inner(style_flat: dict):
        keys = list(style_flat.keys())

        for key in keys:
            if "_" not in key:
                continue

            keyhead, keytail = key.split("_", 1)
            value = style_flat.pop(key)
            if keyhead not in style_flat:
                style_flat[keyhead] = {
                    keytail: value,
                }
            else:
                style_flat[keyhead][keytail] = value

        for key, value in style_flat.items():
            if isinstance(value, dict) and (value not in style_leaves):
                _inner(value)

    # top-level adjustments
    if "zorder" in style_flat:
        style_flat["network_zorder"] = style_flat["grouping_zorder"] = style_flat.pop(
            "zorder"
        )

    # Begin recursion
    _inner(style_flat)


def rotate_style(
    style,
    index: Optional[int] = None,
    id: Optional[Hashable] = None,
    props=style_leaves,
):
    if (index is None) and (id is None):
        raise ValueError(
            "At least one of 'index' or 'id' must be provided to rotate_style."
        )

    style = deepcopy(style)

    for prop in props:
        val = style.get(prop, None)
        if val is None:
            continue
        if (index is not None) and isinstance(
            val, (tuple, list, np.ndarray, pd.Index, pd.Series)
        ):
            style[prop] = np.asarray(val)[index % len(val)]
        if (
            (id is not None)
            and (not isinstance(val, (str, tuple, list, np.ndarray)))
            and hasattr(val, "__getitem__")
        ):
            style[prop] = val[id]

    return style
