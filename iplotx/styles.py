from typing import Union, Sequence
from copy import deepcopy
from contextlib import contextmanager

default = {
    "vertex": {
        "size": 30,
        "color": "black",
        "marker": "o",
    },
    "edge": {
        "linewidth": 1.5,
        "color": "black",
        "curved": True,
        "tension": +1.5,
    },
    "arrow": {
        "marker": "|>",
        "width": 8,
        "color": "black",
    },
}


styles = {
    "default": default,
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

    def _update(style: dict, current: dict):
        for key, value in style.items():
            if key not in current:
                current[key] = value
                continue

            if isinstance(value, dict):
                _update(value, current[key])
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
                style = get_style(style)
            _update(style, current)


def reset():
    """Reset to default style."""
    current = deepcopy(styles["default"])


@contextmanager
def stylecontext(style: Union[str, dict, Sequence]):
    current = get_style()
    try:
        use(style)
        yield
    finally:
        use(current)
