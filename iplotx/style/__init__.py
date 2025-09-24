"""
Main style module for iplotx.
"""

from typing import (
    Any,
    Iterable,
    Optional,
    Sequence,
)
from collections.abc import Hashable
from contextlib import contextmanager
import numpy as np
import pandas as pd

from ..utils.style import (
    copy_with_deep_values,
    sanitize_leaves,
    update_style,
    sanitize_ambiguous,
)
from .library import style_library
from .leaf_info import (
    style_leaves,
    nonrotating_leaves,
)


# Prepopulate default style, it's used later as a backbone for everything else
default = style_library["default"]
styles = {
    "default": default,
}


current = copy_with_deep_values(styles["default"])


def get_style(name: str = "", *args) -> dict[str, Any]:
    """Get a *deep copy* of the chosen style.

    Parameters:
        name: The name of the style to get. If empty, the current style is returned.
            Substyles can be obtained by using a dot notation, e.g. "default.vertex".
            If "name" starts with a dot, it means a substyle of the current style.
        *args: A single argument is accepted. If present, this value (usually a
            dictionary) is returned if the queried style is not found. For example,
            get_style(".nonexistent") raises an Exception but
            get_style("nonexistent", {}) does not, returning an empty dict instead.
    Returns:
        The requected style or substyle.

    NOTE: The deep copy is a little different from standard deep copies. Here, keys
        (which need to be hashables) are never copied, but values are. This can be
        useful for hashables that change hash upon copying, such as Biopython's
        tree nodes.
    """
    if len(args) > 1:
        raise ValueError("get_style() accepts at most one additional argument.")

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
            # which will not fail unless the user tries to enter it
            elif namei not in style_leaves:
                style = {}
            elif len(args) > 0:
                return args[0]
            else:
                raise KeyError(f"Style not found: {name}")

    style = copy_with_deep_values(style)
    return style


def merge_styles(
    styles: Sequence[str | dict[str, Any]] | Iterable[str | dict[str, Any]],
) -> dict[str, Any]:
    """Merge a sequence of styles into a single one.

    Parameters:
        styles: Sequence (list, tuple, etc.) of styles, each either the name of an internal
            style or a dict-like with custom properties.
    Returns:
        The composite style as a dict.
    """

    merged = {}
    for style in styles:
        if isinstance(style, str):
            style = get_style(style)
        else:
            sanitize_leaves(style)
            unflatten_style(style)
            sanitize_ambiguous(style)
        update_style(style, merged)

    return merged


# The following is inspired by matplotlib's style library
# https://github.com/matplotlib/matplotlib/blob/v3.10.3/lib/matplotlib/style/core.py#L45
def use(
    style: Optional[
        str | dict[str, Any] | Sequence[str | dict[str, Any]] | Iterable[str | dict[str, Any]]
    ] = None,
    **kwargs,
):
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

    styles = []
    if isinstance(style, (dict, str)):
        styles.append(style)
    elif style is not None:
        styles.extend(list(style))
    if kwargs:
        styles.append(kwargs)

    # Discard empty styles for speed
    styles = [style for style in styles if style]

    if len(styles) == 0:
        return

    # If the first style is a string (internal style), apply it cold. If it's a
    # dict, apply it hot (on top of the current style(. Any style after the first
    # is always applied hot - otherwise it would invalidate previous styles.
    if not isinstance(styles[0], str):
        # hot insertion on top of current
        styles.insert(0, current)

    old_style = copy_with_deep_values(current)
    try:
        current = merge_styles(styles)
    except:
        current = old_style
        raise


def reset() -> None:
    """Reset to default style."""
    global current
    current = copy_with_deep_values(styles["default"])


@contextmanager
def context(
    style: Optional[
        str | dict[str, Any] | Sequence[str | dict[str, Any]] | Iterable[str | dict[str, Any]]
    ] = None,
    **kwargs,
):
    """Create a style context for iplotx.

    Parameters:
        style: A single style specification or a list of style specifications, which are then
            applied in order. Each style can be a string (for an existing style) or a dictionary
            with the elements that are to change.
        **kwargs: Additional style changes to be applied at the end of all styles.

    Yields:
        A context manager that applies the style and reverts it back to the previous one upon exit.
    """
    current = get_style()
    try:
        use(style, **kwargs)
        yield
    finally:
        use(["default", current])


def unflatten_style(
    style_flat: dict[str, str | dict | int | float],
) -> None:
    """Convert a flat or semi-flat style into a fully structured dict.

    Parameters:
        style_flat: A flat dictionary where keys may contain underscores, which are taken to signify
            subdictionaries.

    NOTE: The dict is changed *in place*.

    Example:
        >>> style = {'vertex_size': 20}
        >>> unflatten_style(style)
        >>> print(style)
        {'vertex': {'size': 20}}
    """

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
            if isinstance(value, dict) and (key not in style_leaves):
                _inner(value)

    # top-level adjustments
    if "zorder" in style_flat:
        style_flat["network_zorder"] = style_flat["grouping_zorder"] = style_flat.pop("zorder")

    # Begin recursion
    _inner(style_flat)


def rotate_style(
    style: dict[str, Any],
    index: Optional[int] = None,
    key: Optional[Hashable] = None,
    key2: Optional[Hashable] = None,
    props: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Rotate leaves of a style for a certain index or key.

    Parameters:
        style: The style to rotate.
        index: The integer to rotate the style leaves into.
        key: For dict-like leaves (e.g. vertex properties specified as a dict-like object over the
            vertices themselves), the key to use for rotation (e.g. the vertex itself).
        key2: For dict-like leaves, a backup key in case the first key fails. If this is None
            or also a failure (i.e. KeyError), default to the empty type constructor for the
            first value of the dict-like style leaf.
        props: The properties to rotate, usually all leaf properties.

    Returns:
        A style with rotated leaves, which describes the properties of a single element (e.g.
        vertex).

    Example:
        >>> style = {'vertex': {'size': [10, 20]}}
        >>> rotate_style(style, index=0)
        {'vertex': {'size': 10}}
    """
    if (index is None) and (key is None):
        raise ValueError("At least one of 'index' or 'key' must be provided to rotate_style.")

    if props is None:
        props = tuple(prop for prop in style_leaves if prop not in nonrotating_leaves)

    style = copy_with_deep_values(style)

    for prop in props:
        val = style.get(prop, None)
        if val is None:
            continue
        # Try integer indexing for ordered types
        if (index is not None) and isinstance(val, (tuple, list, np.ndarray, pd.Index, pd.Series)):
            # NOTE: cannot cast to ndarray because rotation might involve
            # cross-type lists (e.g. ["none", False])
            style[prop] = list(val)[index % len(val)]
        # Try key indexing for unordered, dict-like types
        if (
            (key is not None)
            and (not isinstance(val, (str, tuple, list, np.ndarray)))
            and hasattr(val, "__getitem__")
        ):
            # If only a subset of keys is provided, try the default value a la
            # defaultdict. If that fails, use an empty constructor
            if key in val:
                newval = val[key]
            elif key2 is not None and (key2 in val):
                newval = val[key2]
            else:
                try:
                    newval = val[key]
                except KeyError:
                    valtype = type(next(iter(val.values())))
                    newval = valtype()
            style[prop] = newval

    return style


def add_style(name: str, style: dict[str, Any]) -> None:
    """Add a style to the default dictionary of styles.

    Parameters:
        name: The name of the style to add.
        style: A dictionary with the style properties to add.
    """
    with context(["default", style]):
        styles[name] = get_style()


# Populate style library (default is already there)
for name, style in style_library.items():
    if name != "default":
        add_style(name, style)
        del name, style
