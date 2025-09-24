import copy
from collections import defaultdict

from ..style.leaf_info import (
    style_leaves,
)

try:
    import networkx as nx
except ImportError:
    nx = None


def copy_with_deep_values(style):
    """Make a deep copy of the style dict but do not create copies of the keys."""
    # Defaultdict should be respected
    if hasattr(style, "default_factory"):
        newdict = defaultdict(lambda: style.default_factory())
    else:
        newdict = {}
    for key, value in style.items():
        if isinstance(value, dict):
            newdict[key] = copy_with_deep_values(value)
        else:
            newdict[key] = copy.copy(value)
    return newdict


def sanitize_leaves(style: dict):
    """Sanitize the leaves of a style dictionary.

    Parameters:
        style (dict): A style dictionary.
    Returns:
        None: The style dictionary is modified in place.
    """
    for key, value in style.items():
        if key in style_leaves:
            # Networkx has a few lazy data structures
            # TODO: move this code to provider
            if nx is not None:
                if isinstance(value, nx.classes.reportviews.NodeView):
                    style[key] = dict(value)
                elif isinstance(value, nx.classes.reportviews.EdgeViewABC):
                    style[key] = [v for *e, v in value]

        elif isinstance(value, dict):
            sanitize_leaves(value)


def update_style(new_style: dict, current_style: dict):
    """Update the current style with a new style.

    Parameters:
        new_style (dict): A new style dictionary.
        current_style (dict): The current style dictionary.
    Returns:
        None: The current style dictionary is modified in place.
    """
    for key, value in new_style.items():
        if key not in current_style:
            current_style[key] = value
            continue

        # Style non-leaves are either recurred into or deleted
        if key not in style_leaves:
            if isinstance(value, dict):
                update_style(value, current_style[key])
            elif value is None:
                del current_style[key]
            else:
                raise ValueError(
                    f"Setting non-leaf style value to a non-dict: {key}, {value}",
                )
        else:
            # Style leaves could be incomplete, ensure a sensible default
            if value is None:
                del current_style[key]
                continue

            if not isinstance(value, dict):
                current_style[key] = value
                continue

            if hasattr(value, "default_factory"):
                current_style[key] = value
                continue

            if hasattr(current_style[key], "default_factory"):
                default_value = current_style[key].default_factory()
            else:
                default_value = current_style[key]
            current_style[key] = defaultdict(
                lambda: default_value,
                value,
            )


def sanitize_ambiguous(style: dict):
    """Fix a few ambiguous cases in the style dict.

    Parameters:
        style (dict): A style dictionary. This must be unflattened beforehand.
    Returns:
        None: The style dictionary is modified in place.

    NOTE: This function exists by design, not accident. It is useful for purposeful
    (e.g. node vs vertex) or historical (e.g. edge_padding vs edge_shrink) reasons.
    Either way, it softens the user experience without complicating the API.
    """

    # Accept "node" style as "vertex" style for user flexibility
    if "node" in style:
        style_node = style.pop("node")
        if "vertex" not in style:
            style["vertex"] = style_node
        else:
            # "node" style applies on TOP of "vertex" style
            update_style(style_node, style["vertex"])

    # NOTE: Deprecate edge_padding for edge_shrink
    for edgekey in ["edge", "leafedge"]:
        if "padding" in style.get(edgekey, {}):
            # shrink takes over
            if "shrink" in style[edgekey]:
                del style[edgekey]["padding"]
            else:
                style[edgekey]["shrink"] = style[edgekey].pop("padding")
