from functools import wraps, partial
from math import atan2
import numpy as np
import matplotlib as mpl

from .geometry import (
    _evaluate_squared_bezier,
    _evaluate_cubic_bezier,
)


# NOTE: https://github.com/networkx/grave/blob/main/grave/grave.py
def _stale_wrapper(func):
    """Decorator to manage artist state."""

    @wraps(func)
    def inner(self, *args, **kwargs):
        try:
            func(self, *args, **kwargs)
        finally:
            self.stale = False

    return inner


def _forwarder(forwards, cls=None):
    """Decorator to forward specific methods to Artist children."""
    if cls is None:
        return partial(_forwarder, forwards)

    def make_forward(name):
        def method(self, *args, **kwargs):
            """Each decorated method is called on the decorated class and then, nonrecursively, on all children."""
            ret = getattr(cls.mro()[1], name)(self, *args, **kwargs)
            for c in self.get_children():
                getattr(c, name)(*args, **kwargs)
            return ret

        return method

    for f in forwards:
        method = make_forward(f)
        method.__name__ = f
        method.__doc__ = "broadcasts {} to children".format(f)
        setattr(cls, f, method)

    return cls


def _additional_set_methods(attributes, cls=None):
    """Decorator to add specific set methods for children properties.

    This is useful to autogenerate methods a la set_<key>(value), for
    instance set_alpha(value). It works by delegating to set(alpha=value).

    Overall, this is a minor tweak compared to the previous decorator.
    """
    if cls is None:
        return partial(_additional_set_methods, attributes)

    def make_setter(name):
        def method(self, value):
            self.set(**{name: value})

        return method

    for attr in attributes:
        desc = attr.replace("_", " ")
        method = make_setter(attr)
        method.__name__ = f"set_{attr}"
        method.__doc__ = f"Set {desc}."
        setattr(cls, f"set_{attr}", method)

    return cls


def _get_label_width_height(text, hpadding=18, vpadding=12, dpi=72.0, **kwargs):
    """Get the bounding box size for a text with certain properties.

    Parameters:
        text: The text to measure.
        hpadding: Horizontal padding to add to the width.
        vpadding: Vertical padding to add to the height.
        **kwargs: Additional keyword arguments for text properties. "fontsize" is accepted,
            as is "size". Many other properties are not used and will raise and exception.

    Returns:
        A tuple (width, height) representing the size of the text bounding box. Because
        some text properties such as weight are not taken into account, ths function is not
        very accurate. Yet, it is often good enough and easier to implement than a careful
        orchestration of Figure.draw_without_rendering.
    """
    if "fontsize" in kwargs:
        kwargs["size"] = kwargs.pop("fontsize")
    forbidden_props = [
        "horizontalalignment",
        "verticalalignment",
        "ha",
        "va",
        "color",
        "edgecolor",
        "facecolor",
    ]
    for prop in forbidden_props:
        if prop in kwargs:
            del kwargs[prop]

    path = mpl.textpath.TextPath((0, 0), text, **kwargs)
    boundingbox = path.get_extents()
    width = boundingbox.width
    height = boundingbox.height

    # Scaling with font size appears broken... try to patch it up linearly here, even though we
    # know it does not work terribly accurately
    width *= kwargs.get("size", 12) / 12.0
    height *= kwargs.get("size", 12) / 12.0

    width += hpadding
    height += vpadding

    # Scale by dpi
    width *= dpi / 72.0
    height *= dpi / 72.0

    return (width, height)


def _compute_mid_coord_and_rot(path, trans):
    """Compute mid point of an edge, straight or curved."""
    # Distinguish between straight and curved paths
    if path.codes[-1] == mpl.path.Path.LINETO:
        coord = path.vertices.mean(axis=0)
        vtr = trans(path.vertices)
        rot = atan2(
            vtr[-1, 1] - vtr[0, 1],
            vtr[-1, 0] - vtr[0, 0],
        )

    # Cubic Bezier
    elif path.codes[-1] == mpl.path.Path.CURVE4:
        coord = _evaluate_cubic_bezier(path.vertices, 0.5)
        # TODO:
        rot = 0

    # Square Bezier
    elif path.codes[-1] == mpl.path.Path.CURVE3:
        coord = _evaluate_squared_bezier(path.vertices, 0.5)
        # TODO:
        rot = 0

    else:
        raise ValueError(
            "Curve type not straight and not squared/cubic Bezier, cannot compute mid point."
        )

    return coord, rot


def _build_cmap_fun(values, cmap, norm=None):
    """Map colormap on top of numerical values."""
    cmap = mpl.cm._ensure_cmap(cmap)

    if np.isscalar(values):
        values = [values]

    if isinstance(values, dict):
        values = np.array(list(values.values()))

    if norm is None:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    return lambda x: cmap(norm(x))
