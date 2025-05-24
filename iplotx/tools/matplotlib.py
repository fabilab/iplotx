from functools import wraps, partial
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
    """Decorator to add specific set methods for children properties."""
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


# FIXME: this method appears quite inconsistent, would be better to improve.
# The issue is that to really know the size of a label on screen, we need to
# render it first. Therefore, we should render the labels, then render the
# vertices. Leaving for now, since this can be styled manually which covers
# many use cases.
def _get_label_width_height(text, hpadding=18, vpadding=12, **kwargs):
    """Get the bounding box size for a text with certain properties."""
    forbidden_props = ["horizontalalignment", "verticalalignment", "ha", "va"]
    for prop in forbidden_props:
        if prop in kwargs:
            del kwargs[prop]

    path = mpl.textpath.TextPath((0, 0), text, **kwargs)
    boundingbox = path.get_extents()
    width = boundingbox.width + hpadding
    height = boundingbox.height + vpadding
    return (width, height)


def _compute_mid_coord(path):
    """Compute mid point of an edge, straight or curved."""
    # Distinguish between straight and curved paths
    if path.codes[-1] == mpl.path.Path.LINETO:
        return path.vertices.mean(axis=0)

    # Cubic Bezier
    if path.codes[-1] == mpl.path.Path.CURVE4:
        return _evaluate_cubic_bezier(path.vertices, 0.5)

    # Square Bezier
    if path.codes[-1] == mpl.path.Path.CURVE3:
        return _evaluate_squared_bezier(path.vertices, 0.5)

    raise ValueError(
        "Curve type not straight and not squared/cubic Bezier, cannot compute mid point."
    )


def _compute_group_path_with_vertex_padding(
    points,
    transform,
    vertexpadding=10,
):
    """Offset path for a group based on vertex padding.

    At the input, the structure is [v1, v1, v1, v2, v2, v2, ...]
    """

    # Transform into figure coordinates
    trans = transform.transform
    trans_inv = transform.inverted().transform
    points = trans(points)

    npoints = len(points) // 3
    vprev = points[-1]
    mprev = atan2(points[0, 1] - vprev[1], points[0, 0] - vprev[0])
    for i, vcur in enumerate(points[::3]):
        vnext = points[(i + 1) * 3]
        mnext = atan2(vnext[1] - vcur[1], vnext[0] - vcur[0])

        mprev_orth = -1 / mprev
        points[i * 3] = vcur + vertexpadding * mprev_orth

        vprev = vcur
        mprev = mnext

    points = trans_inv(points)
    return points
