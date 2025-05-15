from functools import wraps, partial
import matplotlib as mpl


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
