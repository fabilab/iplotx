import copy
from collections import defaultdict


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
