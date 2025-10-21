"""
Unrooted tree layout for iplotx.
"""


def _equalangle_tree_layout(
    orientation: str = "right",
    start: float = 180,
    span: float = 360,
    **kwargs,
):
    """Equal angle unrooted tree layout.

    Parameters:
        orientation: Whether the layout fans out towards the right (clockwise) or left
            (anticlockwise).
        start: The starting angle in degrees, default is -180 (left).
        span: The angular span in degrees, default is 360 (full circle). When this is
            360, it leaves a small gap at the end to ensure the first and last leaf
            are not overlapping.
    Returns:
        A dictionary with the layout.

    Reference: "Inferring Phylogenies" by Joseph Felsenstein, ggtree.
    """
    return NotImplementedError("Equal-angle tree layout is not yet implemented.")


def _daylight_tree_layout(
    orientation: str = "right",
    start: float = 180,
    span: float = 360,
    **kwargs,
):
    """Daylight unrooted tree layout.

    Parameters:
        orientation: Whether the layout fans out towards the right (clockwise) or left
            (anticlockwise).
        start: The starting angle in degrees, default is -180 (left).
        span: The angular span in degrees, default is 360 (full circle). When this is
            360, it leaves a small gap at the end to ensure the first and last leaf
            are not overlapping.
    Returns:
        A dictionary with the layout.

    Reference: "Inferring Phylogenies" by Joseph Felsenstein, ggtree.
    """
    return NotImplementedError("Equal-angle tree layout is not yet implemented.")
