# See also this link for the general answer (using scipy to compute coefficients):
# https://stackoverflow.com/questions/12643079/b%C3%A9zier-curve-fitting-with-scipy
def _evaluate_squared_bezier(points, t):
    """Evaluate a squared Bezier curve at t."""
    p0, p1, p2 = points
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def _evaluate_cubic_bezier(points, t):
    """Evaluate a cubic Bezier curve at t."""
    p0, p1, p2, p3 = points
    return (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t**2 * p2
        + t**3 * p3
    )
