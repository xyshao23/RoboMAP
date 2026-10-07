"""Helpers for converting sparse annotations into heatmap parameters."""


def shape_aware_sigmas(
    width: float, height: float, base_sigma: float = 2.0
) -> tuple[float, float]:
    """Return bbox Gaussian sigmas with a fixed short-axis width.

    The camera-ready procedure sets the short axis to ``base_sigma`` and
    scales the long axis proportionally to the bounding-box aspect ratio.
    Width and height can be normalized or pixel-valued because only their
    ratio is used.
    """
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")
    if base_sigma <= 0:
        raise ValueError("base_sigma must be positive")

    short_axis = min(width, height)
    return (
        base_sigma * width / short_axis,
        base_sigma * height / short_axis,
    )
