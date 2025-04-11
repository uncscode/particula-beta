"""
General volume to length conversion
"""

from typing import Union
from numpy.typing import NDArray
import numpy as np


def get_length_from_volume(
    volume: Union[float, NDArray[np.float64]],
    dimension: str = "radius",
) -> float:
    """
    Calculates a length (radius or diameter) from a given volume for a sphere.

    Args:
        volume: The volume of the shape.
        dimension: The dimension to return – "radius" or "diameter".

    Returns:
        The requested length (radius or diameter) as a float.
    """
    # Volume of a sphere = (4/3) * pi * r^3
    radius = ((3 * volume) / (4 * np.pi)) ** (1 / 3)
    return radius if dimension == "radius" else 2.0 * radius


def get_volume_from_length(
    length: Union[float, NDArray[np.float64]],
    dimension: str = "radius",
) -> float:
    """
    Calculates a volume from a given length (radius or diameter) for a sphere.

    Args:
        length: The length specifying the shape's size (radius or diameter).
        dimension: Specifies whether the input is a "radius" or a "diameter".

    Returns:
        The corresponding volume as a float.
    """
    radius = length if dimension == "radius" else length / 2.0
    return (4.0 / 3.0) * np.pi * (radius**3)
