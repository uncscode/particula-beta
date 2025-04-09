"""
General volume to length conversion
"""

def get_length_from_volume(
    volume: float,
    shape: str,
    dimension: str = "radius",
) -> float:
    """
    Calculates a length (radius or diameter) from a given volume for a given shape.

    Args:
        volume: The volume of the shape.
        shape: The shape type (e.g., "sphere").
        dimension: The dimension to return – "radius" or "diameter".

    Returns:
        The requested length (radius or diameter) as a float.
    """
    if shape.lower() == "sphere":
        # Volume of a sphere = (4/3) * pi * r^3
        import math
        radius = ((3 * volume) / (4 * math.pi)) ** (1 / 3)
        return radius if dimension == "radius" else 2.0 * radius

    # Add other shapes here if needed.
    raise ValueError(f"Unsupported shape: {shape}")


def get_volume_from_length(
    length: float,
    shape: str,
    dimension: str = "radius",
) -> float:
    """
    Calculates a volume from a given length (radius or diameter) for a given shape.

    Args:
        length: The length specifying the shape's size (radius or diameter).
        shape: The shape type (e.g., "sphere").
        dimension: Specifies whether the input is a "radius" or a "diameter".

    Returns:
        The corresponding volume as a float.
    """
    if shape.lower() == "sphere":
        import math
        radius = length if dimension == "radius" else length / 2.0
        return (4.0 / 3.0) * math.pi * (radius ** 3)

    # Add other shapes here if needed.
    raise ValueError(f"Unsupported shape: {shape}")
"""

