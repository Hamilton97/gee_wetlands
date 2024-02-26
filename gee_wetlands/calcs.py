"""
TODO: Implement all standard calcs as helper functions
    - NDVI
    - SAVI
    - Tasseled Cap
    - Ratio
    - Phase
    - Amplitude
"""
from typing import Callable
import ee


def compute_ndvi(nir: str, red: str) -> Callable:
    """
    Computes the Normalized Difference Vegetation Index (NDVI) for an image.
    
    Args:
        nir (str): The name of the near-infrared band.
        red (str): The name of the red band.
    
    Returns:
        Callable: A function that takes an ee.Image as input and returns an ee.Image with the NDVI band added.
    """
    def wrapper(img: ee.Image) -> ee.Image:
        return img.addBands(img.normalizedDifference([nir, red]).rename('NDVI'))
    return wrapper


def compute_savi(nir: str, red: str, l: float = 0.5) -> Callable:
    """
    Computes the Soil Adjusted Vegetation Index (SAVI) for an image.

    Args:
        nir (str): The name of the near-infrared band.
        red (str): The name of the red band.
        l (float, optional): The SAVI adjustment factor. Defaults to 0.5.

    Returns:
        Callable: A function that takes an ee.Image as input and returns the image with the SAVI band added.
    """
    
    def wrapper(img: ee.Image) -> ee.Image:
        return img.addBands(img.expression(
            "(1 + L) * (NIR - RED) / (NIR + RED + L)",
            {
                "NIR": img.select(nir),
                "RED": img.select(red),
                "L": l,
            },
        ).rename('SAVI'))
    
    return wrapper


def compute_tasseled_cap(*args) -> Callable:
    """
    Compute the Tasseled Cap transformation for an image.

    Args:
        *args: The band names to be used in the transformation.

    Returns:
        A wrapper function that applies the Tasseled Cap transformation to an image.

    Example:
        tasseled_cap = compute_tasseled_cap('g', 'b', 'r', 'nir', 'swir1', 'swir2')\n
        transformed_image = tasseled_cap(image)
    """
    g, b, r, nir, swir1, swir2 = args

    def wrapper(img: ee.Image) -> ee.Image:
        tmp = img.select(g, b, r, nir, swir1, swir2)
        co_array = [
            [0.3037, 0.2793, 0.4743, 0.5585, 0.5082, 0.1863],
            [-0.2848, -0.2435, -0.5436, 0.7243, 0.0840, -0.1800],
            [0.1509, 0.1973, 0.3279, 0.3406, -0.7112, -0.4572],
        ]

        co = ee.Array(co_array)

        arrayImage1D = tmp.toArray()
        arrayImage2D = arrayImage1D.toArray(1)

        components_image = (
            ee.Image(co)
            .matrixMultiply(arrayImage2D)
            .arrayProject([0])
            .arrayFlatten([["brightness", "greenness", "wetness"]])
        )

        return img.addBands(components_image)

    return wrapper


def compute_ratio(numerator: str, denominator: str) -> Callable:
    name = f"{numerator}_{denominator}"
    def wrapper(img: ee.Image) -> ee.Image:
        return img.addBands(img.select(numerator).divide(img.select(denominator)).rename(name))
    return wrapper