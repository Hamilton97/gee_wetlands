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