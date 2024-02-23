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
    def wrapper(img: ee.Image) -> ee.Image:
        return img.addBands(img.normalizedDifference([nir, red]).rename('NDVI'))
    return wrapper