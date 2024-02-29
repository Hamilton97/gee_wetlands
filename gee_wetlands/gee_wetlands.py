from __future__ import annotations
from typing import Callable, Any

import ee



def _preprocessing(self, *args) -> ee.ImageCollection:
    """
    Preprocesses the input arguments and returns an ImageCollection filtered by the specified AOI and date range.

    Args:
        *args: Variable length argument list containing the AOI, start date, and end date.

    Returns:
        ee.ImageCollection: The filtered ImageCollection.

    """
    aoi, start_date, end_date = args
    return self.filterBounds(aoi).filterDate(start_date, end_date)


def composite(self) -> ee.Image:
    return self.median()


def __add__(self, other: ee.ImageCollection) -> ee.ImageCollection:
    if not isinstance(other, ee.IamgeCollection):
        raise TypeError("Other must be a Image Collection")
    return self.merge(other)


ee.ImageCollection._preprocessing = _preprocessing
ee.ImageCollection.composite = composite
ee.ImageCollection.__add__ = __add__
 

###########
# Optical #
###########


# Functions ----------------------------------------------------------------------------------------

def add_ndvi(nir: str, red: str) -> Callable:
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


def add_savi(nir: str, red: str, l: float = 0.5) -> Callable:
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


def add_tasseled_cap(*args) -> Callable:
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


def mask_s2_clouds(img: ee.Image) -> ee.Image:
    """
    Masks out clouds and cirrus in a Sentinel-2 image.

    Args:
        img (ee.Image): The Sentinel-2 image to mask.

    Returns:
        ee.Image: The masked Sentinel-2 image.
    """
    qa = img.select('QA60')

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloud_bit_mask)
        .eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return img.updateMask(mask)


# Datasets -----------------------------------------------------------------------------------------

class Sentinel2TOA(ee.ImageCollection):
    """
    A class representing the Sentinel-2 Top of Atmosphere (TOA) image collection in Google Earth Engine.
    """

    BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    
    def __init__(self):
        super().__init__("COPERNICUS/S2_HARMONIZED")

    def preprocess(self, aoi, start, end, cldy_per: float = 20.0):
        """
        Preprocesses the Sentinel-2 TOA image collection by filtering based on cloud cover percentage and selecting specific bands.

        Args:
            aoi: The area of interest (AOI) to filter the image collection.
            start: The start date of the time range to filter the image collection.
            end: The end date of the time range to filter the image collection.
            cldy_per: The maximum allowable cloud cover percentage. Defaults to 20.0.

        Returns:
            The preprocessed Sentinel-2 TOA image collection.
        """
        return (
            self._preprocessing(aoi, start, end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cldy_per))
            .map(self.mask_clouds)
            .select(self.BANDS)
        )

    @staticmethod
    def mask_clouds(img: ee.Image) -> ee.Image:
        """
        Masks clouds and cirrus in a Sentinel-2 TOA image.

        Args:
            img: The Sentinel-2 TOA image.

        Returns:
            The masked Sentinel-2 TOA image.
        """
        qa = img.select('QA60')

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return img.updateMask(mask)


#########
# RADAR #
#########


# Functions ----------------------------------------------------------------------------------------


def add_ratio(numerator: str, denominator: str) -> Callable:
    """
    Adds a ratio band to an image.

    Args:
        numerator (str): The name of the numerator band.
        denominator (str): The name of the denominator band.

    Returns:
        Callable: A function that takes an ee.Image as input and returns an ee.Image with the ratio band added.
    """
    name = f"{numerator}_{denominator}"
    def wrapper(img: ee.Image) -> ee.Image:
        return img.addBands(img.select(numerator).divide(img.select(denominator)).rename(name))
    return wrapper


def apply_boxcar(radius: int = 1, **kwargs) -> Callable:
    """
    Applies a boxcar filter to an image.

    Args:
        radius (int, optional): The radius of the boxcar filter. Defaults to 1.
        **kwargs: Additional keyword arguments.

    Returns:
        Callable: A function that applies the boxcar filter to an image.

    """
    def wrapper(img: ee.Image) -> ee.Image:
        return img.convolve(ee.Kernel.square(radius, **kwargs))
    return wrapper

# Datasets -----------------------------------------------------------------------------------------

class Sentinel1(ee.ImageCollection):
    """
    A class representing the Sentinel-1 image collection in Google Earth Engine.

    This class extends the `ee.ImageCollection` class and provides additional methods for preprocessing and filtering
    Sentinel-1 images.

    Attributes:
        None

    Methods:
        preprocess: Preprocesses the Sentinel-1 image collection based on the given parameters.
        add_dv_filter: Adds a dual-polarization filter to the Sentinel-1 image collection.

    Usage:
        sentinel1 = Sentinel1()
        preprocessed_collection = sentinel1.preprocess(aoi, start, end, look_dir='DESCENDING')
        filtered_collection = sentinel1.add_dv_filter()
    """
    
    def __init__(self):
        super().__init__("COPERNICUS/S1_GRD")

    def preprocess(self, aoi: ee.Geometry, start: str, end: str, look_dir: str = None) -> Sentinel1:
        """
        Preprocesses the Sentinel-1 image collection based on the given parameters.

        Args:
            aoi: The area of interest (AOI) to filter the image collection.
            start: The start date of the time range to filter the image collection.
            end: The end date of the time range to filter the image collection.
            look_dir: The look direction of the satellite orbit. Defaults to 'DESCENDING'.

        Returns:
            The preprocessed Sentinel-1 image collection.

        Usage:
            preprocessed_collection = sentinel1.preprocess(aoi, start, end, look_dir='DESCENDING')
        """
        look_dir = look_dir or 'DESCENDING'
        return (
            self._preprocessing(aoi, start, end)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.eq('orbitProperties_pass', look_dir))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .select('V.*')
        )
