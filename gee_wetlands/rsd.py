from __future__ import annotations
"""
rsd.py: Remote Sensing Datasets
"""
from typing import Any
import ee

from .calcs import *


class RemoteSensingDataset(ee.ImageCollection):
    """
    A class representing a remote sensing dataset.

    This class extends the `ee.ImageCollection` class and provides additional methods for preprocessing and compositing the dataset.

    Args:
        args (Any): Arguments to be passed to the `ee.ImageCollection` constructor.

    Attributes:
        None

    Methods:
        preprocess: Preprocesses the dataset by filtering it based on the area of interest and date range.
        composite: Generates a composite image from the dataset by calculating the median pixel values.
    """

    def __init__(self, args: Any):
        super().__init__(args)

    def preprocess(self, aoi: ee.Geometry, start: str, end: str) -> RemoteSensingDataset:
        """
        Preprocesses the dataset by filtering it based on the area of interest and date range.

        Args:
            aoi: Area of interest (e.g., a geometry or feature).
            start: Start date of the time range to filter the dataset.
            end: End date of the time range to filter the dataset.

        Returns:
            Filtered `RemoteSensingDataset` object.

        """
        return self.filterBounds(aoi).filterDate(start, end)

    def composite(self) -> ee.Image:
        """
        Generates a composite image from the dataset by calculating the median pixel values.

        Returns:
            Composite image as an `ee.Image` object.

        """
        return self.median()


class Sentinel2Toa:
    """
    Class representing Sentinel-2 Top of Atmosphere (TOA) data.

    Attributes:
        BANDS (list): List of Sentinel-2 bands.
        dataset (RemoteSensingDataset): RemoteSensingDataset object representing the TOA data.

    Methods:
        __init__(): Initializes a Sentinel2Toa object.
        __add__(other): Adds two Sentinel2Toa objects together.
        preprocess(aoi, start, end, cloudy_percent): Preprocesses the TOA data.
        add_ndvi(): Adds Normalized Difference Vegetation Index (NDVI) to the TOA data.
        add_savi(): Adds Soil-Adjusted Vegetation Index (SAVI) to the TOA data.
        add_tasseled_cap(): Adds Tasseled Cap indices to the TOA data.
        build(): Returns the final TOA dataset.
        s2_cloud_mask(image): Applies cloud and cirrus mask to the TOA image.
    """
    
    BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    
    def __init__(self) -> None:
        """
        Initializes a Sentinel2Toa object.
        """
        self.dataset = RemoteSensingDataset("COPERNICUS/S2_HARMONIZED")
    
    def __add__(self, other: Sentinel2Toa) -> RemoteSensingDataset:
        """
        Adds two Sentinel2Toa objects together.

        Args:
            other (Sentinel2Toa): The other Sentinel2Toa object to be added.

        Returns:
            RemoteSensingDataset: The merged dataset.
        
        Raises:
            TypeError: If the other object is not of type Sentinel2Toa.
        """
        if not isinstance(other, Sentinel2Toa):
            raise TypeError("Other is not of type Sentinel2Toa")
        return self.dataset.merge(self.dataset)

    def preprocess(self, aoi: ee.Geometry, start: str, end: str, cloudy_percent: float = 20.0) -> Sentinel2Toa:
        """
        Preprocesses the TOA data.

        Args:
            aoi (ee.Geometry): Area of interest.
            start (str): Start date of the time range.
            end (str): End date of the time range.
            cloudy_percent (float, optional): Maximum percentage of cloudy pixels to keep. Defaults to 20.0.

        Returns:
            Sentinel2Toa: The updated Sentinel2Toa object.
        """
        self.dataset = (
            self.dataset.preprocess(aoi, start, end)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloudy_percent))
            .map(self.s2_cloud_mask)
            .select(self.BANDS)
        )
        return self

    def add_ndvi(self) -> RemoteSensingDataset:
        """
        Adds Normalized Difference Vegetation Index (NDVI) to the TOA data.

        Returns:
            RemoteSensingDataset: The updated dataset.
        """
        self.dataset = self.dataset.map(compute_ndvi('B8', 'B4'))
        return self
    
    def add_savi(self) -> RemoteSensingDataset:
        """
        Adds Soil-Adjusted Vegetation Index (SAVI) to the TOA data.

        Returns:
            RemoteSensingDataset: The updated dataset.
        """
        self.dataset = self.dataset.map(compute_savi('B8', 'B4'))
        return self
    
    def add_tasseled_cap(self) -> RemoteSensingDataset:
        """
        Adds Tasseled Cap indices to the TOA data.

        Returns:
            RemoteSensingDataset: The updated dataset.
        """
        self.dataset = self.dataset.map(compute_tasseled_cap('B2', 'B3', 'B4', 'B8', 'B11', 'B12'))
        return self
    
    def build(self) -> RemoteSensingDataset:
        """
        Returns the final TOA dataset.

        Returns:
            RemoteSensingDataset: The final TOA dataset.
        """
        return self.dataset

    @staticmethod
    def s2_cloud_mask(image: ee.Image) -> ee.Image:
        """
        Applies cloud and cirrus mask to the TOA image.

        Args:
            image (ee.Image): The TOA image.

        Returns:
            ee.Image: The masked TOA image.
        """
        qa = image.select('QA60')

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return image.updateMask(mask)