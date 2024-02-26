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
