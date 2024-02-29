#!/usr/bin/env python

"""Tests for `gee_wetlands` package."""

import unittest

import ee

from gee_wetlands.gee_wetlands import *


class TestOpticalRasterCalculators(unittest.TestCase):
    def setUp(self) -> None:
        ee.Initialize()
        geom = ee.Geometry.Point([-77.3850, 44.1631])
        self.dataset = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(geom).filterDate('2020', '2021')
    
    def test_add_ndvi_collection(self):
        result = self.dataset.map(add_ndvi('B8', 'B4'))
        ndvi = result.first().select('NDVI').bandNames().getInfo()
        self.assertEqual(['NDVI'], ndvi)
    
    def test_add_savi_to_collecton(self):
        restult = self.dataset.map(add_savi('B8', 'B4'))
        savi = restult.first().select('SAVI').bandNames().getInfo()
        self.assertEqual(['SAVI'], savi)
    
    def test_add_tasseled_cap_to_collection(self):
        restult = self.dataset.map(add_tasseled_cap('B2', 'B3', 'B4', 'B8', 'B11', 'B12'))
        bands = restult.first().select('brightness', 'wetness', 'greenness').bandNames().getInfo()
        self.assertEqual(['brightness', 'wetness', 'greenness'], bands)
    
