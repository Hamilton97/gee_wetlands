import unittest

import ee

from gee_wetlands.calcs import *


class TestRasterCalculators(unittest.TestCase):
    def setUp(self) -> None:
        ee.Initialize()
        geom = ee.Geometry.Point([-77.3850, 44.1631])
        self.dataset = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(geom).filterDate('2020', '2021')
        self.nir = "B8"
        self.red = 'B4'
        
        return super().setUp()
    
    def test_add_ndvi_collection(self):
        result = self.dataset.map(compute_ndvi(self.nir, self.red))
        ndvi = result.first().select('NDVI').bandNames().getInfo()
        self.assertEqual(['NDVI'], ndvi)
    
    def test_add_savi_to_collecton(self):
        restult = self.dataset.map(compute_savi(self.nir, self.red))
        savi = restult.first().select('SAVI').bandNames().getInfo()
        self.assertEqual(['SAVI'], savi)