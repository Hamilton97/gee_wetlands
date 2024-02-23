import unittest

import ee

from gee_wetlands.calcs import *


class TestRasterCalculators(unittest.TestCase):
    def setUp(self) -> None:
        ee.Initialize()
        geom = ee.Geometry.Point([])
        self.dataset = ee.ImageCollection().filterBounds(geom).filterDate('2020', '2021')
        self.nir = "B8"
        self.red = 'B4'
        
        return super().setUp()
    
    def test_add_ndvi_collection(self):
        result = self.dataset.map(lambda x: x.addBands(compute_ndvi(self.nir, self.red)))
        ndvi = result.first().bandNames().getInfo()
        self.assertEqual(['NDVI'], ndvi)