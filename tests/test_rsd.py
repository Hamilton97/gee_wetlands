import unittest
import ee

from pprint import pprint
from gee_wetlands import rsd

class TestRemoteSensingDataSet(unittest.TestCase):
    def setUp(self) -> None:
        ee.Initialize()
        self.aoi = ee.Geometry.Point([-77.3850, 44.1631])
        self.start_date = '2019-06-20'
        self.end_date = '2019-09-21'
        self.dataset_id = "COPERNICUS/S2_HARMONIZED"
        return super().setUp()
    
    def test_pre_processing(self):
        dataset = (rsd.RemoteSensingDataset(self.dataset_id)
                   .preprocess(self.aoi, self.start_date, self.end_date))
        try:
            pprint(dataset.first().getInfo())
        except Exception as e:
            self.fail(msg=e)

    def test_composite_collection(self):
        dataset = (
            rsd.RemoteSensingDataset(self.dataset_id)
                .preprocess(self.aoi, self.start_date, self.end_date)
                .composite()
        )
        self.assertIsInstance(dataset, ee.Image)


class TestSentinel2ToaBuilder(unittest.TestCase):
    def setUp(self) -> None:
        ee.Initialize()
        self.aoi = ee.Geometry.Point([-77.3850, 44.1631])
        self.start_date = '2019-06-20'
        self.end_date = '2019-09-21'
    
    def test_preprocess(self):
        expected = rsd.Sentinel2Toa().preprocess(self.aoi, self.start_date, self.end_date)
        try:
            pprint(expected.dataset.first().getInfo())
        except Exception as e:
            self.fail(msg=e)
    
    def test_add_ndvi(self):
        actual = rsd.Sentinel2Toa().add_ndvi()
        actual_dataset = actual.dataset.first()
        try:
            pprint(actual_dataset.getInfo())
        except Exception as e:
            self.fail(msg=e)
        
        actual_b_name = actual_dataset.select('NDVI').bandNames().getInfo()
        expected_b_name = ['NDVI'] 
        self.assertEqual(expected_b_name, actual_b_name)
    
    def test_add_savi(self):
        
        actual = rsd.Sentinel2Toa().add_savi()
        actual_dataset = actual.dataset.first()
        try:
            pprint(actual_dataset.getInfo())
        except Exception as e:
            self.fail(msg=e)
        
        actual_b_name = actual_dataset.select('SAVI').bandNames().getInfo()
        expected_b_name = ['SAVI'] 
        self.assertEqual(expected_b_name, actual_b_name)
    
    def test_add_tasseled_cap(self):
        
        actual = rsd.Sentinel2Toa().add_tasseled_cap()
        actual_dataset = actual.dataset.first()
        try:
            pprint(actual_dataset.getInfo())
        except Exception as e:
            self.fail(msg=e)
        
        actual_b_name = actual_dataset.select('brightness', 'greenness', 'wetness').bandNames().getInfo()
        expected_b_name = ['brightness', 'greenness', 'wetness'] 
        self.assertEqual(expected_b_name, actual_b_name)