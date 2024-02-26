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