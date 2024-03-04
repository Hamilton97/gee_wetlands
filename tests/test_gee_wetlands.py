#!/usr/bin/env python

"""Tests for `gee_wetlands` package."""

import unittest
from pprint import pprint

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


class TestRadarRasterFunctions(unittest.TestCase):
    def setUp(self) -> None:
        ee.Initialize()
        geom = ee.Geometry.Point([-77.3850, 44.1631])
        self.dataset = Sentinel1().preprocessing(geom, '2020', '2021').apply_acs_filter()
        return super().setUp()

    def test_add_ratio(self):
        actual = self.dataset.map(add_ratio('VV', 'VH')).first().select('VV_VH').bandNames().getInfo()
        expected = ['VV_VH']
        self.assertEqual(expected, actual)
    
    def test_add_boxcar(self):
        try:
            self.dataset.map(apply_boxcar()).first().getInfo()
        except Exception as e:
            self.fail(msg=e)


class TestDataCubeDataset(unittest.TestCase):
    def setUp(self) -> None:
        ee.Initialize()
        self.aoi = ee.FeatureCollection('projects/fpca-336015/assets/NovaScotia/_527_ECO_DIST').geometry()
        self.dataset_id = 'projects/fpca-336015/assets/cnwi-datasets/aoi_novascotia/datacube'
    
    def test_init(self):
        try:
            pprint(DataCube(self.dataset_id).first().getInfo())
        except Exception as e:
            self.fail(msg=e)
    
    def test_preprocessin(self):
        dc = DataCube(self.dataset_id).filterBounds(self.aoi).select_spectral_bands()
        try:
            pprint(dc.first().bandNames().getInfo())
        except Exception as e:
            self.fail(msg=e)
    
    def test_rename(self):
        dc = DataCube(self.dataset_id).filterBounds(self.aoi).select_spectral_bands().rename_bands()
        try:
            pprint(dc.first().bandNames().getInfo())
        except Exception as e:
            self.fail(msg=e)


class TestAlosPalsar2(unittest.TestCase):
    ee.Initialize()
    def test_init(self):
        try:
            pprint(ALOSPalsar2().filterDate('2018', '2020').first().bandNames().getInfo())
        except Exception as e:
            self.fail(msg=e)


class TestSentinel1Dataset(unittest.TestCase):
    def setUp(self) -> None:
        geom = ee.Geometry.Point([-77.3850, 44.1631])
        date = '2019-06-21', '2019-09-20'
        
        self.s1 = Sentinel1().filterBounds(geom).filterDate(*date)

        return super().setUp()

    def test_relative_orbit_numbers(self):
        try:
            pprint(self.s1.get_orbit_numbers().getInfo())
        except Exception as e:
            self.fail(msg=e)

    def test_s1_select(self):
        try:
            pprint(self.s1.select().first().bandNames().getInfo())
        except Exception as e:
            self.fail(msg=e)
    
    def test_s1_select_1_channel(self):
        try:
            pprint(self.s1.select('VV').first().bandNames().getInfo())
        except Exception as e:
            self.fail(msg=e)


class TestHarmonicTimeSeries(unittest.TestCase):
    def setUp(self) -> None:
        ee.Initialize()
        geom = ee.Geometry.Point([-77.3850, 44.1631])
        self.dataset = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(geom).filterDate('2020', '2021')
        self.harmonic = HarmonicTimeSeries(self.dataset, 'B8')

    def test_add_constant(self):
        self.harmonic.add_constant()
        result = self.harmonic.dataset.first().bandNames().getInfo()
        self.assertIn('constant', result)

    def test_add_time(self):
        self.harmonic.add_time()
        result = self.harmonic.dataset.first().bandNames().getInfo()
        self.assertIn('t', result)

    def test_add_harmonics(self):
        self.harmonic.add_constant().add_time().add_harmonics()
        result = self.harmonic.dataset.first().bandNames().getInfo()
        try:
            pprint(result)
        except Exception as e:
            self.fail(e)

    def test_set_harmonic_trend(self):
        self.harmonic.add_constant().add_time().add_harmonics().set_harmonic_trend()
        self.assertIsNotNone(self.harmonic.trend)

    def test_add_harmonic_coefficients(self):
        self.harmonic.add_constant().add_time().add_harmonics().set_harmonic_trend().add_harmonic_coefficients()
        result = self.harmonic.dataset.first().bandNames().getInfo()
        for name in self.harmonic.independent:
            self.assertIn(f"{name}_coef", result)

    def test_add_phase(self):
        self.harmonic.add_constant().add_time().add_harmonics().set_harmonic_trend().add_harmonic_coefficients().add_phase()
        result = self.harmonic.dataset.first().bandNames().getInfo()
        for _, mode in enumerate(list(range(self.harmonic.modes)), start= 1):
            self.assertIn(f'phase_{_}', result)

    def test_add_amplitude(self):
        self.harmonic.add_constant().add_time().add_harmonics().set_harmonic_trend().add_harmonic_coefficients().add_amplitue()
        result = self.harmonic.dataset.first().bandNames().getInfo()
        for _, mode in enumerate(list(range(self.harmonic.modes)), start= 1):
            self.assertIn(f'amplitude_{_}', result)

    def test_build(self):
        result = self.harmonic.build()
        self.assertIsInstance(result, ee.ImageCollection)

    def test_transform(self):
        result = self.harmonic.transform()
        self.assertIsInstance(result, ee.Image)
