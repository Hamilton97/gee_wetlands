from __future__ import annotations
from math import pi
from typing import Callable, Any

import ee



def preprocessing(self, *args) -> ee.ImageCollection:
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


ee.ImageCollection.preprocessing = preprocessing
ee.ImageCollection.composite = composite
ee.ImageCollection.__add__ = __add__
 

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


def add_constant(img: ee.Image) -> ee.Image:
    """
    Adds a constant band with a value of 1 to the input image.
    
    Args:
        img (ee.Image): The input image to which the constant band will be added.
        
    Returns:
        ee.Image: The input image with the constant band added.
    """
    return img.addBands(ee.Image(1))


def add_time(img: ee.Image) -> ee.Image:
    """
    Adds a time band to the input image.
    
    Args:
        img (ee.Image): The input image.
        
    Returns:
        ee.Image: The input image with a time band added.
    """
    date = img.date();
    years = date.difference(ee.Date('1970-01-01'), 'year');
    timeRadians = ee.Image(years.multiply(2 * pi));
    return img.addBands(timeRadians.rename('t').float());


def add_harmonics(freqs: list[int], cos_names: list[str], sin_names: list[str]) -> Callable:
    """
    Adds harmonic bands to an image based on the given frequencies, cosine band names, and sine band names.
    
    Args:
        freqs (list[int]): A list of frequencies for the harmonic bands.
        cos_names (list[str]): A list of names for the cosine bands.
        sin_names (list[str]): A list of names for the sine bands.
    
    Returns:
        Callable: A wrapper function that adds the harmonic bands to the input image.
    """
    def wrapper(img):
        frequencies = ee.Image.constant(freqs)
        time = ee.Image(img).select('t')
        cosines = time.multiply(frequencies).cos().rename(cos_names)
        sines = time.multiply(frequencies).sin().rename(sin_names)
        return img.addBands(cosines).addBands(sines)
    return wrapper


def compute_trend(dataset: ee.ImageCollection, independents: list[str], dependents: str) -> ee.Image:
    """
    Computes the linear regression trend of the given dataset.

    Args:
        dataset (ee.ImageCollection): The input image collection.
        independents (list[str]): A list of independent variable names.
        dependents (str): The dependent variable name.

    Returns:
        ee.Image: The image representing the linear regression trend.

    """
    return dataset.select(independents + [dependents]).reduce(ee.Reducer.linearRegression(len(independents), 1))


def compute_harmonic_trend_coefficients(harmonic_trend: ee.Image, independents: list[str]) -> ee.Image:
    """
    Compute the harmonic trend coefficients for a given harmonic trend image.

    Args:
        harmonic_trend (ee.Image): The harmonic trend image.
        independents (list[str]): The list of independent variables.

    Returns:
        ee.Image: The harmonic trend coefficients image.
    """
    return harmonic_trend.select('coefficients').arrayFlatten([independents, ['coef']])


def add_phase(cos: str, sin: str) -> Callable:
    """
    Adds a phase band to an image based on the provided cosine and sine bands.

    Args:
        cos (str): The name of the cosine band.
        sin (str): The name of the sine band.

    Returns:
        Callable: A function that takes an ee.Image as input and returns an ee.Image with the phase band added.
    """
    mode = cos.split('_')[-2]
    name = f'phase_{mode}'
    def wrapper(img: ee.Image) -> ee.Image:
        return img.addBands(img.select(cos).atan2(img.select(sin)).rename(name))
    return wrapper


def add_amplitude(cos: str, sin: str) -> Callable:
    """
    Adds an amplitude band to an image based on the provided cosine and sine bands.

    Args:
        cos (str): The name of the cosine band.
        sin (str): The name of the sine band.

    Returns:
        Callable: A function that takes an image and adds the amplitude band to it.
    """
    mode = cos.split('_')[-2]
    name = f'amplitude_{mode}'
    def wrapper(img: ee.Image) -> ee.Image:
        return img.addBands(img.select(cos).hypot(img.select(sin)).rename(name))
    return wrapper


def transform(dataset: ee.ImageCollection) -> ee.Image:
    """
    Transforms the given image collection into a single composite image.
    
    Args:
        dataset (ee.ImageCollection): The image collection to be transformed.
    
    Returns:
        ee.Image: The transformed composite image.
    """
    return dataset.composite().unitScale(-1, 1)


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

    def apply_dv_filter(self) -> Sentinel1:
        return (self.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

    def apply_iw_mode_filter(self) -> Sentinel1:
        return self.filter(ee.Filter.eq('instrumentMode', 'IW'))

    def apply_acs_filter(self) -> Sentinel1:
        return self.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    
    def apply_desc_filter(self) -> Sentinel1:
        return self.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

    def get_orbit_numbers(self) -> ee.List:
        return self.aggregate_array('relativeOrbitNumber_start').distinct()
    
    def select(self, args: Any = None) -> Sentinel1:
        if args is None:
            return super().select('V.*')
        return super().select(args)


class Sentinel2TOA(ee.ImageCollection):
    """
    A class representing the Sentinel-2 Top of Atmosphere (TOA) image collection in Google Earth Engine.
    """

    BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    
    def __init__(self):
        super().__init__("COPERNICUS/S2_HARMONIZED")

    def apply_cloud_percent_filer(self, percent: float = 20.0) -> Sentinel2TOA:
        self.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', percent))

    def apply_cloud_mask(self):
        
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
        return self.map(mask_clouds)

    def select(self, args: Any = None):
        if args is None:
            return super().select(self.BANDS)
        return super().select(args)


class ALOSPalsar2(ee.ImageCollection):
    """
    A class representing the ALOS PALSAR-2 Image Collection in Google Earth Engine.
    
    This class extends the `ee.ImageCollection` class and provides additional functionality
    specific to the ALOS PALSAR-2 data.
    """
    
    def __init__(self):
        super().__init__("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
    
    def select(self, args: Any = None) -> ALOSPalsar2:
        if args is None:
            return super().select('H.*')
        return super().select(args)


class DataCube(ee.ImageCollection):
    """
    A class representing a data cube in Google Earth Engine.

    This class extends the `ee.ImageCollection` class and provides additional methods for processing and manipulating the data cube.

    Attributes:
        BANDS (list): A list of spectral bands in the data cube.

    Methods:
        select_spectral_bands: Selects the spectral bands based on a pattern.
        rename_bands: Renames the spectral bands in the data cube.
        preproces: Preprocesses the data cube by applying a series of operations.

    """

    BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    def __init__(self, args: Any):
        super().__init__(args)

    def select_spectral_bands(self) -> DataCube:
        """
        Selects the spectral bands based on a pattern.

        Returns:
            DataCube: A new data cube with the selected spectral bands.
        """
        pattern = "a_spri_b0[2-9].*|a_spri_b[1-2].*|b_summ_b0[2-9].*|b_summ_b[1-2].*|c_fall_b0[2-9].*|c_fall_b[1-2].*"
        return self.select(pattern)

    def rename_bands(self) -> DataCube:
        """
        Renames the spectral bands in the data cube.

        Returns:
            DataCube: A new data cube with the renamed bands.
        """
        spring_bands = self.BANDS
        summer_bands = [f'{_}_1' for _ in self.BANDS]
        fall_bands = [f'{_}_2' for _ in self.BANDS]

        new_names = spring_bands + summer_bands + fall_bands
        return self.select(self.first().bandNames(), new_names)

    def preprocess(self, aoi: ee.Geometry) -> DataCube:
        return self.filterBounds(aoi).select_spectral_bands().rename_bands()


# Time Series Modeling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def compute_fourier_transform(dataset: ee.ImageCollection, dependent: str, modes: int = 3) -> ee.Image:
    """
    Computes the Fourier transform of the given dataset.

    Args:
        dataset (ee.ImageCollection): The input dataset. Assumed to have inital pre processing done
        dependent (str): The dependent variable.
        modes (int, optional): The number of Fourier modes to compute. Defaults to 3.

    Returns:
        ee.Image: The Fourier transform of the dataset.
    """
    return (
        HarmonicTimeSeries(dataset=dataset, dependent=dependent, modes=modes)
        .add_constant()
        .add_time()
        .add_harmonics()
        .set_harmonic_trend()
        .add_harmonic_coefficients()
        .add_phase()
        .add_amplitue()
        .transform()
    )


class HarmonicTimeSeries:
    """
    Represents a harmonic time series analysis on an Earth Engine ImageCollection.

    Args:
        dataset (ee.ImageCollection): The Earth Engine ImageCollection to perform the analysis on.
        dependent (str): The name of the dependent variable.
        modes (int, optional): The number of harmonic modes to include. Defaults to 3.

    Attributes:
        dataset (ee.ImageCollection): The Earth Engine ImageCollection used for the analysis.
        independent (list[str]): The list of independent variable names.
        dependent (str): The name of the dependent variable.
        modes (int): The number of harmonic modes to include.
        trend (ee.Image | None): The computed trend image.
        coef (list[str] | None): The list of harmonic coefficient variable names.

    Methods:
        add_constant(): Adds a constant variable to the dataset.
        add_time(): Adds a time variable to the dataset.
        add_harmonics(): Adds harmonic variables to the dataset.
        set_harmonic_trend(): Computes the trend image using the independent and dependent variables.
        add_harmonic_coefficients(): Adds harmonic coefficient variables to the dataset.
        add_phase(): Adds phase variables to the dataset.
        add_amplitude(): Adds amplitude variables to the dataset.
        build() -> ee.ImageCollection: Returns the modified dataset.
        transform() -> ee.Image: Transforms the dataset into a single image.
        get_sin_cos_selectors() -> tuple[list[str], list[str]]: Returns the selectors for sin and cos variables.
    """

    def __init__(self, dataset: ee.ImageCollection, dependent: str, modes: int = 3) -> None:
        """
        Initialize the HarmonicTimeSeries object.

        Args:
            dataset (ee.ImageCollection): The Earth Engine Image Collection.
            dependent (str): The dependent variable.
            modes (int, optional): The number of modes to compute. Defaults to 3.
        """
        self.dataset = dataset
        self.independent = []
        self.dependent = dependent
        self.modes = modes
        self.trend = None
        self.coef: list[str] | None = None
    
    def add_constant(self):
        """
        Adds a constant column to the dataset and updates the list of independent variables.

        Returns:
            self: The updated instance of the class.
        """
        self.dataset = self.dataset.map(add_constant)
        self.independent.append('constant')
        return self
    
    def add_time(self) -> HarmonicTimeSeries:
        """
        Adds a time dimension to the dataset and updates the list of independent variables.
        
        Returns:
            HarmonicTimeSeries: The updated HarmonicTimeSeries object.
        """
        self.dataset = self.dataset.map(add_time)
        self.independent.append('t')
        return self
    
    def add_harmonics(self) -> HarmonicTimeSeries:
        """
        Adds harmonics to the dataset.

        Returns:
            HarmonicTimeSeries: The updated HarmonicTimeSeries object.
        """
        freqs = list(range(1, self.modes + 1))
        cos_names = [f'cos_{_}' for _ in freqs]
        sin_names = [f'sin_{_}' for _ in freqs]
        self.dataset = self.dataset.map(add_harmonics(freqs=freqs, cos_names=cos_names, sin_names=sin_names))
        self.independent.extend(cos_names)
        self.independent.extend(sin_names)
        return self
    
    def set_harmonic_trend(self) -> HarmonicTimeSeries:
        """
        Computes the harmonic trend of the dataset using the provided independent and dependent variables.

        Returns:
            HarmonicTimeSeries: The computed harmonic trend.
        """
        self.trend = compute_trend(self.dataset, self.independent, self.dependent)
        return self

    def add_harmonic_coefficients(self) -> HarmonicTimeSeries:
        """
        Adds harmonic trend coefficients to the dataset.

        Returns:
            HarmonicTimeSeries: The updated HarmonicTimeSeries object.
        """
        selectors = [f"{_}_coef" for _ in self.independent] + [self.dependent]
        coef = compute_harmonic_trend_coefficients(self.trend, self.independent)
        self.dataset = self.dataset.map(lambda x: x.addBands(coef)).select(selectors)
        self.coef = selectors
        return self

    def add_phase(self) -> HarmonicTimeSeries:
        """
        Adds phase information to the dataset by applying sine and cosine functions.

        Returns:
            HarmonicTimeSeries: The updated HarmonicTimeSeries object.
        """
        for cos, sin in self.get_sin_cos_selectors():
            self.dataset = self.dataset.map(add_phase(cos, sin))
        return self
    
    def add_amplitue(self) -> HarmonicTimeSeries:
        """
        Adds the amplitude of harmonic components to the dataset.

        Returns:
            HarmonicTimeSeries: The updated HarmonicTimeSeries object.
        """
        for cos, sin in self.get_sin_cos_selectors():
            self.dataset = self.dataset.map(add_amplitude(cos, sin))
        return self
    
    def build(self) -> ee.ImageCollection:
        """
        Builds and returns the image collection.

        Returns:
            ee.ImageCollection: The built image collection.
        """
        return self.dataset
    
    def transform(self) -> ee.Image:
        """
        Transforms the dataset into an Earth Engine Image.

        Returns:
            ee.Image: The transformed Earth Engine Image.
        """
        return transform(self.dataset)

    def get_sin_cos_selectors(self) -> tuple[list[str], list[str]]:
        """
        Returns a tuple of two lists containing the 'cos' and 'sin' selectors from the 'coef' list.

        Returns:
            A tuple containing two lists:
            - The first list contains the selectors from 'coef' that contain the substring 'cos'.
            - The second list contains the selectors from 'coef' that contain the substring 'sin'.
        """
        return zip(
            [_ for _ in self.coef if 'cos' in _],
            [_ for _ in self.coef if 'sin' in _]
        )
        