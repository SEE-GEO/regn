"""
==========================
regn.data.csu.preprocessor
==========================

This module contains the PreprocessorFile class which provides an interface
to read CSU preprocessor files.
"""
import logging

import numpy as np
import xarray

LOGGER = logging.getLogger(__name__)

###############################################################################
# Struct types
###############################################################################

N_SPECIES = 5
N_TEMPERATURES = 12
N_LAYERS = 28
N_PROFILES = 80
N_CHANNELS = 15

DATE_TYPE = np.dtype(
    [("year", "i2"),
     ("month", "i2"),
     ("day", "i2"),
     ("hour", "i2"),
     ("minute", "i2"),
     ("second", "i2")]
)

ORBIT_HEADER_TYPES = np.dtype(
    [("satellite", "a12"),
     ("sensor", "a12"),
     ("preprocessor", "a12"),
     ("profile_database_file", "a128"),
     ("radiometer_file", "a128"),
     ("calibration_file", "a128"),
     ("granule_number", "i"),
     ("number_of_scans", "i"),
     ("number_of_pixels", "i"),
     ("n_channels", "i"),
     ("frequencies", f"{N_CHANNELS}f4"),
     ("comment", "a40")]
)

SCAN_HEADER_TYPES = np.dtype(
    [("scan_date", DATE_TYPE),
     ("scan_latitude", "f4"),
     ("scan_longitude", "f4"),
     ("scan_altitude", "f4"),
     ])

DATA_RECORD_TYPES = np.dtype(
    [("latitude", "f4"),
     ("longitude", "f4"),

     ("brightness_temperatures", f"{N_CHANNELS}f4"),
     ("earth_incidence_angle", f"{N_CHANNELS}f4"),
     ("wet_bulb_temperature", f"f4"),
     ("lapse_rate", f"f4"),
     ("total_column_water_vapor", f"f4"),
     ("surface_temperature", f"f4"),
     ("two_meter_temperature", f"f4"),

     ("quality_flag", f"i"),
     ("sunglint_angle", f"i1"),
     ("surface_type", f"i1"),
     ("airmass_type", f"i2")]
    )

###############################################################################
# Preprocessor file class
###############################################################################

class PreprocessorFile:
    """
    Interface to read CSU preprocessor files.

    Attibutes:
        filename: The path of the source file.
        orbit_header: Numpy structured array containing the orbit header.
        n_scans: The number of scans in the file.
        n_pixels: The number of pixels in the file.
    """
    def __init__(self, filename):
        """
        Read preprocessor file.

        Args:
            filename: Path to the file to read.
        """
        self.filename = filename
        self.orbit_header = np.fromfile(self.filename,
                                        ORBIT_HEADER_TYPES,
                                        count=1)
        self.n_scans = self.orbit_header["number_of_scans"][0]
        self.n_pixels = self.orbit_header["number_of_pixels"][0]

    @property
    def satellite(self):
        """
        The satellite from which the data in this file originates.
        """
        return self.orbit_header["satellite"]

    @property
    def sensor(self):
        """
        The sensor from which the data in this file originates.
        """
        return self.orbit_header["sensor"]

    @property
    def scans(self):
        """
        Iterates of the scans in the file. Each scan is returned as Numpy
        structured array of size n_pixels and dtype DATA_RECORD_TYPES.
        """
        for i in range(self.n_scans):
            yield self.get_scan(i)

    def get_scan(self, i):
        """
        Return scan as Numpy structured array of size n_pixels and dtype
        DATA_RECORD_TYPES.
        """
        offset = ORBIT_HEADER_TYPES.itemsize
        offset += i * (SCAN_HEADER_TYPES.itemsize
                       + self.n_pixels * DATA_RECORD_TYPES.itemsize)
        offset += SCAN_HEADER_TYPES.itemsize
        return np.fromfile(self.filename,
                           DATA_RECORD_TYPES,
                           count=self.n_pixels,
                           offset=offset)

    def to_xarray_dataset(self):
        """
        Return data in file as xarray dataset.
        """
        data = {k: np.zeros((self.n_scans, self.n_pixels) + d[0].shape)
                for k, d in DATA_RECORD_TYPES.fields.items()}
        for i, s in enumerate(self.scans):
            for k, d in data.items():
                d[i] = s[k]

        dims = ["scans", "pixels", "channels"]
        data = {k: (dims[:len(d.shape)], d) for k, d in data.items()}
        return xarray.Dataset(data)


pf = PreprocessorFile("/home/simon/scratch/GMI_190101_027510.pp")
ds = pf.to_xarray_dataset()
