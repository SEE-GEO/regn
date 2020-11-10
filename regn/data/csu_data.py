"""
regn.data.csu_data
==================

This module provides two classes to read in the binary data used
for the training of the GPROF algorithm.
"""
from pathlib import Path
import numpy as np
import netCDF4


def check_sample(data):
    """
    Check that brightness temperatures of a sample are within a valid range.

    Arguments:
        data: The data array containing the data of one training sample.

    Return:
        Bool indicating whether the given sample contains valid brightness
        temperatures.
    """
    return all([data[i] > 0 and data[i] < 1000 for i in range(13, 26)])


def get_files(base_path, year, month, day):
    """
    Iterate over files for specific year, month and day.

    Args:
        base_path(``pathlib.Path``): Root folder containing the binary data
            files.
        year(``int``): Year as two-digit integer.
        month(``int``): Month as two-digit integer.
        day(``int``): The day as two-digit integer.
    """
    base_path = Path(base_path)
    return base_path.glob(f"{year:02d}{month:02d}/"
                          f"*20{year:02d}{month:02d}{day:02d}*.dat")

###############################################################################
# GMI Binary data
###############################################################################


HEADER_TYPES_GMI = [('satcode', 'S5'),
                    ('sensor', 'S5')]
HEADER_TYPES_GMI += [(f'freq_{i}', "f4") for i in range(13)]
HEADER_TYPES_GMI += [(f'viewing_angle_{i}', "f4") for i in range(13)]

PIXEL_TYPES_GMI = [('nx', 'i4'),
                   ('ny', 'i4'),
                   ('year', 'i4'),
                   ('month', 'i4'),
                   ('day', 'i4'),
                   ('hour', 'i4'),
                   ('minute', 'i4'),
                   ('second', 'i4'),
                   ('lat', 'f4'),
                   ('lon', 'f4'),
                   ('sfccode', 'i4'),
                   ('tcwv', 'f4'),
                   ('T2m', 'f4')]
PIXEL_TYPES_GMI += [(f'Tb_{i}', 'f4') for i in range(13)]
PIXEL_TYPES_GMI += [(f'sfcprcp', 'f4'),
                    (f'cnvprcp', 'f4')]


class GMIBinaryFile:
    """
    Class to extract data from GMI CSU binary files and store to
    NetCDF file.
    """
    def __init__(self, filename):
        """
        Open CSU binary file containing GPROF training data for
        GMI sensor..

        Args:
            filename(``pathlib.Path``): The file to open.
        """
        self.header = np.memmap(filename,
                                dtype=HEADER_TYPES_GMI,
                                mode="r",
                                shape=(1,))
        self.pixels = np.memmap(filename,
                                dtype=PIXEL_TYPES_GMI,
                                offset= 10 + 8 * 13,
                                mode="r")

    @classmethod
    def create_output_file(cls, filename):
        """
        Create NetCDF output file.

        Args:
            filename(``pathlib.Path``): The filename of the output file.
        """
        file = netCDF4.Dataset(filename, "w")
        file.createDimension("channels", size=13)
        file.createDimension("samples", size=None)
        file.createVariable("brightness_temperature",
                            "f4",
                            dimensions=("samples",
                                        "channels"))
        file.createVariable("tbs_min", "f4", dimensions=("channels"))
        file.createVariable("tbs_max", "f4", dimensions=("channels"))
        file.createVariable("latitude", "f4", dimensions=("samples",))
        file.createVariable("longitude", "f4", dimensions=("samples",))
        file.createVariable("surface_type", "f4", dimensions=("samples",))
        file.createVariable("tcwv", "f4", dimensions=("samples",))
        file.createVariable("t2m", "f4", dimensions=("samples",))
        file.createVariable("surface_precipitation",
                            "f4",
                            dimensions=("samples",))
        file.createVariable("convective_precipitation",
                            "f4",
                            dimensions=("samples",))
        # Also include date.
        file.createVariable("year", "i4", dimensions=("samples",))
        file.createVariable("month", "i4", dimensions=("samples",))
        file.createVariable("day", "i4", dimensions=("samples",))
        file.createVariable("hour", "i4", dimensions=("samples",))
        file.createVariable("minute", "i4", dimensions=("samples",))
        file.createVariable("second", "i4", dimensions=("samples",))

        file["tbs_min"][:] = 1e30
        file["tbs_max"][:] = 0.0
        return file

    def write_to_file(self, file, samples=-1):
        """
        Write data to NetCDF file.

        Arguments
            file(``netCDF4.Dataset``): File handle to the output file.
            samples(``int``): How many samples to extract from the file.
        """
        v_tbs = file.variables["brightness_temperature"]
        v_lats = file.variables["latitude"]
        v_lons = file.variables["longitude"]
        v_sfccode = file.variables["surface_type"]
        v_tcwv = file.variables["tcwv"]
        v_t2m = file.variables["t2m"]
        v_surf_precip = file.variables["surface_precipitation"]
        v_conv_precip = file.variables["convective_precipitation"]
        v_tbs_min = file.variables["tbs_min"]
        v_tbs_max = file.variables["tbs_max"]

        v_year = file.variables["year"]
        v_month = file.variables["month"]
        v_day = file.variables["day"]
        v_hour = file.variables["hour"]
        v_minute = file.variables["minute"]
        v_second = file.variables["second"]

        n_samples = len(self.pixels)
        if samples < 0:
            samples = n_samples
        indices = np.random.permutation(np.arange(n_samples))

        n_extracted = 0
        for index in indices:
            if n_extracted >= samples:
                break
            if not check_sample(self.pixels[index]):
                continue
            d = self.pixels[index]
            i = file.dimensions["samples"].size

            v_year[i] = d[2]
            v_month[i] = d[3]
            v_day[i] = d[4]
            v_hour[i] = d[5]
            v_minute[i] = d[6]
            v_second[i] = d[7]

            v_lats[i] = d[8]
            v_lons[i] = d[9]
            v_sfccode[i] = d[10]
            v_tcwv[i] = d[11]
            v_t2m[i] = d[12]
            for j in range(13):
                tb = d[13 + j]
                v_tbs_min[j] = np.minimum(v_tbs_min[j], tb)
                v_tbs_max[j] = np.maximum(v_tbs_max[j], tb)
                v_tbs[i, j] = tb
            sp = d[26]
            v_surf_precip[i] = sp
            cp = d[27]
            v_conv_precip[i] = cp
            n_extracted += 1


###############################################################################
# MHS binary data
###############################################################################
HEADER_TYPES_MHS = [('satcode', 'S5'),
                    ('sensor', 'S5')]
HEADER_TYPES_MHS += [(f'freq_{i}', "f4") for i in range(5)]
HEADER_TYPES_MHS += [(f'viewing_angle_{i}', "f4") for i in range(10)]

PIXEL_TYPES_MHS = [('nx', 'i4'),
                   ('ny', 'i4'),
                   ('year', 'i4'),
                   ('month', 'i4'),
                   ('day', 'i4'),
                   ('hour', 'i4'),
                   ('minute', 'i4'),
                   ('second', 'i4'),
                   ('lat', 'f4'),
                   ('lon', 'f4'),
                   ('sfccode', 'i4'),
                   ('tcwv', 'f4'),
                   ('T2m', 'f4')]
PIXEL_TYPES_MHS += [(f'Tb_{i}_{j}', 'f4') for i in range(10) for j in range(5)]
PIXEL_TYPES_MHS += [(f'sfcprcp_{i}', 'f4') for i in range(10)]
PIXEL_TYPES_MHS += [(f'cnvprcp_{i}', 'f4') for i in range(10)]


class MHSBinaryFile:
    """
    Class to extract data from GMI CSU binary files and store to
    NetCDF file.
    """
    def __init__(self, filename):
        """
        Open CSU binary file containing GPROF training data for
        MHS sensor.

        Args:
            filename(``pathlib.Path``): The file to open.
        """
        self.header = np.memmap(filename,
                                dtype=HEADER_TYPES_MHS,
                                mode="r",
                                shape=(1,))
        self.pixels = np.memmap(filename,
                                dtype=PIXEL_TYPES_MHS,
                                offset=70,
                                mode="r")

    @classmethod
    def create_output_file(cls, filename):
        """
        Create NetCDF output file.

        Args:
            filename(``pathlib.Path``): The filename of the output file.
        """
        file = netCDF4.Dataset(filename, "w")
        file.createDimension("viewing_angles", size=10)
        file.createDimension("channels", size=5)
        file.createDimension("samples", size=None)
        file.createVariable("brightness_temperature",
                            "f4",
                            dimensions=("samples",
                                        "viewing_angles",
                                        "channels"))
        file.createVariable("viewing_angles", "f4", dimensions=("viewing_angles",))
        file.createVariable("frequencies", "f4", dimensions=("channels",))

        file.createVariable("tbs_min", "f4", dimensions=("viewing_angles",
                                                         "channels"))
        file.createVariable("tbs_max", "f4", dimensions=("viewing_angles",
                                                         "channels"))
        file.createVariable("latitude", "f4", dimensions=("samples",))
        file.createVariable("longitude", "f4", dimensions=("samples",))
        file.createVariable("surface_type", "f4", dimensions=("samples",))
        file.createVariable("tcwv", "f4", dimensions=("samples"))
        file.createVariable("t2m", "f4", dimensions=("samples"))
        file.createVariable("surface_precipitation",
                            "f4",
                            dimensions=("samples",
                                        "viewing_angles"))
        file.createVariable("convective_precipitation",
                            "f4",
                            dimensions=("samples",
                                        "viewing_angles"))
        # Also include date.
        file.createVariable("year", "i4", dimensions=("samples",))
        file.createVariable("month", "i4", dimensions=("samples",))
        file.createVariable("day", "i4", dimensions=("samples",))
        file.createVariable("hour", "i4", dimensions=("samples",))
        file.createVariable("minute", "i4", dimensions=("samples",))
        file.createVariable("second", "i4", dimensions=("samples",))

        file["viewing_angles"][:] = [mhs_data.header[0][i] for i in range(7, 17)]
        file["frequencies"][:] = [mhs_data.header[0][i] for i in range(2, 7)]
        file["tbs_min"][:] = 1e30
        file["tbs_max"][:] = 0.0
        return file

    def write_to_file(self, file, samples=-1):
        """
        Write data to NetCDF file.

        Arguments
            file(``netCDF4.Dataset``): File handle to the output file.
            samples(``int``): How many samples to extract from the file.
        """
        v_tbs = file.variables["brightness_temperature"]
        v_lats = file.variables["latitude"]
        v_lons = file.variables["longitude"]
        v_sfccode = file.variables["surface_type"]
        v_tcwv = file.variables["tcwv"]
        v_t2m = file.variables["t2m"]
        v_surf_precip = file.variables["surface_precipitation"]
        v_conv_precip = file.variables["convective_precipitation"]
        v_tbs_min = file.variables["tbs_min"]
        v_tbs_max = file.variables["tbs_max"]

        v_year = file.variables["year"]
        v_month = file.variables["month"]
        v_day = file.variables["day"]
        v_hour = file.variables["hour"]
        v_minute = file.variables["minute"]
        v_second = file.variables["second"]

        n_samples = len(self.pixels)
        if samples < 0:
            samples = n_samples
        indices = np.random.permutation(np.arange(n_samples))

        n_extracted = 0
        for index in indices:
            if n_extracted >= samples:
                break
            if not check_sample(self.pixels[index]):
                continue
            d = self.pixels[index]
            i = file.dimensions["samples"].size

            v_year[i] = d[2]
            v_month[i] = d[3]
            v_day[i] = d[4]
            v_hour[i] = d[5]
            v_minute[i] = d[6]
            v_second[i] = d[7]

            v_lats[i] = d[8]
            v_lons[i] = d[9]
            v_sfccode[i] = d[10]
            v_tcwv[i] = d[11]
            v_t2m[i] = d[12]
            for j in range(10):
                for k in range(5):
                    tb = d[13 + j * 5 + k]
                    v_tbs_min[j, k] = np.minimum(v_tbs_min[j, k], tb)
                    v_tbs_max[j, k] = np.maximum(v_tbs_max[j, k], tb)
                    v_tbs[i, j, k] = tb
                sp = d[13 + 50 + j]
                v_surf_precip[i, j] = sp
                cp = d[13 + 60 + j]
                v_conv_precip[i, j] = cp
            n_extracted += 1




path = "/home/simonpf/Dendrite/UserAreas/Teo/MHS/1409/MHS.CSU.20140902.002900.dat"
mhs_data = MHSBinaryFile(path)
