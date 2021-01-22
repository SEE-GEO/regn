"""
=================
regn.data.csu.bin
=================

This module contains function to read and convert CSU .bin data for GPROF
 v. 7.
"""
import asyncio
import contextlib
import logging
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import re

from netCDF4 import Dataset
import numpy as np
import tqdm.asyncio

LOGGER = logging.getLogger(__name__)

N_LAYERS = 28
N_FREQS = 15
GMI_BIN_HEADER_TYPES = np.dtype(
    [("satellite_code", "a5"),
     ("sensor", "a5"),
     ("frequencies", [(f"f_{i:02}", np.float32) for i in range(N_FREQS)]),
     ("nominal_eia", [(f"f_{i:02}", np.float32) for i in range(N_FREQS)]),
     ])


GMI_BIN_RECORD_TYPES = np.dtype(
    [("dataset_number", "i4"),
     ("surface_precip", np.float32),
     ("convective_precip", np.float32),
     ("brightness_temps", [(f"tb_{i:02}", np.float32) for i in range(N_FREQS)]),
     ("delta_tb", [(f"tb_{i:02}", np.float32) for i in range(N_FREQS)]),
     ("rain_water_path", np.float32),
     ("cloud_water_path", np.float32),
     ("ice_water_path", np.float32),
     ("total_column_water_vapor", np.float32),
     ("two_meter_temperature", np.float32),
     ("rain_water_content", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
     ("cloud_water_content", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
     ("snow_water_content", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
     ("latent_heat", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
     ])

PROFILE_NAMES = ["rain_water_content",
                 "cloud_water_content",
                 "snow_water_content",
                 "latent_heat"]

###############################################################################
# Input file.
###############################################################################

class GPROFGMIBinFile:
    """
    This class can be used to read a GPROF v7 .bin file.

    Attributes:
        temperature(``float``): The surface temperature corresponding to the
            bin.
        tpw(``float``): The total precipitable water corresponding to the bin.
        surface_type(``int``): The surface type corresponding to the bin.
        airmass_type(``int``): The airmass type corresponding to the bin.
        header: Structured numpy array containing the file header data.
        handle: Structures numpy array containing the file of the data.
    """
    def __init__(self,
                 filename,
                 include_profiles=False):
        """
        Open file.

        Args:
            filename(``str``): File to open
            include_profiles(``bool``): Whether or not to include profiles
                 in the extracted data.
        """
        self.filename = filename
        self.include_profiles = include_profiles

        parts = Path(filename).name[:-4].split("_")
        self.temperature = float(parts[1])
        self.tpw = float(parts[2])
        self.surface_type = int(parts[-1])

        if len(parts) == 4:
            self.airmass_type = 0
        elif len(parts) == 5:
            self.airmass_type = int(parts[-2])
        else:
            raise Exception(
                f"Filename {filename} does not match expected format!"
            )

        # Read the header
        self.header = np.fromfile(self.filename,
                                  GMI_BIN_HEADER_TYPES,
                                  count=1)

        self.handle = np.fromfile(self.filename,
                                  GMI_BIN_RECORD_TYPES,
                                  offset=GMI_BIN_HEADER_TYPES.itemsize)
        self.n_profiles = self.handle.shape[0]

    def get_attributes(self):
        """
        Return file header as dictionary of attributes.

        Returns:
            Dictionary containing frequencies and nominal earth incidence
            angles in this file.
        """
        attributes = {
            "frequencies": self.header["frequencies"].view("15f4"),
            "nominal_eia": self.header["nominal_eia"].view("15f4"),
        }
        return attributes

    def load_data(self, start=0.0, end=1.0):
        """
        Load data as dictionary of variables.

        Args:
            start: Fractional position from which to start reading the data.
            end: Fractional position up to which to read the data.

        Returns:
            Dictionary containing each database variables as numpy array.
        """
        n_start = int(start * self.n_profiles)
        n_end = int(end * self.n_profiles)

        results = {}
        for k, t in GMI_BIN_RECORD_TYPES.descr:

            if (not self.include_profiles) and k in PROFILE_NAMES:
                continue
            if type(t) is str:
                view = self.handle[k].view(t)
                results[k] = view[n_start:n_end]
            else:
                view = self.handle[k].view(f"{len(t)}{t[0][1]}")
                results[k] = view[n_start:n_end]
        results["surface_type"] = self.surface_type * np.ones(1, dtype=np.int)
        results["airmass_type"] = self.airmass_type * np.ones(1, dtype=np.int)
        results["tpw"] = self.tpw * np.ones(1, dtype=np.float)
        results["temperature"] = self.temperature * np.ones(1)
        return results

def load_data(filename, start=0.0, end=1.0):
    """
    Wrapper function to load data from a file.

    Args:
        filename: The path of the file to load the data from.
        start: Fractional position from which to start reading the data.
        end: Fractional position up to which to read the data.

    Returns:
        Dictionary containing each database variables as numpy array.
    """
    input_file = GPROFGMIBinFile(filename)
    return input_file.load_data(start, end)

###############################################################################
# Output file.
###############################################################################


EXPECTED_DIMENSIONS = {N_FREQS: "channel",
                       N_LAYERS: "layers",
                       None: "samples"}


class GPROFGMIOutputFile:
    """
    The GPROFGMIOutputFile class manages a NetCDF4 file, which is used to
    store the extracted GPROF data.

    The data in the file consists of GMI observations and corresponding
    precipitation. Observations are stored along an infinite dimension
    `samples`.
    """
    def __init__(self, filename):
        """
        Create a new output file with the given name.
        """
        print("Createin output file: ", filename)
        self.filename = filename
        Dataset(filename, "w").close()
        manager = multiprocessing.Manager()
        self.lock = manager.Lock()

    @property
    def handle(self):
        """


        """
        return contextlib.closing(Dataset(self.filename, "r+"))


    def add_attributes(self, attributes):
        """
        Add attributes to file.

        Args:
            attributes: Dictionary of attibutes to add to the file.
        """
        with self.handle as handle:
            for k in attributes:
                setattr(handle, k, attributes[k].ravel())

    def _initialize_file(self, data, handle):
        """
        Initializes file variable when data is added for the
        first time.

        Args:
            data: Dictionary containing variable names and data
                to add to store in the output file.
        """
        for n, name in EXPECTED_DIMENSIONS.items():
            handle.createDimension(name, n)

        for k in data:
            d = data[k]
            dims = ("samples",)
            dims += tuple([EXPECTED_DIMENSIONS[i] for i in d.shape[1:]])
            handle.createVariable(k, d.dtype, dims)

    def add_data(self, data):
        """
        Adds the given data to the file along the samples dimension.

        Args:
            data: Dictionary containing variable names and data
                to add to store in the output file.
        """
        with self.lock:
            with self.handle as handle:
                if len(handle.variables) == 0:
                    self._initialize_file(data, handle)

                n = 0
                for k in data:
                    n = max(data[k].shape[0], n)
                    if data[k].shape[0] == 0:
                        return

                i = handle.dimensions["samples"].size
                for k in data:
                    v = handle.variables[k]
                    d = data[k]
                    if d.size == 1:
                        v[i:i + n] = d[0]
                    else:
                        v[i:i + n] = d

    def __repr__(self):
        return (f"GPROFGMIOutputFile(filename={self.filename})")

###############################################################################
# File processor.
###############################################################################

GPM_FILE_REGEXP = re.compile(r"gpm_(\d\d\d)_(\d\d)(_(\d\d))?_(\d\d).bin")


def _process_input(input_filename,
                   output_file,
                   start=1.0,
                   end=1.0):
    data = load_data(input_filename, start, end)
    output_file.add_data(data)

async def process_input(loop,
                        pool,
                        input_filename,
                        output_file,
                        start=0.0,
                        end=1.0):
    """
    Asynchronous processing of an intput file.

    Args:
        loop: Event loop to execute in.
        pool: Executor to use for concurrent processing.
        input_filename: The input file to process.
        output_file: The output file object to store the data in.
        output_file_lock: asyncio.lock to use to gain access to output file.
        start: Fractional position from which to start reading the data.
        end: Fractional position up to which to read the data.
    """
    await loop.run_in_executor(pool, _process_input, input_filename, output_file, start, end)


class FileProcessor:
    """
    File processor class to process GPROF .bin files in given folder.
    """
    def __init__(self,
                 path,
                 st_min=227.0,
                 st_max=307.0,
                 tpw_min=0.0,
                 tpw_max=76.0):
        """
        Create file processor to process file in given path.

        Args:
            path: The path containing the files to process.
            st_min: The minimum bin surface temperature for which to consider bins.
            st_max: The maximum bin surface temperature for which to consider bins.
            tpw_min: The minimum bin-tpw value to consider.
            tpw_max: The maximum bin-tpw value to consider.

        """
        self.path = path

        self.files = []

        for f in Path(path).iterdir():
            match = re.match(GPM_FILE_REGEXP, f.name)
            if match:
                groups = match.groups()
                t = float(groups[0])
                tpw = float(groups[1])
                if t < st_min or t > st_max or tpw < tpw_min or tpw > tpw_max:
                    continue
                self.files.append(f)


    def run_async(self,
                  output_file,
                  start_fraction,
                  end_fraction,
                  n_processes=4):
        """
        Asynchronous processing of files in folder.

        Args:
            output_file(``str``): Filename of the output file.
            start_fraction(``float``): Fractional start value for the observations
                 to extract from each bin file.
            end_fraction(``float``): Fractional end value for the observations
                 to extract from each bin file.
            n_processes(``int``): How many processes to use for the parallel reading
                 of input files.
        """
        pool = ProcessPoolExecutor(max_workers=n_processes)
        loop = asyncio.new_event_loop()

        output_file = GPROFGMIOutputFile(output_file)
        input_file = GPROFGMIBinFile(self.files[0])
        output_file.add_attributes(input_file.get_attributes())

        async def coro():
            tasks = [process_input(loop,
                                   pool,
                                   f,
                                   output_file,
                                   start=start_fraction,
                                   end=end_fraction)
                     for f in self.files]
            for t in tqdm.asyncio.tqdm.as_completed(tasks):
                await t

        loop.run_until_complete(coro())
        loop.close()
