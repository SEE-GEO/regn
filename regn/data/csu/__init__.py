import asyncio
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import re


from netCDF4 import Dataset
import numpy as np

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
     ("integrated_water_vapor", np.float32),
     ("total_column_water_vapor", np.float32),
     ("two_meter_temperature", np.float32),
     ("rain_water_content", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
     ("cloud_water_content", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
     ("snow_water_content", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
     ("latent_heat", [(f"l_{i:02}", np.float32) for i in range(N_LAYERS)]),
     ])

class GPROFGMIBinFile:
    """
    File to read GPROF GMI .bin files.
    """
    def __init__(self, filename):
        """
        Open file.

        Args:
             filename(``str``): File to open
        """
        self.filename = filename
        parts = Path(filename).name[:-4].split("_")
        self.temperature = float(parts[1])
        self.tpw = float(parts[2])
        self.surface_type = int(parts[-1])

        if len(parts) == 4:
            self.airmass_type = 0
        elif len(parts) == 5:
            self.airmass_type = parts[-2]
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


async def load_data_async(self, loop, pool, filename, start=0.0, end=0.0):
    """
    Asynchornous loading of data from bin file.

    Args:
        loop: Event loop in which to perform the loading
        pool: Process pool to use to offload the loading.
        filename: Filename of the .bin file to read.
        start: Fractional position from which to start reading the data.
        end: Fractional position up to which to read the data.

    Returns:
        Future containing the data dictionary to load.
    """
    def run():
        input_file = GPROFGMIBinFile(filename)
        return input_file.load_data(start, end)
    return loop.run_in_executor(pool, run)


EXPECTED_DIMENSIONS = {N_FREQS: "channel",
                        N_LAYERS: "layers",
                        None: "samples"}


class GPROFGMIOutputFile:


    def __init__(self, filename):
        self.filename = filename
        self.handle = Dataset(filename, "w")
        self.handle.close()
        self.lock = asyncio.Lock()

    def _initialize_file(self, data):

        for n, name in EXPECTED_DIMENSIONS.items():
            self.handle.createDimension(name, n)

        for k in data:
            d = data[k]
            dims = ("samples",)
            dims += tuple([EXPECTED_DIMENSIONS[i] for i in d.shape[1:]])
            self.handle.createVariable(k, d.dtype, dims)

    def add_attributes(self, attributes):
        for k in attributes:
            self.handle = Dataset(self.filename, "r+")
            self.handle.__dict__[k] = attributes[k]

    def add_data(self, data):
        self.handle = Dataset(self.filename, "r+")

        if len(self.handle.variables) == 0:
            self._initialize_file(data)

        i = self.handle.dimensions["samples"].size
        for k in data:
            v = self.handle.variables[k]
            d = data[k]
            n = d.shape[0]
            v[i:i + n] = d

        self.handle.close()

    async def add_data_async(self, loop, pool, data):
        async with self.lock:
            loop.run_in_executor(pool, self.add_data, data)

    def __repr__(self):
        return (f"GPROFGMIOutputFile(filename={self.filename})")



GPM_FILE_REGEXP = re.compile("gpm_(\d\d\d)_(\d\d)(_(\d\d))?_(\d\d).bin")


async def process_input(loop, pook, input_filename, output_file):
    print(f"Starting processing {input_filename}")
    input_file = GPROFGMIBinFile(input_filename)
    data = await input_file.get_data_sync(loop, pool)
    output_file.add_data_async(loop, pool, data)
    print(f"Finished processing {input_filename}")


class FileProcessor:
    def __init__(self,
                 path,
                 tpw_min=0.0,
                 tpw_max=76.0,
                 sst_min=227.0,
                 sst_max=307.0):
        self.path = path

        self.files = []
        for f in Path(path).iterdir():
            print(f)
            match = re.match(GPM_FILE_REGEXP, f.name)
            if match:
                self.files.append(f)


    def process_async(self,
                      output_file,
                      start_fraction,
                      end_fraction,
                      n_processes=4):

        pool = ProcessPoolExecutor(max_workers=n_processes)
        loop = asyncio.get_event_loop()

        output_file = GPROFGMIOutputFile(output_file)
        input_file = GPROFGMIBinFile(self.files[0])
        output_file.add_attributes(input_file.get_attributes())

        def coro():
            tasks = [process_input(loop, pool, f, output_file)
                     for f in self.files]
            return tasks

        loop.run_until_complete(asyncio.wait(coro()))
        loop.close()
