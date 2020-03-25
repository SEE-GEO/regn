import numpy as np
import glob
import os
import netCDF4
import torch
from torch.utils.data import Dataset

################################################################################
# Torch dataloader interface
################################################################################

class GprofData(Dataset):
    """
    Pytorch dataset for the Gprof training data.

    This class is a wrapper around the netCDF4 files that are used to store
    the GProf training data. It provides as input vector the brightness
    temperatures and as output vector the surface precipitation.

    """
    def __init__(self,
                 path,
                 batch_size = None,
                 surface_type = -1):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size
            surface_type: If positive, only samples of the given surface type
                are provided. Otherwise the surface type index is provided as
                input features.
        """
        super().__init__()
        self.batch_size = batch_size

        self.file = netCDF4.Dataset(path, mode = "r")

        tcwv = self.file.variables["tcwv"][:]
        surface_type_data = self.file.variables["surface_type"][:]
        t2m = self.file.variables["t2m"][:]
        bt = self.file.variables["brightness_temperature"][:]
        bt_min = self.file.variables["bt_min"][:].reshape(1, -1)
        bt_max = self.file.variables["bt_max"][:].reshape(1, -1)

        valid = surface_type_data > 0
        valid *= t2m > 0
        valid *= tcwv > 0
        if surface_type > 0:
            valid *= surface_type_data == surface_type

        inds = np.random.permutation(np.where(valid)[0])

        tcwv = tcwv[inds].reshape(-1, 1)
        tcwv_min = tcwv.min()
        tcwv_max = tcwv.max()
        tcwv = (tcwv - tcwv.min()) / (tcwv.max() - tcwv.min())

        t2m = t2m[inds].reshape(-1, 1)
        t2m_min = t2m.min()
        t2m_max = t2m.max()
        t2m = (t2m - t2m.min()) / (t2m.max() - t2m.min())

        surface_type_data = surface_type_data[inds].reshape(-1, 1)
        surface_type_min = surface_type_data.min()
        surface_type_max = surface_type_data.max()
        surface_type_data = ((surface_type_data - surface_type_min)
                             / (surface_type_max - surface_type_min))

        bt = bt[inds]
        bt = (bt - bt_min) / (bt_max - bt_min)

        if surface_type < 0:
            self.x = np.concatenate([bt, t2m, tcwv, surface_type_data], axis=1)
        else:
            self.x = np.concatenate([bt, t2m, tcwv], axis=1)
        self.x = self.x.data
        self.y = self.file.variables["surface_precipitation"][inds]
        self.y = self.y.data

    def __len__(self):
        """
        The number of entries in the training data. This is part of the
        pytorch interface for datasets.

        Return:
            int: The number of samples in the data set
        """
        if self.batch_size is None:
            return self.x.shape[0]
        else:
            return self.x.shape[0] // self.batch_size

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if self.batch_size is None:
            return (torch.tensor(self.x[[i], :]),
                    torch.tensor(self.y[[i]]))
        else:
            i_start = self.batch_size * i
            i_end = self.batch_size * (i + 1)
            return (torch.tensor(self.x[i_start : i_end, :]),
                    torch.tensor(self.y[i_start : i_end]))

################################################################################
# Interface to binary data.
################################################################################

types =  [('nx', 'i4'), ('ny', 'i4')]
types += [('year', 'i4'), ('month', 'i4'), ('day', 'i4'), ('hour', 'i4'), ('minute', 'i4'), ('second', 'i4')]
types +=  [('lat', 'f4'), ('lon', 'f4')]
types += [('sfccode', 'i4'), ('tcwv', 'i4'), ('T2m', 'i4')]
types += [('Tb_{}'.format(i), 'f4') for i in range(13)]
types += [('sfcprcp', 'f4'), ('cnvprcp', 'f4')]

def read_file(f):
    """
    Read GPM binary file.

    Arguments:
        f(str): Filename of file to read

    Returns:
        numpy memmap object pointing to the data.
    """
    data = np.memmap(f,
                     dtype = types,
                     mode =  "r",
                     offset = 10 + 26 * 4)
    return data

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

def write_to_file(file, data, samples = -1):
    """
    Write data to NetCDF file.

    Arguments
        file: File handle to the output file.
        data: numpy memmap object pointing to the binary data
    """
    v_tbs = file.variables["brightness_temperature"]
    v_lats = file.variables["latitude"]
    v_lons = file.variables["longitude"]
    v_sfccode = file.variables["surface_type"]
    v_tcwv = file.variables["tcwv"]
    v_t2m = file.variables["t2m"]
    v_surf_precip = file.variables["surface_precipitation"]
    v_tbs_min = file.variables["bt_min"]
    v_tbs_max = file.variables["bt_max"]

    v_year = file.variables["year"]
    v_month = file.variables["month"]
    v_day = file.variables["day"]
    v_hour = file.variables["hour"]
    v_minute = file.variables["minute"]
    v_second = file.variables["second"]

    i = file.dimensions["samples"].size

    inds = np.random.permutation(np.arange(data.shape[0]))
    j = 0
    for _ in range(samples):
        while j < data.shape[0] and not check_sample(data[inds[j]]):
            j = j+1
        if j >= data.shape[0]:
            break
        d = data[inds[j]]

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
        for k in range(13):
            v_tbs_min[k] = np.minimum(v_tbs_min[k], d[13 + k])
            v_tbs_max[k] = np.maximum(v_tbs_max[k], d[13 + k])
            v_tbs[i, k] = d[13 + k]
        v_surf_precip[i] = d[26]
        i += 1
        j += 1

def create_output_file(path):
    """
    Creates netCDF4 output file to store training data in.

    Arguments:
        path: Filename of the file to create.

    Returns:
        netCDF4 Dataset object pointing to the created file
    """
    file = netCDF4.Dataset(path, "w")
    file.createDimension("channels", size = 13)
    file.createDimension("samples", size = None) #unlimited dimensions
    file.createVariable("brightness_temperature", "f4", dimensions = ("samples", "channels"))
    file.createVariable("bt_min", "f4", dimensions = ("channels"))
    file.createVariable("bt_max", "f4", dimensions = ("channels"))
    file.createVariable("latitude", "f4", dimensions = ("samples",))
    file.createVariable("longitude", "f4", dimensions = ("samples",))
    file.createVariable("surface_type", "f4", dimensions = ("samples",))
    file.createVariable("tcwv", "f4", dimensions = ("samples",))
    file.createVariable("t2m", "f4", dimensions = ("samples",))
    file.createVariable("surface_precipitation", "f4", dimensions = ("samples",))
    # Also include date.
    file.createVariable("year", "i4", dimensions = ("samples",))
    file.createVariable("month", "i4", dimensions = ("samples",))
    file.createVariable("day", "i4", dimensions = ("samples",))
    file.createVariable("hour", "i4", dimensions = ("samples",))
    file.createVariable("minute", "i4", dimensions = ("samples",))
    file.createVariable("second", "i4", dimensions = ("samples",))

    file["bt_min"][:] = 1e30
    file["bt_max"][:] = 0.0

    return file

def extract_data(base_path,
                 year,
                 month,
                 day,
                 samples,
                 file):
    """
    Extract training data from binary files for given year, month and day.

    Arguments:
        base_path(str): Base directory containing the training data.
        year(int): The year from which to extract the training data.
        month(int): The month from which to extract the training data.
        day(int): The day for which to extract training data.
        file: File handle of the netCDF4 file into which to store the results.
    """
    path = os.path.join(base_path,
                        "{:02d}{:02d}".format(year, month),
                        "*20{:02d}{:02d}{:02d}*.dat".format(year, month, day))
    files = glob.glob(path)
    if len(files) > 0:
        samples_per_file = samples // len(files)
        for i, f in enumerate(files):
            n = samples_per_file
            if i < samples % len(files):
                n += 1
            with open(f, 'rb') as fn:
                data = read_file(fn)
                write_to_file(file, data, samples = n)
