import numpy as np
import glob
import os
import netCDF4

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
    Check that brightness temperatures are within a valid range.

    Arguments:
        data: The data array containing the data of one training sample.

    Return:
        Whether the given sample contains valid tbs.
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

def extract_data(base_path, y, m, d, samples, file):
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
                        "{:02d}{:02d}".format(y, m),
                        "*20{:02d}{:02d}{:02d}*.dat".format(y, m, d))
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
