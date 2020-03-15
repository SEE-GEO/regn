import numpy.np
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

def write_to_file(file, data, subsampling = 0.01):
    """
    Write data to NetCDF file.

    Arguments
        file: File handle to the output file.
        data: numpy memmap object pointing to the binary data
    """
    v_tbs = file.variables["tbs"]
    v_lats = file.variables["lats"]
    v_lons = file.variables["lons"]
    v_sfccode = file.variables["sfccode"]
    v_tcwv = file.variables["tcwv"]
    v_t2m = file.variables["T2m"]
    v_surf_precip = file.variables["surface_precipitation"]
    v_tbs_min = file.variables["tbs_min"]
    v_tbs_max = file.variables["tbs_max"]

    v_year = file.variables["year"]
    v_month = file.variables["month"]
    v_day = file.variables["day"]
    v_hour = file.variables["hour"]
    v_minute = file.variables["minute"]
    v_second = file.variables["second"]

    i = file.dimensions["samples"].size
    for d in data:

        if np.random.rand() > subsampling:
            continue

        # Check if sample is valid.
        if not check_sample(d):
            continue

        v_year[i] = d[2]
        v_month[i] = d[3]
        v_day[i] = d[4]
        v_hour[i] = d[5]
        v_year[i] = d[6]
        v_second[i] = d[7]

        v_lats[i] = d[8]
        v_lons[i] = d[9]
        v_sfccode[i] = d[10]
        v_tcwv[i] = d[11]
        v_t2m[i] = d[12]
        for j in range(13):
            v_tbs_min[j] = np.minimum(v_tbs_min[j], d[13 + j])
            v_tbs_max[j] = np.maximum(v_tbs_max[j], d[13 + j])
            v_tbs[i, j] = d[13 + j]
        v_surf_precip[i] = d[26]
        i += 1

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
    file.createVariable("tbs", "f4", dimensions = ("samples", "channels"))
    file.createVariable("tbs_min", "f4", dimensions = ("channels"))
    file.createVariable("tbs_max", "f4", dimensions = ("channels"))
    file.createVariable("lats", "f4", dimensions = ("samples",))
    file.createVariable("lons", "f4", dimensions = ("samples",))
    file.createVariable("sfccode", "f4", dimensions = ("samples",))
    file.createVariable("tcwv", "f4", dimensions = ("samples",))
    file.createVariable("T2m", "f4", dimensions = ("samples",))
    file.createVariable("surface_precipitation", "f4", dimensions = ("samples",))
    # Also include date.
    file.createVariable("year", "i4", dimensions = ("samples",))
    file.createVariable("month", "i4", dimensions = ("samples",))
    file.createVariable("day", "i4", dimensions = ("samples",))
    file.createVariable("hour", "i4", dimensions = ("samples",))
    file.createVariable("minute", "i4", dimensions = ("samples",))
    file.createVariable("second", "i4", dimensions = ("samples",))

    file["tbs_min"][:] = 1e30
    file["tbs_max"][:] = 0.0

    return file

def extract_data(base_path, file, subsampling = 0.01):
    """
    Extract training data from GPROF binary files for given year, month and day.

    Arguments:
        year(int): The year from which to extract the training data.
        month(int): The month from which to extract the training data.
        day(int): The day for which to extract training data.
        file: File handle of the netCDF4 file into which to store the results.
    """
    files = glob.glob(os.path.join(base_path, "**", "*.dat"))
    for f in tqdm.tqdm(files):
        with open(f, 'rb') as fn:
                data = read_file(fn)
                write_to_file(f, data, subsampling = subsampling)
