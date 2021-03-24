"""
====================
regn.data.validation
====================

This module provides class to read in the MRMS validation data for GPM.
"""
from datetime import datetime, timedelta
import re
from pathlib import Path

import numpy as np
import xarray as xr

PRECIPRATE_REGEX = re.compile("PRECIPRATE\.GC\.(\d{8})\.(\d{6})\.(\d{5})\.dat\.gz")
MASK_REGEX = re.compile("MASK\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz")
RQI_REGEX = re.compile("RQI\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz")
FILE_REGEX = re.compile("[\w*\.]*\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz")

def list_overpasses(base_directory):
    """
    Collects the granule numbers and corresponding dattes for which ground validation
    data is available.

    Args:
        base_directory: Path to the directory that contains the validation data.

    Returns:
        Dictionary mapping granule numbers of GPM CO CONUS overpasses to dates.
    """
    granule_numbers = {}
    for path in Path(base_directory).glob("**/*.dat.gz"):
        m = PRECIPRATE_REGEX.match(path.name)
        if m:
            granule_number = int(m.group(3))

            time = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            granule_numbers[granule_number] = time
    return granule_numbers

def find_files(granule_number, base_directory):
    """
    Finds files corresponding to a given granule number.

    Args:
         granule_number: The granule number given as integer.
         base_directory: Root directory containing the validation data.

    Returns:
         List of files matching the given granule number.
    """
    files = []
    for path in Path(base_directory).glob("**/*.gz"):
        m = FILE_REGEX.match(path.name)
        if m:
            n = int(m.group(3))
            if n == granule_number:
                files.append(path)
    return files

def get_date(path):
    """
    Extract date from filename.

    Args:
        Path to a validation data file.

    Returns:
        datetime object representing the time to which the data in the
        file corresponds.
    """
    m = FILE_REGEX.match(path.name)
    if m:
        granule_number = int(m.group(3))
        time = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return time
    return None

def open_validation_dataset(granule_number, base_directory):
    """
    Open the validation data for a given granule number
    as xarray.Dataset.

    Args:
        granule_number: GPM granule number for which to open the validation
             data.
        base_directory: Path to root of the directory tree containing the
             validation data.

    Returns:
        xarray.Dataset containing the validation data.
    """
    files = find_files(granule_number, base_directory)

    # Load precip-rate data.
    precip_files = [f for f in files if PRECIPRATE_REGEX.match(f.name)]
    times = [get_date(f) for f in precip_files]

    header = np.loadtxt(files[0], usecols=(1,), max_rows=6)
    n_cols = int(header[0])
    n_rows = int(header[1])
    lon_ll = int(header[2])
    lat_ll = int(header[3])
    dl = int(header[4])

    lons = (lon_ll + np.arange(n_cols) * dl)[::-1]
    lats = lat_ll + np.arange(n_rows) * dl

    precip_files = sorted(precip_files, key=get_date)
    precip_rate = np.zeros((len(times), n_rows, n_cols))
    for i, f in enumerate(precip_files):
        precip_rate[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.float32)

    rqi_files = [f for f in files if RQI_REGEX.match(f.name)]
    rqi_files = sorted(rqi_files, key=get_date)
    rqi = np.zeros((len(times), n_rows, n_cols), dtype=np.int32)
    for i, f in enumerate(rqi_files):
        rqi[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.int32)

    mask_files = [f for f in files if MASK_REGEX.match(f.name)]
    mask_files = sorted(mask_files, key=get_date)
    mask = np.zeros((len(times), n_rows, n_cols), dtype=np.int32)
    for i, f in enumerate(mask_files):
        mask[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.int32)

    dims = ("time", "latitude", "longitude")
    data = {
        "latitude": (("latitude",), lats),
        "longitude": (("longitude",), lons),
        "time": (("time",), times),
        "precip_rate": (dims, precip_rate),
        "mask": (dims, mask),
        "radar_quality_index": (dims, rqi)
    }

    return xr.Dataset(data)

overpasses = list_overpasses("/home/simonpf/src/regn/data/validation")

files = find_files(16565, "/home/simonpf/src/regn/data/validation")
data = open_validation_dataset(16565, "/home/simonpf/src/regn/data/validation")
