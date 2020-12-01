"""
Command line program to extract training, validation and test data from GPROF
binary files.
"""
import argparse
from calendar import monthrange
from concurrent.futures import ProcessPoolExecutor
from tempfile import mkstemp
import os

import numpy as np
import tqdm
from regn.data.csu_data import (get_files,
                                get_gprof_file,
                                GMIBinaryFile,
                                MHSBinaryFile,
                                GPROFBinaryFile)
import xarray

sensor_classes = {"mhs": MHSBinaryFile,
                  "gmi": GMIBinaryFile}

executor = ProcessPoolExecutor(12)

###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(description="Extract data from GPROF files.")
parser.add_argument('input', metavar='input', type=str, nargs=1,
                    help='Path to input data.')
parser.add_argument('n_samples', metavar='n', type=int, nargs=1,
                    help='How many samples to extract from each file.')
parser.add_argument('output', metavar="output", type=str, nargs=1,
                    help='Filename for output file.')
parser.add_argument('--sensor', nargs=1, type=str, default="mhs",
                    help="Which sensor the data comes from (GMI or MHS).")
parser.add_argument('--type', nargs=1, type=str, default="training",
                    help="The type of dataset to extract (training, test, "
                    " or validation.)")
parser.add_argument('--include_gprof', action="store_true",
                    help="Whether or not to include GPROF reference values"
                    "in the data.")

args = parser.parse_args()
data_path = args.input[0]
samples = args.n_samples[0]
output = args.output[0]
sensor = args.sensor[0].lower()
type = args.type[0].lower()
include_gprof_data = args.include_gprof

###############################################################################
# Helper functions.
###############################################################################

def extract_data(output_file,
                 input_file,
                 filename,
                 samples,
                 start_index=None):
    """
    Extract data from input file and write to output file.

    Args:
        output_file: Handle to neCDF4 file to which to write the data.
        input_file: The input file from which to read the data
        samples: The number of samples to extract from the file.
        start_index: The start index in the output file at which to
            start adding the data.
    """

    if start_index:
        n_before = start_index
        n_after = start_index + samples
    else:
        n_before = output_file.dimensions["samples"].size
        n_after = n_before + samples

    input_file.write_to_file(output_file,
                             samples,
                             start_index)

    if include_gprof_data:
        lats = output_file["latitude"][n_before: n_after]
        lons = output_file["longitude"][n_before: n_after]
        gprof_file = get_gprof_file(filename)
        GPROFBinaryFile(gprof_file).add_to_file(output_file, lats, lons)

def open_input_file(sensor_class, filename):
    """ Open file object corresponding to sensor class. """
    return sensor_class(filename)

def process_month(year, month):
    """
    Process data for given month and store data in temporary netCDF4 file.

    Args:
        year(int): The year from which to process the data.
        month(int): The month from which to process the data.

    Return:
        The filename of the temporary file that contains the data from the
        given month.
    """
    _, n_days = monthrange(2000 + year, month)
    if type == "training":
        days = np.arange(5, n_days + 1)
    elif type == "validation":
        days = np.arange(3, 5)
    else:
        days = np.arange(1, 2)

    _, output_file = mkstemp()

    with sensor_class.create_output_file(output_file) as netcdf_file:
        for day in tqdm.tqdm(days):
            files = get_files(data_path, year, month, day)

            for file in files:
                input_file = sensor_class(file)
                extract_data(netcdf_file,
                             input_file,
                             file,
                             samples)
    return output_file

###############################################################################
# Run extraction.
###############################################################################

months = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
years = [14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15]
sensor_class = sensor_classes[sensor]

tasks = []
for y, m in zip(years, months):
    tasks.append(executor.submit(process_month, y, m))
output_files = [t.result() for t in tasks]
with xarray.open_mfdataset(output_files,
                           combine="nested",
                           concat_dim="samples") as combined_data:
    combined_data.to_netcdf(output)
for file in output_files:
    os.remove(file)
