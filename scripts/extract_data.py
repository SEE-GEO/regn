import glob
import os
import numpy as np
import tqdm
from calendar import monthrange
from regn.data.csu_data import (get_files,
                                get_gprof_file,
                                GMIBinaryFile,
                                MHSBinaryFile,
                                GPROFBinaryFile)
import argparse

sensor_classes = {"mhs": MHSBinaryFile,
                  "gmi": GMIBinaryFile}

###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(description='Extract training data')
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
# Extract data.
###############################################################################

months = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
years = [14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15]

sensor_class = sensor_classes[sensor]
output_file = sensor_class.create_output_file(output)
for y, m in zip(years, months):
    _, n_days = monthrange(2000 + y, m)


    if type == "training":
        days = np.arange(5, n_days + 1)
    elif type == "validation":
        days = np.arange(3, 5)
    else:
        days = np.arange(1, 2)

    print("Processing {}/{}".format(y, m))
    for d in tqdm.tqdm(days):
        files = get_files(data_path, y, m, d)
        for input_file in files:
            n_before = output_file.dimensions["samples"].size
            sensor_class(input_file).write_to_file(output_file, samples)
            n_after = output_file.dimensions["samples"].size

            if include_gprof_data:
                nx = output_file["nx"][n_before: n_after]
                ny = output_file["ny"][n_before: n_after]
                gprof_file = get_gprof_file(input_file)
                GPROFBinaryFile(gprof_file).add_to_file(output_file, nx, ny)




output_file.close()
