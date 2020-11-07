import glob
import os
import numpy as np
import tqdm
from calendar import monthrange
from regn.data.csu_data import (get_files,
                                GMIBinaryFile,
                                MHSBinaryFile)
import argparse

sensor_classes = {"mhs": MHSBinaryFile,
                  "gmi": GMIBinaryFile}

################################################################################
# Command line arguments.
################################################################################

parser = argparse.ArgumentParser(description='Extract training data')
parser.add_argument('input', metavar='input', type=str, nargs=1,
                    help='Path to input data.')
parser.add_argument('n_samples', metavar='n', type=int, nargs=1,
                    help='How many samples to extract from each file.')
parser.add_argument('output', metavar="output", type=str, nargs=1,
                    help='Filename for output file.')
parser.add_argument('--sensor', nargs=1, type=str, default="mhs")

args = parser.parse_args()
data_path = args.input[0]
samples = args.n_samples[0]
output = args.output[0]
sensor = args.sensor[0].lower()

################################################################################
# Extract data.
################################################################################

months = [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
years = [14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15]

sensor_class = sensor_classes[sensor]
output_file = sensor_class.create_output_file(output)
for y, m in zip(years, months):
    _, n_days = monthrange(2000 + y, m)
    days = np.arange(5, n_days + 1)
    print("Processing {}/{}".format(y, m))
    for d in tqdm.tqdm(days):
        files = get_files(data_path, y, m, d)
        for input_file in files:
            sensor_class(input_file).write_to_file(output_file, samples)
output_file.close()
