"""
Command line program to extract convolutional training, validation and test
 data from GPM .sim and L1C files.
"""
import argparse
from regn.data.csu.sim import SimFileProcessor

# Parse arguments
parser = argparse.ArgumentParser(description="Extract data from GPROF files.")
parser.add_argument('sim_file_path', metavar='<sim_file_path>', type=str, nargs=1,
                    help='Path to the root of the directory tree containing the .sim files.')
parser.add_argument('l1c_path', metavar="<l1c_path>", type=str, nargs=1,
                    help='Path to the root of the directory tree containing the L1C files.')
parser.add_argument('output_file', metavar="<output_path>", type=str, nargs=1,
                    help='Name of the outputfile to store the extracted data to')
parser.add_argument('--days', metavar='[...]', type=float, nargs="+",
                    help='Day indices for which to extract data for each month.')
args = parser.parse_args()
sim_file_path = args.sim_file.path[0]
l1c_path = args.l1c_path[0]
output_file = args.output_file[0]
days = args.days

# Run processing.
processor = SimFileProcessor(sim_file_path,
                             l1c_path,
                             output_file,
                             days)
processor.run(output_file)

