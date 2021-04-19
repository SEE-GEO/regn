import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess

from tqdm import tqdm

from quantnn.qrnn import QRNN
from quantnn.normalizer import Normalizer
from regn.gprof import InputData

# Parse arguments
parser = argparse.ArgumentParser(description="Run GPROF on validation data.")
parser.add_argument('model', metavar="model", type=str, nargs=1,
                    help='The model to use for the retrieval.')
parser.add_argument('year', metavar="year", type=int, nargs=1,
                    help='The year for which to run the retrievals.')
parser.add_argument('month', metavar='month', type=int, nargs=1,
                    help='The month for which to run the retrievals.')
args = parser.parse_args()
model = args.model[0]
year = args.year[0]
month = args.month[0]

sensitivity_file = Path("/home/simon/GMI_ERA5_V7_chansens18.txt")
output_path = Path(f"/gdata/simon/validation/qprof/{year}/{month:02}")
output_path.mkdir(exist_ok=True, parents=True)
log_path = Path("/home/simon/src/GPROF_2020_V1_4D_prf/log/")

# Find validation files.
path = Path(f"/gdata/simon/validation/preprocessor/{year}/{month:02}")
files = path.glob("*.pp")

# Load qrnn model
qrnn = QRNN.load(model)
qrnn.quantile_axis = 1
normalizer = Normalizer.load("sftp://129.16.35.202/mnt/array1/share/MLDatasets/gprof/simple/gprof_gmi_normalizer.pckl")


def run_retrieval(f):
    stem = f.stem
    qrnn = QRNN.load(model)
    qrnn.quantile_axis = 1
    qrnn.bin_axis = 1
    normalizer = Normalizer.load("sftp://129.16.35.202/mnt/array1/share/MLDatasets/gprof/simple/gprof_gmi_normalizer.pckl")
    input_data = InputData(f, normalizer, 256)
    results = input_data.run_retrieval_conv(qrnn)
    output_file = input_data.write_retrieval_results(output_path, results)
    subprocess.run(["gzip", "-f", str(output_file)])

pool = ProcessPoolExecutor(max_workers=8)
tasks = []
for f in files:
    tasks.append(pool.submit(run_retrieval, f))

for t in tqdm(tasks):
    t.result()
