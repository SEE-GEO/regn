import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess

from tqdm import tqdm

# Parse arguments
parser = argparse.ArgumentParser(description="Run GPROF on validation data.")
parser.add_argument('year', metavar="year", type=int, nargs=1,
                    help='The year for which to run the retrievals.')
parser.add_argument('month', metavar='month', type=int, nargs=1,
                    help='The month for which to run the retrievals.')
args = parser.parse_args()
year = args.year[0]
month = args.month[0]

sensitivity_file = Path("/home/simon/GMI_ERA5_V7_chansens18.txt")
output_path = Path(f"/gdata/simon/validation/gprof/{year}/{month:02}")
output_path.mkdir(exist_ok=True, parents=True)
log_path = Path("/home/simon/src/GPROF_2020_V1_4D_prf/log/")

# Find validation files.
path = Path(f"/gdata/simon/validation/preprocessor/{year}/{month:02}")
files = path.glob("*.pp")

def run_retrieval(f):
    stem = f.stem
    subprocess.run(["GPROF_2020_V1",
                    str(f),
                    str(output_path / (stem + ".pp")),
                    str(log_path / (stem + ".log")),
                    "/qdata1/pbrown/gpm/ancillary/",
                    "0"])

pool = ProcessPoolExecutor(max_workers=8)
tasks = []
for f in files:
    tasks.append(pool.submit(run_retrieval, f))

for t in tqdm(tasks):
    t.result()
