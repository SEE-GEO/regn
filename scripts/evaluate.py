"""
Command line program to run model on validation data.
"""
import argparse
from pathlib import Path

import torch

from regn.data.csu.training_data import GPROFValidationDataset
from quantnn.normalizer import Normalizer
from quantnn.qrnn import QRNN

# Parse arguments
parser = argparse.ArgumentParser(description="Run model(s) on validation data.")
parser.add_argument('validation_data', metavar="validation_data", type=str, nargs=1,
                    help='Path to the validation data')
parser.add_argument('models', metavar='models', type=str, nargs="+",
                    help='Path to the models to evaluate')

args = parser.parse_args()
validation_data = args.validation_data[0]
models = args.models

normalizer = Normalizer.load(
    "sftp://129.16.35.202/mnt/array1/share/Datasets/"
    "gprof/simple/gprof_gmi_normalizer.pckl"
)

validation_data = GPROFValidationDataset(validation_data,
                                         normalizer=normalizer)

for m in models:
    print(f"Running evaluate for model {m}.")
    model = QRNN.load(m)
    results = validation_data.evaluate(model, 16 * 512, torch.device("cpu"))
    results.attrs["model"] = Path(m).name

    model_path = Path(m)
    output_file = model_path.parent / model_path.stem + ".nc"
    results.to_netcdf(output_file)
