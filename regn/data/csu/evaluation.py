"""
========================
regn.data.csu.evaluation
========================

This module provides functionality to read in GPROF retrieval
results from a validation run on the retrieval database.
"""
from pathlib import Path

import numpy as np
import xarray as xr

_TYPES = np.dtype([
    ("two_meter_temperature", "i"),
    ("total_column_water_vapor", "i"),
    ("air_lifting_index", "i"),
    ("y_gprof", "f4"),
    ("y_true", "f4"),
    ("precip_flag", "i1"),
    ])

def open_evaluation_data(path):
    """
    Open files containing the evaluation data and combine data
    into a single xarray dataset.

    Args:
        path: Path to the folder containing the evaluation results.
            Expects filenames to adhere to the pattern 'db_run_*.bin'.

    Return:
        The data found in all files in the given folder combined into
        an xarray dataset.
    """
    files = Path(path).glob("db_run_*.bin")

    datasets = []
    dims = ("samples",)

    for file in files:
        data_raw = np.fromfile(file, dtype=_TYPES)
        data = {}

        for t in _TYPES.fields:
            data[t] = (dims, data_raw[t])

        st = int(file.name.split("_")[-1][:2])
        data["surface_type"] = (dims, st * np.ones_like(data_raw[t]))

        datasets.append(xr.Dataset(data))

    return xr.concat(datasets, "samples")




