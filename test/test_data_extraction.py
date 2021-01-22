"""
Test data extraction of new GPROF GMI database.
"""
from pathlib import Path

import numpy as np
from regn.data.csu.bin import (FileProcessor,
                               GPROFGMIBinFile)
from regn.data.csu.training_data import GPROFDataset
from netCDF4 import Dataset



def test_file_processor(tmp_path):
    """
    This tests the extraction of data from a bin file and ensures that
    the extracted dataset matches the original data.
    """
    path = Path(__file__).parent
    processor = FileProcessor(path / "data")
    output_file = tmp_path / "test_file.nc"
    processor.run_async(output_file, 0.0, 1.0, 1)

    input_file = GPROFGMIBinFile(path / "data"/ "gpm_300_40_00_18.bin")

    dataset = GPROFDataset(output_file, normalize=False)
    normalizer = dataset.normalizer

    assert np.all(np.isclose(input_file.handle["brightness_temps"].view("15f4"),
                             dataset.x[:, :15]))
    assert np.all(np.isclose(input_file.handle["two_meter_temperature"],
                             dataset.x[:, 15]))
    assert np.all(np.isclose(input_file.handle["total_column_water_vapor"],
                             dataset.x[:, 16]))

    surface_types = np.where(dataset.x[:, 17:36])[1]
    assert np.all(np.isclose(input_file.surface_type,
                             surface_types))
    airmass_types = np.where(dataset.x[:, 36:])[1]
    assert np.all(np.isclose(input_file.airmass_type,
                             airmass_types))
