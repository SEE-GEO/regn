"""
Tests to test the extraction of training data from the GPROF
retrieval database.
"""
from pathlib import Path

import numpy as np
from regn.data.csu.bin import (FileProcessor,
                               GPROFGMIBinFile)
from regn.data.csu.training_data import GPROFDataset
from regn.data.csu.retrieval import (ORBIT_HEADER_TYPES,
                                     PROFILE_INFO_TYPES,
                                     SCAN_HEADER_TYPES,
                                     DATA_RECORD_TYPES)
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

    input_file = GPROFGMIBinFile(path / "data" / "gpm_300_40_00_18.bin")

    dataset = GPROFDataset(output_file, normalize=False)
    normalizer = dataset.normalizer

    bts_input = input_file.handle["brightness_temps"].view("15f4")[:, :3]
    bts = dataset.x[:, :3]
    assert np.all(np.isclose(bts_input.mean(), bts.mean()))
    assert np.all(np.isclose(input_file.handle["two_meter_temperature"].mean(),
                             dataset.x[:, 15].mean()))
    assert np.all(np.isclose(input_file.handle["total_column_water_vapor"].mean(),
                             dataset.x[:, 16].mean()))

    surface_types = np.where(dataset.x[:, 17:36])[1]
    assert np.all(np.isclose(input_file.surface_type,
                             surface_types))
    airmass_types = np.where(dataset.x[:, 36:])[1]
    assert np.all(np.isclose(input_file.airmass_type,
                             airmass_types))


def test_retrieval_file_types():
    """
    Ensure that struct type defintions match the expected sizes.
    """
    assert ORBIT_HEADER_TYPES.itemsize == 400
    assert PROFILE_INFO_TYPES.itemsize == 537864
    assert SCAN_HEADER_TYPES.itemsize == 28
    assert DATA_RECORD_TYPES.itemsize == 88

