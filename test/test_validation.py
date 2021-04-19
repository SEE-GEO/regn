from pathlib import Path
import tempfile

import numpy as np

from regn.data.csu.l1c import L1CFile
from regn.data.csu.validation import FileProcessor

def test_footprint_mathing(tmp_path):
    """
    Ensure that matched footprint rain rates have the expected
    FWHM.

    For this there are two test granules, which each have two
    pixels that are located such that the only non-raining MRMS
    pixel is located directly at the center for the first pixel
    and at half FWHM distance from the center for the second
    pixel. There's one granule for each direction.

        Granule 16559: Pixels are at 9 km distance along
            the satellite orbit.
        Granule 16560: Pixels are at 5 km distance along
            the satellite orbit.
    """

    path = Path(__file__).parent
    input_path = path / "data"

    processor = FileProcessor(input_path,
                              input_path,
                              tmp_path)

    roi = [-20, -20, 20, 20]

    granule_number = 16559
    _, l1c_file_sub = tempfile.mkstemp()
    try:
        l1c_file = L1CFile.open_granule(granule_number,
                                        processor.l1c_data_path,
                                        processor.granules[granule_number])
        l1c_file.extract_scans(roi, l1c_file_sub)
        l1c_data = L1CFile(l1c_file_sub).open()
        matched = processor.match_granule(granule_number, l1c_data)
        precip_rate = matched["surface_precipitation"]
        assert np.isclose(precip_rate[1] / precip_rate[0], 2.0)

    finally:
        Path(l1c_file_sub).unlink()

    processor = FileProcessor(input_path,
                              input_path,
                              tmp_path)
    granule_number = 16560
    _, l1c_file_sub = tempfile.mkstemp()
    try:
        l1c_file = L1CFile.open_granule(granule_number,
                                        processor.l1c_data_path,
                                        processor.granules[granule_number])
        l1c_file.extract_scans(roi, l1c_file_sub)
        l1c_data = L1CFile(l1c_file_sub).open()
        matched = processor.match_granule(granule_number, l1c_data)
        precip_rate = matched["surface_precipitation"]
        print(precip_rate[1] / precip_rate[0])
        assert np.isclose(precip_rate[1] / precip_rate[0], 2.0)

    finally:
        Path(l1c_file_sub).unlink()

def test_process_month(tmp_path):
    """
    Ensure that parallel processing of a month works.
    """

    path = Path(__file__).parent
    input_path = path / "data"

    processor = FileProcessor(input_path,
                              input_path,
                              tmp_path)

    processor.process_month(2017, 1)

    output_path = tmp_path / "match_ups" / "2017" / "01"
    assert output_path.exists()
    assert len(list(output_path.glob("*.nc"))) == 2
