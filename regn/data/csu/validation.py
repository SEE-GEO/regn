"""
====================
regn.data.validation
====================

This module provides class to read in the MRMS validation data for GPM.
"""
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import tempfile
import subprocess

import numpy as np
from pykdtree.kdtree import KDTree
from pyproj import Transformer
import xarray as xr

from regn.data.csu.l1c import L1CFile


PRECIPRATE_REGEX = re.compile("PRECIPRATE\.GC\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz")
MASK_REGEX = re.compile("MASK\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz")
RQI_REGEX = re.compile("RQI\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz")
FILE_REGEX = re.compile("[\w*\.]*\.(\d{8})\.(\d{6})\.(\d{5})\.\w*\.gz")

def local_east(xyz):
    """
    Calculate local east vectors for ECEF locations.

    Args:
        xyz: Rank-k tensor containing coordinates with the values along
             the last dimension corresponding to the x, y, z coordinates.

    Returns:
        Tensor of same shape containing the local, normalized east vector.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]

    ex = -y
    ey = x
    ez = np.zeros_like(x)

    exyz = np.stack([ex, ey, ez], axis=-1)
    exyz /= np.sqrt(np.sum(exyz ** 2, axis=-1, keepdims=True))

    return exyz

def local_north(xyz):
    """
    Calculate local north vectors for ECEF locations.

    Args:
        xyz: Rank-k tensor containing coordinates with the values along
             the last dimension corresponding to the x, y, z coordinates.

    Returns:
        Tensor of same shape containing the local, normalized north vector.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    nx = -x * z
    ny = -y * z
    nz = (x ** 2 + y ** 2)

    nxyz = np.stack([nx, ny, nz], axis=-1)
    nxyz /= np.sqrt(np.sum(nxyz ** 2, axis=-1, keepdims=True))

    return nxyz


def calculate_footprint_weights(lats,
                                lons,
                                lat_pixel,
                                lon_pixel,
                                lat_spacecraft,
                                lon_spacecraft,
                                alt_spacecraft,
                                fwhm_a,
                                fwhm_x):
    """
    Calculate Gaussian footprint weights for varying footprint widths
    along and across the scan.

    Args:
        lats: Tensor of latitude values of the locations for which to calculate
            the weights.
        lons: Tensor of longitude values of the locations for which to
            calculate the weights.
        lat_pixel: The latitude coordinate of the pixel for which to calculate
            the weights.
        lon_pixel: The longitude coordinate corresponding to lat_pixel.
        lat_spacecraft: The latitude coordinate of the spacecraft, which is
             used to calculate the along- and across-scan directions.
        fwhm_a: The full-width at half-maximum of the footprint in along-track
            direction.
        fwhm_x: The FWHM of the footprint in across-track direction.

    Returns:

        A rank-k tensor with the same shape as ``lats`` containing the weights
        for each point represented by ``lats`` and ``lons`` normalized so
        that all weights sum to 1.
    """
    xyz = np.stack(_WGS84_TO_ECEF.transform(lats, lons, np.zeros_like(lats)), axis=-1)
    xyz_c = np.stack(_WGS84_TO_ECEF.transform(lat_pixel, lon_pixel, 0), axis=-1)
    xyz_sc = np.stack(_WGS84_TO_ECEF.transform(lat_spacecraft,
                                               lon_spacecraft,
                                               alt_spacecraft),
                                               axis=-1)

    shape = [1] * len(xyz.shape)
    shape[-1] = -1
    xyz_c = xyz_c.reshape(shape)

    # Calculate LOS.
    d_xyz = xyz_c - xyz_sc
    e_c = local_east(xyz_c)
    n_c = local_north(xyz_c)
    los_n = np.sum(d_xyz * n_c, axis=-1)
    los_e = np.sum(d_xyz * e_c, axis=-1)

    # Vector along local LOS
    n_a = (e_c * los_e[..., np.newaxis] +
           n_c * los_n[..., np.newaxis])
    n_a /= np.linalg.norm(n_a, axis=-1)[..., np.newaxis]
    n_x = (n_c * los_e[..., np.newaxis] -
           e_c * los_n[..., np.newaxis])
    n_x /= np.linalg.norm(n_x, axis=-1)[..., np.newaxis]
    n_a = n_a.reshape(shape)
    n_x = n_x.reshape(shape)

    d_xyz = xyz - xyz_c
    a = -np.log(0.5) * (2.0 * np.sum(d_xyz * n_a, axis=-1) / fwhm_a) ** 2
    x = -np.log(0.5) * (2.0 * np.sum(d_xyz * n_x, axis=-1) / fwhm_x) ** 2

    ws = np.exp(-(a + x))
    ws /= ws.sum()
    return ws

def list_overpasses(base_directory):
    """
    Collects the granule numbers and corresponding dates for which ground validation
    data is available.

    Args:
        base_directory: Path to the directory that contains the validation data.

    Returns:
        Dictionary mapping granule numbers of GPM CO CONUS overpasses to dates.
    """
    granule_numbers = {}
    for path in Path(base_directory).glob("**/*.???.gz"):
        m = PRECIPRATE_REGEX.match(path.name)
        if m:
            granule_number = int(m.group(3))

            time = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            granule_numbers[granule_number] = time
    return granule_numbers

def find_files(granule_number, base_directory):
    """
    Finds files corresponding to a given granule number.

    Args:
         granule_number: The granule number given as integer.
         base_directory: Root directory containing the validation data.

    Returns:
         List of files matching the given granule number.
    """
    files = []
    for path in Path(base_directory).glob("**/*.gz"):
        m = FILE_REGEX.match(path.name)
        if m:
            n = int(m.group(3))
            if n == granule_number:
                files.append(path)
    return files

def get_date(path):
    """
    Extract date from filename.

    Args:
        Path to a validation data file.

    Returns:
        datetime object representing the time to which the data in the
        file corresponds.
    """
    m = FILE_REGEX.match(path.name)
    if m:
        granule_number = int(m.group(3))
        time = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        return time
    return None

def open_validation_dataset(granule_number, base_directory):
    """
    Open the validation data for a given granule number
    as xarray.Dataset.

    Args:
        granule_number: GPM granule number for which to open the validation
             data.
        base_directory: Path to root of the directory tree containing the
             validation data.

    Returns:
        xarray.Dataset containing the validation data.
    """
    files = find_files(granule_number, base_directory)

    # Load precip-rate data.
    precip_files = [f for f in files if PRECIPRATE_REGEX.match(f.name)]
    times = [get_date(f) for f in precip_files]

    header = np.loadtxt(files[0], usecols=(1,), max_rows=6)
    n_cols = int(header[0])
    n_rows = int(header[1])
    lon_ll = float(header[2])
    lat_ll = float(header[3])
    dl = float(header[4])

    lons = lon_ll + np.arange(n_cols) * dl
    lats = (lat_ll + np.arange(n_rows) * dl)[::-1]

    precip_files = sorted(precip_files, key=get_date)
    precip_rate = np.zeros((len(times), n_rows, n_cols))
    for i, f in enumerate(precip_files):
        precip_rate[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.float32)
    precip_rate[precip_rate < 0.0] = np.nan

    rqi_files = [f for f in files if RQI_REGEX.match(f.name)]
    rqi_files = sorted(rqi_files, key=get_date)
    rqi = np.zeros((len(times), n_rows, n_cols), dtype=np.int32)
    for i, f in enumerate(rqi_files):
        rqi[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.int32)

    mask_files = [f for f in files if MASK_REGEX.match(f.name)]
    mask_files = sorted(mask_files, key=get_date)
    mask = np.zeros((len(times), n_rows, n_cols), dtype=np.int32)
    for i, f in enumerate(mask_files):
        mask[i, :, :] = np.loadtxt(f, skiprows=6, dtype=np.int32)

    dims = ("time", "latitude", "longitude")
    data = {
        "latitude": (("latitude",), lats),
        "longitude": (("longitude",), lons),
        "time": (("time",), times),
        "precip_rate": (dims, precip_rate),
        "mask": (dims, mask),
        "radar_quality_index": (dims, rqi)
    }

    return xr.Dataset(data)


_WGS84_TO_ECEF = Transformer.from_crs("epsg:4326", "epsg:4978")
_ECEF_TO_WGS84 = Transformer.from_crs("epsg:4978", "epsg:4326")
_KD_TREE = None

def find_closest_mrms_pixel(lat, lon, mrms_data):
    """
    Finds the coordinates of the MRMS pixel that is closest to
    a given point.

    Args:
        lat: The latitude position of the point.
        lon: The longitude position of the point.
        mrms_data: Dataset containing the MRMS data.

    Return:
        Tuple ``(ri, ci)`` containing the row index ``ri`` and column index
        ``ci`` of the closes pixel in the MRMS grid or ``None`` if the closest
        point is more than 5000m away.
    """
    global _KD_TREE

    if _KD_TREE is None:
        lons = mrms_data["longitude"]
        lats = mrms_data["latitude"]
        lons, lats = np.meshgrid(lons, lats)

        xyz = np.stack(_WGS84_TO_ECEF.transform(lats,
                                                lons,
                                                np.zeros_like(lats)), axis=-1)
        xyz = xyz.reshape(-1, 3)
        _KD_TREE = KDTree(xyz)

    xyz = np.stack(_WGS84_TO_ECEF.transform(lat, lon, 0), axis=-1)
    dist, idx = _KD_TREE.query(xyz.reshape(1, -1))

    print(mrms_data["latitude"])
    print(mrms_data["longitude"])
    if dist > 5e3:
        return None

    NX = mrms_data.longitude.size
    row_index = idx[0] // NX
    column_index = idx[0] % NX
    return row_index, column_index

def _run_preprocessor(l1c_file,
                      output_file):
    """
    Run preprocessor on L1C GMI file.

    Args:
        l1c_file: Path of the L1C file for which to extract the input data
             using the preprocessor.
    """
    output_file = Path(output_file)
    if not output_file.exists():
        jobid = str(os.getpid()) + "_pp"
        prodtype = "CLIMATOLOGY"
        prepdir = "/qdata2/archive/ERA5"
        ancdir = "/qdata1/pbrown/gpm/ancillary"
        ingestdir = "/qdata1/pbrown/gpm/ppingest"
        subprocess.run(["gprof2020pp_GMI_L1C",
                        jobid,
                        prodtype, str(l1c_file),
                        prepdir,
                        ancdir,
                        ingestdir,
                        str(output_file)])


class FileProcessor:
    """
    Processor class to match GPM GMI observations with MRMS validation data and
    extract validation data.
    """
    def __init__(self,
                 validation_data_path,
                 l1c_data_path,
                 output_path):
        self.validation_data_path = validation_data_path
        self.l1c_data_path = l1c_data_path
        self.output_path = output_path
        self.granules = list_overpasses(validation_data_path)

    def match_granule(self,
                      granule_number,
                      l1c_data):

        mrms_data = open_validation_dataset(granule_number,
                                            self.validation_data_path)

        lats = l1c_data["latitude"]
        lons = l1c_data["longitude"]

        precip_rate = np.zeros(lats.shape)
        rain_fraction = np.zeros(lats.shape)
        warm_strat_fraction = np.zeros(lats.shape)
        cold_strat_fraction = np.zeros(lats.shape)
        snow_fraction = np.zeros(lats.shape)
        conv_fraction = np.zeros(lats.shape)
        hail_fraction = np.zeros(lats.shape)
        tropical_strat_fraction = np.zeros(lats.shape)
        tropical_conv_fraction = np.zeros(lats.shape)
        radar_quality_index = np.zeros(lats.shape)


        n_scans, n_pixels = precip_rate.shape

        # Size of the rectangle to use to calculate footprint average.
        DX, DY = 20, 20
        NX = mrms_data.longitude.size
        NY = mrms_data.latitude.size
        FWHM_A = 18e3
        FWHM_X = 10e3

        lats_mrms = mrms_data["latitude"]
        lons_mrms = mrms_data["longitude"]

        for i in range(n_scans):
            lat_sc = l1c_data["spacecraft_latitude"][i]
            lon_sc = l1c_data["spacecraft_longitude"][i]
            alt_sc = l1c_data["spacecraft_altitude"][i]
            time = l1c_data["scan_time"][i]
            for j in range(n_pixels):
                lon = lons[i, j]
                lat = lats[i, j]
                print(lon, lat)
                coords = find_closest_mrms_pixel(lat, lon, mrms_data)
                if coords is None:
                    precip_rate[i, j] = np.nan
                    continue
                ri, ci = coords
                if ((ri < DY) or ((ri + 1) > (NY - DY))
                   or (ci < DX) or ((ci + 1) > NX - DX)):
                    precip_rate[i, j] = np.nan
                    continue

                lats_w = lats_mrms[ri-DY:ri+DY+1]
                lons_w = lons_mrms[ci-DX:ci+DX+1]
                lons_w, lats_w = np.meshgrid(lons_w, lats_w)
                weights = calculate_footprint_weights(lats_w,
                                                      lons_w,
                                                      lat,
                                                      lon,
                                                      lat_sc,
                                                      lon_sc,
                                                      alt_sc,
                                                      FWHM_A,
                                                      FWHM_X)

                keys = ["precip_rate", "mask", "radar_quality_index"]
                mrms_sub = mrms_data[keys][{
                    "latitude": slice(ri - DY, ri + DY + 1),
                    "longitude": slice(ci - DX, ci + DX + 1),
                }]

                if mrms_sub.time.size > 2:
                    mrms_sub = mrms_sub.interp({"time": time})[0]
                else:
                    mrms_sub = mrms_sub[{"time": 0}]

                precip_rate[i, j] = np.sum(
                    weights * mrms_sub["precip_rate"].data
                )
                radar_quality_index[i, j] = np.sum(
                    weights * mrms_sub["radar_quality_index"].data
                )

                rain_mask = (mrms_sub["mask"] > 0).data.astype(np.float32)
                rain_fraction[i, j] = np.sum(weights * rain_mask)

                warm_strat_mask = mrms_sub["mask"].isin([1, 2]).data.astype(np.float32)
                warm_strat_fraction[i, j] = np.sum(weights * warm_strat_mask)

                cold_strat_mask = mrms_sub["mask"].isin([10]).data.astype(np.float32)
                cold_strat_fraction[i, j] = np.sum(weights * cold_strat_mask)

                snow_mask = mrms_sub["mask"].isin([3, 4]).data.astype(np.float32)
                snow_fraction[i, j] = np.sum(weights * snow_mask)

                conv_mask = mrms_sub["mask"].isin([6.0]).data.astype(np.float32)
                conv_fraction[i, j] = np.sum(weights * conv_mask)

                hail_mask = mrms_sub["mask"].isin([7.0]).data.astype(np.float32)
                hail_fraction[i, j] = np.sum(weights * hail_mask)

                trop_strat_mask = mrms_sub["mask"].isin([91.0]).data.astype(np.float32)
                tropical_strat_fraction[i, j] = np.sum(weights * trop_strat_mask)

                trop_conv_mask = mrms_sub["mask"].isin([96.0]).data.astype(np.float32)
                tropical_conv_fraction[i, j] = np.sum(weights * trop_conv_mask)

        match_data = {}
        match_data["latitude"] = (
            ("scans", "pixels"),
            l1c_data["latitude"].data
        )
        match_data["longitude"] = (
            ("scans", "pixels"),
            l1c_data["latitude"].data
        )
        match_data["scan_time"] = (
            ("scans",),
            l1c_data["scan_time"].data
        )
        match_data["surface_precipitation"] = (
            ("scans", "pixels"),
            precip_rate,
            {"description":
             "18x10 km footprint-averaged surface precipitation rate."
            }
        )
        match_data["radar_quality_index"] = (
            ("scans", "pixels"),
            precip_rate,
            {"description":
             "Footprint-averaged radar quality index."
            }
        )
        match_data["rain_fraction"] = (
            ("scans", "pixels"),
            rain_fraction,
            {"description":
             "Footprint-averaged fraction of MRMS pixels with a mask "
             "value greater than 0."
            }
        )
        match_data["warm_stratiform_fraction"] = (
            ("scans", "pixels"),
            warm_strat_fraction,
            {"description":
             "Footprint-averaged fraction of MRMS pixels with a mask "
             "value corresponding to warm stratiform rain (1, 2)."
            }
        )
        match_data["cool_stratiform_fraction"] = (
            ("scans", "pixels"),
            cold_strat_fraction,
            {"description":
             "Footprint-averaged fraction of MRMS pixels with a mask "
             "value corresponding to cool stratiform rain (10)."
            }
        )
        match_data["snow_fraction"] = (
            ("scans", "pixels"),
            snow_fraction,
            {"description":
             "Footprint-averaged fraction of MRMS pixels with a mask "
             "value corresponding to snow (3, 4)."
            }
        )
        match_data["convective_fraction"] = (
            ("scans", "pixels"),
            conv_fraction,
            {"description":
             "Footprint-averaged fraction of MRMS pixels with a mask "
             "value corresponding to convective rain (6.0)."
            }
        )
        match_data["hail_fraction"] = (
            ("scans", "pixels"),
            hail_fraction,
            {"description":
             "Footprint-averaged fraction of MRMS pixels with a mask "
             "value corresponding to convective hail (7.0)."
            }
        )
        match_data["tropical_stratiform_fraction"] = (
            ("scans", "pixels"),
            tropical_strat_fraction,
            {"description":
             "Footprint-averaged fraction of MRMS pixels with a mask "
             "value corresponding to tropical stratiform rain mix (91.0)."
            }
        )
        match_data["tropical_convective_fraction"] = (
            ("scans", "pixels"),
            tropical_conv_fraction,
            {"description":
             "Footprint-averaged fraction of MRMS pixels with a mask "
             "value corresponding to tropical convective rain mix (96.0)."
            }
        )
        return xr.Dataset(match_data)

    def process_granule(self,
                        granule_number):
        if granule_number not in self.granules:
            raise ValueError(
                f"No MRMS/GMI matches found for Granule number "
                "{granule_number}."
            )

        date = self.granules[granule_number]
        preprocessor_output_path = (self.output_path / "preprocessor"
                                    / f"{date.year}" / f"date.month")
        matchup_output_path = (self.output_path / "match_ups"
                               / f"{date.year}" / f"date.month")


        roi = [-130, 20, 60.0, 55]
        _, l1c_file_sub = tempfile.mkstemp()
        try:
            # Extract scans over CONUS
            l1c_file = L1CFile.open_granule(granule_number,
                                            self.l1c_data_path,
                                            self.granules[granule_number])
            l1c_file.extract_scans(roi, l1c_file_sub)

            # Run preprocessor
            preprocessor_output = (preprocessor_output_path /
                                   (Path(l1c_file.filename).stem + ".pp"))
            preproc
            run_preprocessor(l1c_file_sub, preprocessor_output)

            # Process granule
            matchup_output = (matchup_output_path /
                              Path(l1c_file.filename).stem + ".nc")
            l1c_data = L1CFile(l1c_file_sub).open()
            match_up = self.match_granule(granule_number, l1c_data)
            match_up.to_netcdf(matchup_output)
        finally:
            Path(l1c_file_sub).unlink()

    def process_month(self, year, month):
        # 1. Find granules
        granules = [g for (g, d) in self.granules.items() if
                    d.year == year and d.month == month]

        # 2. Run preprocessor, store output

        # 3. Match observations, store output















