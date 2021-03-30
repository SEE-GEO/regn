"""
=================
regn.data.csu.l1c
=================

Functionality to read L1C files.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import h5py
import xarray as xr

class L1CFile:
    """


    """

    @staticmethod
    def open_granule(granule, path, date=None):
        if date is not None:
            year = date.year - 2000
            month = date.month
            day = date.day
            path = Path(path) / f"{year:02}{month:02}" / f"{year:02}{month:02}{day:02}"
            files = path.glob(f"1C-R.GPM.GMI.*{granule:06}.V05A.HDF5")
        else:
            path = Path(path)
            files = path.glob(f"**/1C-R.GPM.GMI.*{granule:06}.V05A.HDF5")

        try:
            f = next(iter(files))
            return L1CFile(f)
        except StopIteration:
            raise Exception(
                f"Could not find a L1C file with granule number {granule}."
            )


    def __init__(self, path):
        self.path = Path(path)

    def __repr__(self):
        return f"L1CFile(filename='{self.path.name}')"

    def extract_scans(self, roi, output_filename):
        """
        Extract scans over a rectangular region of interest (ROI).

        Args:
            roi: The region of interest given as an length-4 iterable
                 containing the lower-left corner longitude and latitude
                 coordinates followed by the upper-right corner longitude
                 and latitude coordinates.
            output_filename: Name of the file to which to write the extracted
                 scans.
        """
        lon_min, lat_min, lon_max, lat_max = roi

        with h5py.File(self.path, "r") as input:
            lats = input["S1/Latitude"][:]
            lons = input["S1/Longitude"][:]

            indices = np.where(np.any(
                (lats > lat_min) * (lats < lat_max) *
                (lons > lon_min) * (lons < lon_max),
                axis=-1
            ))[0]

            with h5py.File(output_filename, "w") as output:

                g = output.create_group("S1")
                n_scans = indices.size
                for name, item in input["S1"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g.create_dataset(name,
                                         shape=(n_scans, ) + shape[1:],
                                         data=item[indices])

                g_st = g.create_group("ScanTime")
                for name, item in input["S1/ScanTime"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g_st.create_dataset(name,
                                            shape=(n_scans, ) + shape[1:],
                                            data=item[indices])

                g_sc = g.create_group("SCstatus")
                for name, item in input["S1/SCstatus"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g_sc.create_dataset(name,
                                            shape=(n_scans, ) + shape[1:],
                                            data=item[indices])

                g = output.create_group("S2")
                for name, item in input["S2"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g.create_dataset(name,
                                         shape=(n_scans, ) + shape[1:],
                                         data=item[indices])

                g_st = g.create_group("ScanTime")
                for name, item in input["S2/ScanTime"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g_st.create_dataset(name,
                                            shape=(n_scans, ) + shape[1:],
                                            data=item[indices])

                g_sc = g.create_group("SCstatus")
                for name, item in input["S2/SCstatus"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g_sc.create_dataset(name,
                                            shape=(n_scans, ) + shape[1:],
                                            data=item[indices])

                for a in input.attrs:
                    output.attrs[a] = input.attrs[a]

    def extract_scans_and_pixels(self,
                                 scans,
                                 pixels,
                                 output_filename):
        """
        Extract scans over a rectangular region of interest (ROI).

        Args:
            roi: The region of interest given as an length-4 iterable
                 containing the lower-left corner longitude and latitude
                 coordinates followed by the upper-right corner longitude
                 and latitude coordinates.
            output_filename: Name of the file to which to write the extracted
                 scans.
        """

        with h5py.File(self.path, "r") as input:
            lats = input["S1/Latitude"][scans, 0]
            lons = input["S1/Longitude"][scans, 0]

            with h5py.File(output_filename, "w") as output:

                g = output.create_group("S1")
                n_scans = len(scans)
                n_pixels = len(pixels)
                for name, item in input["S1"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g.create_dataset(name,
                                         shape=(n_scans, n_pixels) + shape[2:],
                                         data=item[scans, 0])

                g_st = g.create_group("ScanTime")
                for name, item in input["S1/ScanTime"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        if len(shape) > 1:
                            g_st.create_dataset(name,
                                                shape=(n_scans, n_pixels) + shape[2:],
                                                data=item[scans, 0])
                        else:
                            g_st.create_dataset(name,
                                                shape=(n_scans,),
                                                data=item[scans])

                g_sc = g.create_group("SCstatus")
                for name, item in input["S1/SCstatus"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        if len(shape) > 1:
                            g_sc.create_dataset(name,
                                                shape=(n_scans, n_pixels) + shape[2:],
                                                data=item[scans, 0])
                        else:
                            g_sc.create_dataset(name,
                                                shape=(n_scans,),
                                                data=item[scans])

                g = output.create_group("S2")
                for name, item in input["S2"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        g.create_dataset(name,
                                         shape=(n_scans, n_pixels) + shape[2:],
                                         data=item[scans, 0])

                g_st = g.create_group("ScanTime")
                for name, item in input["S2/ScanTime"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        if len(shape) > 1:
                            g_st.create_dataset(name,
                                                shape=(n_scans, n_pixels) + shape[2:],
                                                data=item[scans, 0])
                        else:
                            g_st.create_dataset(name,
                                                shape=(n_scans,),
                                                data=item[scans])

                g_sc = g.create_group("SCstatus")
                for name, item in input["S2/SCstatus"].items():
                    if isinstance(item, h5py.Dataset):
                        shape = item.shape
                        if len(shape) > 1:
                            g_sc.create_dataset(name,
                                                shape=(n_scans, n_pixels) + shape[2:],
                                                data=item[scans, 0])
                        else:
                            g_sc.create_dataset(name,
                                                shape=(n_scans,),
                                                data=item[scans])

                for a in input.attrs:
                    output.attrs[a] = input.attrs[a]

    def open(self):
        with h5py.File(self.path, "r") as input:

            lats = input["S1/Latitude"][:]
            lons = input["S1/Longitude"][:]
            lats_sc = input["S1/SCstatus/SClatitude"][:]
            lons_sc = input["S1/SCstatus/SClongitude"][:]
            alt_sc = input["S1/SCstatus/SClongitude"][:]
            tbs = np.concatenate([input["S1/Tc"][:], input["S2/Tc"][:]], axis=-1)

            n_scans = lats.shape[0]
            times = np.zeros(n_scans, dtype='datetime64[ms]')
            g_t = input["S1/ScanTime"]
            for i in range(n_scans):
                times[i] = datetime(g_t["Year"][i],
                                    g_t["Month"][i],
                                    g_t["DayOfMonth"][i],
                                    g_t["Hour"][i],
                                    g_t["Minute"][i],
                                    g_t["Second"][i],
                                    g_t["MilliSecond"][i] * 1000)

            dims = ("scans", "pixels")
            data = {
                "latitude": (dims, lats),
                "longitude": (dims, lons),
                "spacecraft_latitude": (dims[:1], lats_sc),
                "spacecraft_longitude": (dims[:1], lons_sc),
                "spacecraft_altitude": (dims[:1], alt_sc),
                "brightness_temperatures": (dims + ("channels",), tbs),
                "scan_time": (dims[:1], times)
            }

        return xr.Dataset(data)








l1c = L1CFile.open_granule(16566, "/home/simonpf/src/regn/data/validation")

file = h5py.File(l1c.path, "r")
roi = [-130, 20, 60.0, 55]
l1c.extract_scans(roi, "test.hdf5")
l1c.open()