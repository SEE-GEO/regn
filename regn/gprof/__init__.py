"""
==========
regn.gprof
==========

This module contains the interface for the operational processing of GPROF
data.
"""
from torch.utils.data import Dataset
from regn.data.csu.preprocessor import PreprocessorFile
import numpy as np
import quantnn.quantiles as qq
import xarray
import torch

N_CHANNELS = 15

class InputData(Dataset):
    """
    PyTorch dataset interface class for GPORF preprocessor files.
    """
    def __init__(self,
                 filename,
                 normalizer,
                 scans_per_batch=4):
        self.filename = filename
        self.data = PreprocessorFile(filename).to_xarray_dataset()
        self.normalizer = normalizer

        self.n_scans = self.data.scans.size
        self.n_pixels = self.data.pixels.size
        self.scans_per_batch = scans_per_batch

        self.n_batches = self.n_scans // scans_per_batch
        remainder = self.n_scans // scans_per_batch
        if remainder > 0:
            self.n_batches += 1

        bts = self.data["brightness_temperatures"]
        self.pixel_mask = np.any(np.all(bts < 0.0, axis=0), axis=1)

    def get_batch(self, i):

        i_start = i * self.scans_per_batch
        i_end = (i + 1) * self.scans_per_batch

        bts = self.data["brightness_temperatures"][i_start:i_end, :, :].data
        bts = bts.reshape(-1, N_CHANNELS)

        # 2m temperature
        t2m = self.data["two_meter_temperature"][i_start:i_end, :].data
        t2m = t2m.reshape(-1, 1)
        # Total precipitable water.
        tcwv = self.data["total_column_water_vapor"][i_start:i_end, :].data
        tcwv = tcwv.reshape(-1, 1)

        # Surface type
        n = bts.shape[0]
        st = self.data["surface_type"][i_start:i_end, :].data
        st = st.reshape(-1, 1).astype(int)
        n_types = 19
        st_1h = np.zeros((n, n_types), dtype=np.float32)
        st_1h[np.arange(n), st.ravel()] = 1.0

        # Airmass type
        am = self.data["airmass_type"][i_start:i_end, :].data
        am = np.maximum(am.reshape(-1, 1).astype(int), 0)
        n_types = 4
        am_1h = np.zeros((n, n_types), dtype=np.float32)
        am_1h[np.arange(n), am.ravel()] = 1.0

        x = np.concatenate([bts, t2m, tcwv, st_1h, am_1h], axis=1)
        return self.normalizer(x)

    def get_conv_input(self, i):

        m = 32 * int((self.n_scans / 32 + 0.5))
        dm = m - self.n_scans

        n = 32 * int((self.n_pixels / 32 + 0.5))
        dn = n - self.n_pixels


        p_m_l = dm // 2
        p_m_r = dm - p_m_l
        p_n_l = dn // 2
        p_n_r = dn - p_n_l


        bts = self.data["brightness_temperatures"].data.copy()
        bts[bts < 0.0] = np.nan
        bts[bts > 500.0] = np.nan

        mask = np.isnan(bts[:, :, 10])
        #bts[:, :, 9][mask] = np.nan

        x = torch.zeros(1, 15, self.n_scans, self.n_pixels)
        for i in range(15):
            x[0, i] = torch.tensor(bts[:, :, i])

        x = torch.nn.functional.pad(x, [p_n_l, p_n_r, p_m_l, p_m_r], "reflect")

        return x


    def run_retrieval(self, qrnn):

        mean = np.zeros((self.n_scans, self.n_pixels))
        first_tertial = np.zeros((self.n_scans, self.n_pixels))
        second_tertial = np.zeros((self.n_scans, self.n_pixels))
        pop = np.zeros((self.n_scans, self.n_pixels))

        with torch.no_grad():
            for i in range(len(self)):
                x = self[i]
                y = qrnn.predict(x)

                i_start = i * self.scans_per_batch
                i_end = (i + 1) * self.scans_per_batch

                means = qrnn.posterior_mean(y_pred=y)
                mean[i_start:i_end] = means.reshape(-1, self.n_pixels).numpy()

                t = qrnn.posterior_quantiles(y_pred=y, quantiles=[0.333])
                first_tertial[i_start:i_end] = t.reshape(-1, self.n_pixels).numpy()
                t = qrnn.posterior_quantiles(y_pred=y, quantiles=[0.666])
                second_tertial[i_start:i_end] = t.reshape(-1, self.n_pixels).numpy()

                p = qrnn.probability_larger_than(y_pred=y, y=0.01)
                pop[i_start:i_end] = p.reshape(-1, self.n_pixels).numpy()


        dims = ["scans", "pixels"]

        data = {
            "precip_mean": (dims[:2], mean),
            "precip_1st_tertial": (dims[:2], first_tertial),
            "precip_3rd_tertial": (dims[:2], second_tertial),
            "precip_pop": (dims[:2], pop)
        }
        return xarray.Dataset(data)

    def write_retrieval_results(self, path, results):
        preprocessor_file = PreprocessorFile(self.filename)
        return preprocessor_file.write_retrieval_results(path, results)


    def __len__(self):
        return self.n_batches

    def __getitem__(self, i):
        if i > self.n_batches:
            raise IndexError()
        return self.get_batch(i)
