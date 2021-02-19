"""
===========================
regn.data.csu.training_data
===========================

This module provides an interface to the GPROF training data.
"""
import logging
LOGGER = logging.getLogger(__name__)

from netCDF4 import Dataset
import numpy as np
import torch
from quantnn.normalizer import MinMaxNormalizer, Normalizer
from quantnn.drnn import _to_categorical
import quantnn.quantiles as qq
import quantnn.density as qd
import xarray

from regn.data.augmentation import extract_subscene, mask_stripe

class GPROFDataset:
    """
    Dataset interface to load GPROF data from NetCDF file.

    Attributes:
        x: Rank-2 tensor containing the input data with
           samples along first dimension.
        y: The target values
        filename: The filename from which the data is loaded.
        target: The name of the variable used as target variable.
        batch_size: The size of data batches returned by __getitem__ method.
        normalizer: The normalizer used to normalize the data.
        shuffle: Whether or not the ordering of the data is shuffled.
    """
    def __init__(self,
                 filename,
                 target="surface_precip",
                 normalize=True,
                 transform_zero_rain=True,
                 batch_size=None,
                 normalizer=None,
                 shuffle=True,
                 bins=None):
        """
        Create GPROF dataset.

        Args:
            filename: Path to the NetCDF file containing the data to
                 load.
            target: The variable to use as target (output) variable.
            normalize: Whether or not to normalize the input data.
            transform_zero_rain: Whether or not to replace very small
                 and zero rain with random amounts in the range [1e-6, 1e-4]
        """
        self.filename = filename
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._load_data()

        indices_1h = list(range(17, 40))
        if normalizer is None:
            self.normalizer = Normalizer(self.x,
                                         exclude_indices=indices_1h)
        elif isinstance(normalizer, type):
            self.normalizer = normalizer(self.x,
                                         exclude_indices=indices_1h)
        else:
            self.normalizer = normalizer

        if normalize:
            self.x = self.normalizer(self.x)

        if transform_zero_rain:
            self._transform_zero_rain()

        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

        if bins is not None:
            self.binned = True
            self.y = _to_categorical(self.y, bins)
        else:
            self.binned = False

        self._shuffled = False
        if self.shuffle:
            self._shuffle()


    def _transform_zero_rain(self):
        """
        Transforms rain amounts smaller than 1e-4 to a random amount withing
        the range [1e-6, 1e-4], which helps to stabilize training of QRNNs.
        """
        indices = self.y < 1e-4
        self.y[indices] = np.random.uniform(1e-6, 1.0e-4, indices.sum())

    def _load_data(self):
        """
        Loads the data from the file into the classes ``x`` attribute.
        """
        with Dataset(self.filename, "r") as dataset:

            LOGGER.info("Loading data from file: %s",
                        self.filename)

            variables = dataset.variables
            n = dataset.dimensions["samples"].size

            #
            # Input data
            #

            # Brightness temperatures
            m = dataset.dimensions["channel"].size
            bts = np.zeros((n, m))
            index_start = 0
            chunk_size = 8192
            v = dataset["brightness_temps"]
            while index_start < n:
                index_end = index_start + chunk_size
                bts[index_start: index_end, :] = v[index_start: index_end, :].data
                index_start += chunk_size

            invalid = (bts > 500.0) + (bts < 0.0)
            bts[invalid] = -1.0

            LOGGER.info("Loaded %n brightness temperatures.", n)

            # 2m temperature
            t2m = variables["two_meter_temperature"][:].reshape(-1, 1)
            # Total precitable water.
            tcwv = variables["total_column_water_vapor"][:].reshape(-1, 1)
            # Surface type
            st = variables["surface_type"][:]
            n_types = 19
            st_1h = np.zeros((n, n_types), dtype=np.float32)
            st_1h[np.arange(n), st.ravel()] = 1.0
            # Airmass type
            am = variables["airmass_type"][:]
            n_types = 4
            am_1h = np.zeros((n, n_types), dtype=np.float32)
            am_1h[np.arange(n), am.ravel()] = 1.0

            self.x = np.concatenate([bts, t2m, tcwv, st_1h, am_1h], axis=1)

            #
            # Output data
            #

            self.y = variables[self.target][:]

    def _shuffle(self):
        if not self._shuffled:
            indices = np.random.permutation(self.x.shape[0])
            self.x = self.x[indices, :]
            self.y = self.y[indices]
            self._shuffled = True


    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """

        self._shuffled = False
        if self.batch_size is None:
            return (torch.tensor(self.x[[i], :]),
                    torch.tensor(self.y[[i]]))

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)

        if i + 1 == len(self):
            print("shufflin' ...")
            self._shuffle()
            if not self.binned:
                self._transform_zero_rain()

        if i >= len(self):
            raise IndexError()
        return (torch.tensor(self.x[i_start:i_end, :]),
                torch.tensor(self.y[i_start:i_end]))

    def __len__(self):
        """
        The number of samples in the dataset.
        """
        if self.batch_size:
            n = self.x.shape[0] // self.batch_size
            return n
        else:
            return self.x.shape[0]

    def evaluate(self,
                 qrnn,
                 batch_size=16384):
        """
        Run retrieval on dataset.
        """
        n_samples = self.x.shape[0]
        quantiles = qrnn.quantiles
        y_mean = np.zeros(n_samples)
        y_median = np.zeros(n_samples)
        dy_mean = np.zeros(n_samples)
        dy_median = np.zeros(n_samples)
        pop = np.zeros(n_samples)
        y_true = np.zeros(n_samples)
        calibration = np.zeros(len(qrnn.quantiles))

        i_start = 0
        quantiles = torch.tensor(qrnn.quantiles).float()
        while (i_start < n_samples):

            i_end = i_start + batch_size
            x = torch.tensor(self.x[i_start:i_end]).float().detach()
            y = torch.tensor(self.y[i_start:i_end]).float().detach()

            y_pred = qrnn.model(x)
            y_m = qq.posterior_mean(y_pred, quantiles).reshape(-1)
            y_mean[i_start:i_end] = y_m.detach().numpy()
            dy_mean[i_start:i_end] = (y_m - y).detach().numpy()

            y_pred = qrnn.model(x)
            y_m = qq.posterior_quantiles(y_pred, quantiles, [0.5]).reshape(-1)
            y_median[i_start:i_end] = y_m.detach().numpy().ravel()
            dy_median[i_start:i_end] = (y_m - y).detach().numpy().ravel()

            pop[i_start:i_end] = qq.probability_larger_than(
                y_pred, quantiles, 1e-2).detach().numpy()

            y_true[i_start:i_end] = y.numpy()

            calibration += (y.reshape(-1, 1) <= y_pred).sum(axis=0).detach().numpy()

            i_start = i_end

        calibration /= n_samples

        results = {"y_mean": y_mean,
                   "dy_mean": dy_mean,
                   "y_median": y_median,
                   "dy_median": dy_median,
                   "pop": pop,
                   "y_true": y_true,
                   "calibration": calibration}
        return results


def evaluate(data,
             model,
             device=torch.device("cuda")):

    if not torch.cuda.is_available():
        device = torch.device("cpu")

    cpu = torch.device("cpu")

    quantiles = torch.tensor(model.quantiles).float().to(device)
    n_quantiles = len(quantiles)

    means = []
    dy_means = []
    medians = []
    dy_medians = []
    ys = []
    surfaces = []
    airmasses = []
    calibration = torch.zeros_like(quantiles)
    n_samples = 0

    model.model.eval()
    model.model.to(device)

    st_indices = torch.arange(19).reshape(1, -1).to(device)
    am_indices = torch.arange(4).reshape(1, -1).to(device)

    with torch.no_grad():
        for x, y in data:


            x = x.float().to(device)
            y = torch.unsqueeze(y.float().to(device), 1)

            y_pred = model.model(x)


            mean = qq.posterior_mean(y_pred, quantiles, quantile_axis=1)
            dy_mean = mean - y
            median = qq.posterior_quantiles(y_pred,
                                            quantiles,
                                            [0.5], quantile_axis=1)
            dy_median = median - y

            means += [mean.to(cpu).squeeze(1)]
            dy_means += [dy_mean.to(cpu).squeeze(1)]
            dy_medians += [dy_mean.to(cpu).squeeze(1)]
            ys += [y.to(cpu).squeeze(1)]

            print(ys[-1].shape)


            #shape[1] = -1
            calibration += (y < y_pred).sum((0, 2, 3))

            n_samples += x.shape[0]

            if x.shape[1] > 15:
                surfaces += [(x[:, 17:36] * st_indices).sum(1).cpu()]
                airmasses += [(x[:, 36:] * am_indices).sum(1).cpu()]

        means = torch.cat(means, 0)
        dy_means = torch.cat(dy_means, 0)
        medians = torch.cat(dy_medians, 0)
        dy_medians = torch.cat(dy_medians, 0)
        ys = torch.cat(ys, 0)
        if x.shape[1] > 15:
            surfaces = torch.cat(surfaces, 0)
            airmasses = torch.cat(airmasses, 0)


    dims = ["samples", "scans", "pixels"]

    data = {
        "y_mean": (dims, means.numpy()),
        "y_median": (dims, medians.numpy()),
        "dy_mean": (dims, dy_means.numpy()),
        "dy_median": (dims, dy_medians.numpy()),
        "y": (dims, ys.numpy()),
        "quantiles": (("quantiles",), quantiles.cpu().numpy()),
        "calibration": (("quantiles",), calibration.cpu().numpy() / n_samples),
        }
    if x.shape[1] > 15:
        data["surface_type"] = (("samples",), surfaces.numpy())
        data["airmass_type"] = (("samples",), surfaces.numpy())


    del means
    del dy_mean
    del medians
    del dy_median
    del ys
    del y_pred

    return xarray.Dataset(data)

def evaluate_drnn(data,
                  model,
                  device=torch.device("cuda")):

    if not torch.cuda.is_available():
        device = torch.device("cpu")

    cpu = torch.device("cpu")

    bins = torch.tensor(model.bins).float().to(device)

    means = []
    dy_means = []
    medians = []
    dy_medians = []
    ys = []
    n_samples = 0
    tercile_1 = []
    tercile_2 = []
    surfaces = []
    airmasses = []

    model.model.eval()
    model.model.to(device)

    st_indices = torch.arange(19).reshape(1, -1).to(device)
    am_indices = torch.arange(4).reshape(1, -1).to(device)

    with torch.no_grad():
        for x, y in data:

            x = x.float().to(device)
            y = y.float().to(device).reshape(-1)


            y_pred = torch.softmax(model.model(x), 1)
            y_pred = qd.normalize(y_pred, bins)


            mean = qd.posterior_mean(y_pred, bins, bin_axis=1).reshape(-1)
            dy_mean = mean - y
            median = qd.posterior_quantiles(y_pred,
                                            bins,
                                            [0.5], bin_axis=1).reshape(-1)
            t1 = qd.posterior_quantiles(y_pred,
                                        bins,
                                        [0.01], bin_axis=1).reshape(-1)
            t2 = qd.posterior_quantiles(y_pred,
                                        bins,
                                        [0.99], bin_axis=1).reshape(-1)
            dy_median = mean - y

            means += [mean.to(cpu)]
            medians += [median.to(cpu)]
            dy_means += [dy_mean.to(cpu)]
            dy_medians += [dy_median.to(cpu)]
            ys += [y.to(cpu)]
            tercile_1 += [t1.to(cpu)]
            tercile_2 += [t2.to(cpu)]

            n_samples += x.shape[0]

            surfaces += [(x[:, 17:36] * st_indices).sum(1).cpu()]
            airmasses += [(x[:, 36:] * am_indices).sum(1).cpu()]

        means = torch.cat(means, 0)
        dy_means = torch.cat(dy_means, 0)
        medians = torch.cat(medians, 0)
        dy_medians = torch.cat(dy_medians, 0)
        ys = torch.cat(ys, 0)
        surfaces = torch.cat(surfaces, 0)
        airmasses = torch.cat(airmasses, 0)
        tercile_1 = torch.cat(tercile_1, 0)
        tercile_2 = torch.cat(tercile_2, 0)


    dims = ["samples"]

    data = {
        "y_mean": (("samples",), means.numpy()),
        "y_median": (("samples",), medians.numpy()),
        "dy_mean": (("samples",), dy_means.numpy()),
        "dy_median": (("samples",), dy_medians.numpy()),
        "y": (("samples",), ys.numpy()),
        "surface_type": (("samples",), surfaces.numpy()),
        "airmass_type": (("samples",), airmasses.numpy()),
        "1st_tercile": (("samples",), tercile_1.numpy()),
        "2nd_tercile": (("samples",), tercile_2.numpy())
        }

    del means
    del dy_mean
    del medians
    del dy_median
    del ys
    del y_pred

    return xarray.Dataset(data)

import math

def _pixel_offset(pixel_index):
    a = 30 / 110 ** 2
    b = 110
    offset = a * (pixel_index - b) ** 2
    return offset


def _extract_subscene(input_data, p_in, p_out):
    """
    Data augmentation function that extracts 164 x 164 patches from larger
    patch and transforms the data to simulate retrievals at different regions
    of the swath.
    """
    p_in = np.clip(p_in, -1.0, 1.0)
    p_out = np.clip(p_out, -1.0, 1.0)
    c_in = int(110  + 0.5 * (157 - 65) * p_in)
    c_out = int(110  + 0.5 * (157 - 65) * p_out)

    offsets_in = _pixel_offset(np.arange(c_in - 64, c_in + 64))
    offsets_out = _pixel_offset(np.arange(c_out - 64, c_out + 64))

    out = np.zeros_like(input_data, shape=input_data.shape[:-2] + (128, 128))
    scan_start = 57
    for i in range(128):
        o_in = offsets_in[i]
        o_out = offsets_out[i]
        d_o = o_out - o_in

        i_start = math.floor(scan_start + d_o)
        c = scan_start + d_o - i_start

        out[..., :, i] = (1.0 - c) * input_data[..., i_start:i_start + 128, c_in - 64 + i]
        out[..., :, i] += c * input_data[..., i_start + 1:i_start + 129, c_in - 64 + i]
    return out


class GPROFConvDataset:
    """
    Dataset interface to load GPROF  data from NetCDF file.

    Attributes:
        x: Rank-4 tensor containing the input data with
           samples along first dimension.
        y: Rank-3 tensor containing the target values.
        filename: The filename from which the data is loaded.
        target: The name of the variable used as target variable.
        batch_size: The size of data batches returned by __getitem__ method.
        normalizer: The normalizer used to normalize the data.
        shuffle: Whether or not the ordering of the data is shuffled.
    """
    def __init__(self,
                 filename,
                 target="surface_precip",
                 normalize=True,
                 transform_zero_rain=True,
                 batch_size=None,
                 normalizer=None,
                 shuffle=True,
                 bins=None,
                 augment=True):
        """
        Create GPROF dataset.

        Args:
            filename: Path to the NetCDF file containing the data to
                 load.
            target: The variable to use as target (output) variable.
            normalize: Whether or not to normalize the input data.
            transform_zero_rain: Whether or not to replace very small
                 and zero rain with random amounts in the range [1e-6, 1e-4]
        """
        self.filename = filename
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self._load_data()

        indices_1h = list(range(17, 40))
        if normalizer is None:
            self.normalizer = MinMaxNormalizer(self.x)
        elif isinstance(normalizer, type):
            self.normalizer = normalizer(self.x)
        else:
            self.normalizer = normalizer

        if normalize:
            self.x = self.normalizer(self.x)

        if transform_zero_rain:
            self._transform_zero_rain()

        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

        if bins is not None:
            self.binned = True
            self.y = _to_categorical(self.y, bins)
        else:
            self.binned = False

        self._shuffled = False
        if self.shuffle:
            self._shuffle()


    def _transform_zero_rain(self):
        """
        Transforms rain amounts smaller than 1e-4 to a random amount withing
        the range [1e-6, 1e-4], which helps to stabilize training of QRNNs.
        """
        indices = (self.y < 1e-4) * (self.y >= 0.0)
        self.y[indices] = np.random.uniform(1e-6, 1.0e-4, indices.sum())

    def _load_data(self):
        """
        Loads the data from the file into the classes ``x`` attribute.
        """
        with Dataset(self.filename, "r") as dataset:

            LOGGER.info("Loading data from file: %s",
                        self.filename)

            variables = dataset.variables
            n = dataset.dimensions["samples"].size
            h = dataset.dimensions["scans"].size
            w = dataset.dimensions["pixels"].size
            c = dataset.dimensions["channels"].size

            #
            # Input data
            #

            # Brightness temperatures
            bt = np.zeros((n, h, w, c), np.float32)
            sp = np.zeros((n, h, w), np.float32)

            index_start = 0
            chunk_size = 128
            v_bt = dataset["brightness_temperatures"]
            v_sp = dataset["surface_precip"]
            while index_start < n:
                index_end = index_start + chunk_size
                bts = v_bt[index_start: index_end].data
                bt[index_start: index_end] = bts
                sp[index_start: index_end] = v_sp[index_start: index_end].data
                index_start += chunk_size

            bt[bt < 0.0] = np.nan
            bt[bt > 500.0] = np.nan

            valid = np.where(~np.all(np.isnan(sp), axis=(1, 2)))[0]

            bt = bt[valid]
            sp = sp[valid]

            self.x = np.zeros_like(sp, shape=(n, c, 128, 128))
            self.y = np.zeros_like(sp, shape=(n, 128, 128))

            for i in range(self.x.shape[0]):

                if not self.augment:
                    self.x[i] = np.transpose(extract_subscene(bt[i], 0.0, 0.0), [2, 0, 1])
                    self.y[i] = extract_subscene(sp[i], 0.0, 0.0)
                    continue

                p_in = np.random.uniform(-1, 1)
                p_out = np.random.uniform(-1, 1)

                self.x[i] = np.transpose(extract_subscene(bt[i], p_in, p_out), [2, 0, 1])
                self.y[i] = extract_subscene(sp[i], p_in, p_out)


                r = np.random.rand()
                if r < 0.2:
                    mask_stripe(self.x[i], p_out)

                r = np.random.rand()
                if (r > 0.5):
                    self.x[i] = np.flip(self.x[i], axis=2)
                    self.y[i] = np.flip(self.y[i], axis=1)
                if (r > 0.5):
                    self.x[i] = np.flip(self.x[i], axis=1)
                    self.y[i] = np.flip(self.y[i], axis=0)

                self.y[np.isnan(self.y)] = -2.0


    def _shuffle(self):
        if not self._shuffled:
            indices = np.random.permutation(self.x.shape[0])
            self.x = self.x[indices]
            self.y = self.y[indices]
            self._shuffled = True


    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """

        self._shuffled = False
        if self.batch_size is None:
            return (torch.tensor(self.x[i]),
                    torch.tensor(self.y[i]))

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)

        if i + 1 == len(self):
            print("shufflin' ...")
            self._shuffle()
            if not self.binned:
                self._transform_zero_rain()

        if i >= len(self):
            raise IndexError()
        return (torch.tensor(self.x[i_start:i_end]),
                torch.tensor(self.y[i_start:i_end]))

    def __len__(self):
        """
        The number of samples in the dataset.
        """
        if self.batch_size:
            n = self.x.shape[0] // self.batch_size
            return n
        else:
            return self.x.shape[0]

