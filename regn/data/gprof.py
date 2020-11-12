"""
regn.data.grof
==============

"""
from abc import ABC, abstractmethod

import matplotlib as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import netCDF4
import torch
from torch.utils.data import Dataset
import xarray
import pandas as pd


###############################################################################
# Input-data normalizer.
###############################################################################
class Normalizer:
    """
    The normalizer object keeps track of means and standard deviations to
    use to normalize a dataset.
    """
    @staticmethod
    def load(filename):
        """
        Load normalizer from file.

        Args:
            filename: Path of netCDF4 file containing the normalization data.

        Returns:
            Normalizer object initialized with the normalization data stored
            in the given file.
        """
        data = xarray.open_dataset(filename)
        x_mean = data["x_mean"]
        x_sigma = data["x_sigma"]
        return Normalizer(x_mean, x_sigma)

    def __init__(self, x_mean, x_sigma):
        """
        Create normalizer with given mean and standard deviation vectors.

        Args:
            x_mean(``np.array``): Array containing the mean values for all
                input features.
            x_sigma(``np.array``): Array containing the standard deviation
                values for all input features.
        """
        self.x_mean = x_mean.reshape(1, -1)
        self.x_sigma = x_sigma.reshape(1, -1)

    def __call__(self, input_data):
        """
        Linearly transforms the inputs in x to have zero mean and unit
        standard deviation.

        Args:
            input_data: Array of shape ``(m, n)`` with samples along first
            dimension and input features along second dimension.

        Return:
            The given inputs in ``x`` normalized by the mean and standard
            deviation stored in the normalizer.
        """
        x_normed = (input_data.astype(np.float64) - self.x_mean) / self.x_sigma
        return x_normed.astype(np.float32)

    def save(self, filename):
        """ Store normalization data to file. """
        file = netCDF4.Dataset(filename, "w")
        file.createVariable("x_mean", "f4", dimensions=())
        file["x_mean"] = self.x_mean
        file.createVariable("x_sigma", "f4", dimensions=())
        file["x_sigma"] = self.x_sigma
        file.close()


###############################################################################
# Base class for GPROF datasets.
###############################################################################
class GPROFDataset(ABC):
    """
    Base class for datasets base on GPROF databses.
    """
    def __init__(self,
                 path,
                 batch_size=None,
                 normalizer=Normalizer,
                 log_rain_rates=False,
                 rain_threshold=None,
                 shuffle=True):
        self.x = None
        self.y = None
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._load_data(path)

        if not normalizer:
            self.normalizer = self._get_normalizer()
        elif isinstance(normalizer, Normalizer):
            self.normalizer = normalizer
        else:
            self.normalizer = Normalizer.load(normalizer)

        self.x = self.normalizer(self.x)

        if log_rain_rates:
            self.transform_log()

        if rain_threshold:
            self.y = (self.y > rain_threshold).astype(np.float32)

    @abstractmethod
    def _get_normalizer(self):
        """
        Return normalizer object for data of this dataset instance.
        """

    @abstractmethod
    def _load_data(self, path):
        """
        Load data from file into x and y attributes of the class
        instance.

        Args:
            path(``str`` or ``pathlib.Path``): Path to netCDF file containing
                the data to load.
        """

    def __len__(self):
        """
        The number of entries in the training data. This is part of the
        pytorch interface for datasets.

        Return:
            int: The number of samples in the data set
        """
        if self.batch_size is None:
            return self.x.shape[0]
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if self.shuffle and i == 0:
            indices = np.random.permutation(self.x.shape[0])
            self.x = self.x[indices, :]
            self.y = self.y[indices]

        if self.batch_size is None:
            return (torch.tensor(self.x[[i], :]),
                    torch.tensor(self.y[[i]]))

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)
        if i >= len(self):
            raise IndexError()
        return (torch.tensor(self.x[i_start:i_end, :]),
                torch.tensor(self.y[i_start:i_end]))

    def transform_log(self):
        """
        Transforms output rain rates to log space. Samples with
        zero rain are replaced by uniformly sampling values from the
        range [0, rr.min()].
        """
        y = np.copy(self.y)
        y_min = y[y > 0.0].min()
        inds = y == 0.0
        y[inds] = np.random.uniform(0.0, y_min, inds.sum())
        y = np.log10(y)
        self.y = y

    def store_normalization_data(self, filename):
        """
        Store means and standard deviations used to normalize input data in
        a NetCDF4 file.

        Args:
            filename: Path of  the file to which to write the normalization
                data.
        """
        self.normalization_data.to_netcdf(filename)


###############################################################################
# GMI data
###############################################################################
class GMIDataset(GPROFDataset):
    """
    Pytorch dataset interface for the Gprof training data for the GMI sensor.

    This class is a wrapper around the netCDF4 files that are used to store
    the GProf training data. It provides as input vector the brightness
    temperatures and as output vector the surface precipitation.

    """
    def __init__(self,
                 path,
                 batch_size=None,
                 surface_type=-1,
                 normalizer=None,
                 log_rain_rates=False,
                 rain_threshold=None,
                 normalize=True):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size
            surface_type: If positive, only samples of the given surface type
                are provided. Otherwise the surface type index is provided as
                input features.
            normalization_data: Filename of normalization data to use to
                normalize inputs.
            log_rain_rates: Boolean indicating whether or not to transform
                output to log space.
            rain_threshold: If provided, the given value is applied to the
                output rain rates to turn them into binary non-raining /
                raining labels.
        """
        self.normalize = normalize
        self.surface_type = surface_type
        super().__init__(path,
                         batch_size,
                         normalizer,
                         log_rain_rates,
                         rain_threshold)

    def _get_normalizer(self):
        if self.normalize:
            x = self.x.astype(np.float64)
            x_mean = np.mean(x, axis=0, keepdims=True)
            x_sigma = np.std(x, axis=0, keepdims=True)
            if self.surface_type < 0:
                x_mean[0, 15:] = 0.5
                x_sigma[0, 15:] = 1.0
            return Normalizer(x_mean, x_sigma)
        return Normalizer(0.0, 1.0)

    def _load_data(self, path):
        self.file = netCDF4.Dataset(path, mode="r")

        tcwv = self.file.variables["tcwv"][:]
        surface_type_data = self.file.variables["surface_type"][:]
        t2m = self.file.variables["t2m"][:]
        bt = self.file.variables["brightness_temperature"][:]

        valid = surface_type_data > 0
        valid *= t2m > 0
        valid *= tcwv > 0
        if self.surface_type > 0:
            valid *= surface_type_data == self.surface_type

        inds = np.arange(np.sum(valid))

        tcwv = tcwv[inds].reshape(-1, 1)
        t2m = t2m[inds].reshape(-1, 1)

        print("loading 1")

        surface_type_data = surface_type_data[inds].reshape(-1, 1)
        surface_type_min = 1
        surface_type_max = 15
        n_classes = int(surface_type_max - surface_type_min)
        surface_type_1h = np.zeros((surface_type_data.size, n_classes),
                                   dtype=np.float32)
        indices = (surface_type_data - surface_type_min).astype(int)
        surface_type_1h[np.arange(surface_type_1h.shape[0]),
                        indices.ravel()] = 1.0
        bt = bt[inds]

        print("loading 2")
        self.n_obs = bt.shape[-1]
        self.n_surface_classes = n_classes
        if self.surface_type < 0:
            self.x = np.concatenate([bt, t2m, tcwv, surface_type_1h], axis=1)
        else:
            self.x = np.concatenate([bt, t2m, tcwv], axis=1)
        self.x = self.x.data
        self.y = self.file.variables["surface_precipitation"][inds]
        self.y = self.y.data.reshape(-1, 1)
        self.input_features = self.x.shape[1]


###############################################################################
# MHS Dataset
###############################################################################

class MHSDataset(GPROFDataset):
    """
    Pytorch dataset for the Gprof training data.

    This class is a wrapper around the netCDF4 files that are used to store
    the GProf training data. It provides as input vector the brightness
    temperatures and as output vector the surface precipitation.

    """
    def __init__(self,
                 path,
                 batch_size=None,
                 surface_type=-1,
                 normalizer=None,
                 log_rain_rates=False,
                 rain_threshold=None,
                 normalize=True):
        self.normalize = normalize
        self.surface_type = surface_type
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size
            surface_type: If positive, only samples of the given surface type
                are provided. Otherwise the surface type index is provided as
                input features.
            normalization_data: Filename of normalization data to use to
                normalize inputs.
            log_rain_rates: Boolean indicating whether or not to transform
                output to log space.
            rain_threshold: If provided, the given value is applied to the
                output rain rates to turn them into binary non-raining /
                raining labels.
        """
        super().__init__(path,
                         batch_size,
                         normalizer,
                         log_rain_rates,
                         rain_threshold)

    def _get_normalizer(self):
        if self.normalize:
            x = self.x.astype(np.float64)
            x_mean = np.mean(x, axis=0, keepdims=True)
            x_sigma = np.std(x, axis=0, keepdims=True)
            if self.surface_type < 0:
                x_mean[0, 7:-1] = -0.5
                x_sigma[0, 7:-1] = 1.0
            return Normalizer(x_mean, x_sigma)
        return Normalizer(np.array([0.0]), np.array([1.0]))

    def _load_data(self, path):
        self.file = netCDF4.Dataset(path, mode="r")

        tcwv = self.file.variables["tcwv"][:]
        surface_type_data = self.file.variables["surface_type"][:]
        t2m = self.file.variables["t2m"][:]
        bt = self.file.variables["brightness_temperature"][:]

        valid = surface_type_data > 0
        valid *= t2m > 0
        valid *= tcwv > 0
        if self.surface_type > 0:
            valid *= surface_type_data == self.surface_type
        self.valid_samples = valid

        inds = np.arange(np.sum(valid))

        tcwv = tcwv[inds].reshape(-1, 1, 1)
        t2m = t2m[inds].reshape(-1, 1, 1)

        surface_type_data = surface_type_data[inds].reshape(-1, 1)
        surface_type_min = 1
        surface_type_max = 15
        n_classes = int(surface_type_max - surface_type_min)
        surface_type_1h = np.zeros((surface_type_data.size, n_classes),
                                   dtype=np.float32)
        indices = (surface_type_data - surface_type_min).astype(int)
        surface_type_1h[np.arange(surface_type_1h.shape[0]),
                        indices.ravel()] = 1.0
        surface_type_1h = surface_type_1h.reshape(-1, 1, n_classes)

        viewing_angles = self.file.variables["viewing_angles"][:]
        viewing_angles = viewing_angles.reshape(1, -1, 1)
        self.n_obs = bt.shape[-1]
        self.n_surface_classes = n_classes
        bt = bt[inds]

        tcwv = np.broadcast_to(tcwv, bt.shape[:2] + (1,))
        tcwv = tcwv.reshape(-1, tcwv.shape[-1])
        t2m = np.broadcast_to(t2m, bt.shape[:2] + (1,))
        t2m = t2m.reshape(-1, t2m.shape[-1])
        surface_type_1h = np.broadcast_to(surface_type_1h,
                                          bt.shape[:2] + (n_classes,))
        surface_type_1h = surface_type_1h.reshape(-1,
                                                  surface_type_1h.shape[-1])
        viewing_angles = np.broadcast_to(viewing_angles,
                                         (bt.shape[:2] + (1,)))
        viewing_angles = viewing_angles.reshape(-1, viewing_angles.shape[-1])

        bt = bt.reshape(-1, bt.shape[-1])

        if self.surface_type < 0:
            self.x = np.concatenate(
                [bt, t2m, tcwv, surface_type_1h, viewing_angles], axis=1
            )
        else:
            self.x = np.concatenate([bt, t2m, tcwv, viewing_angles], axis=1)
        self.x = self.x.data

        self.y = self.file.variables["surface_precipitation"][inds]
        self.y = self.y.data.reshape(-1, 1)

        self.input_features = self.x.shape[1]

    def evaluate(self, model):

        if self.batch_size:
            batch_size = self.batch_size
        else:
            batch_size = 1

        i_start = 0
        n_samples = self.x.shape[0]
        viewing_angle_mean = self.normalizer.x_mean[0, -1]
        viewing_angle_std = self.normalizer.x_sigma[0, -1]

        columns = ["y_true",
                   "surface_type",
                   "viewing_angle",
                   "y_mean"]
        columns += [rf"y({t})" for t in model.quantiles]
        results = pd.DataFrame(columns=columns)

        while i_start < n_samples:
            x = self.x[i_start:i_start + batch_size, :]
            y = self.y[i_start:i_start + batch_size, :]
            y_pred = model.predict(x)
            y_mean = model.posterior_mean(x)
            surface_type = np.where(x[:, 7:-1] - 0.5)[1]
            viewing_angles = x[:, -1] * viewing_angle_std + viewing_angle_mean
            viewing_angles = np.round(viewing_angles, decimals=2)

            results_tmp = np.concatenate([y,
                                          surface_type.reshape(-1, 1),
                                          viewing_angles.reshape(-1, 1),
                                          y_mean.reshape(-1, 1),
                                          y_pred],
                                         axis=-1)
            results = results.append(pd.DataFrame(results_tmp, columns=columns))
            i_start += batch_size
        return results


###############################################################################
# GProf spatial data
###############################################################################

class GprofSpatialData:
    def __init__(self, filename, n=-1):
        self.file = Dataset(filename)

        x = self.file["x"][:n, :, :, :]

        valid = []
        x_normed = np.zeros(x.shape, dtype=np.float32)

        self.x_mins = np.zeros(x.shape[-1])
        self.x_maxs = np.zeros(x.shape[-1])

        for i in range(x.shape[-1]):

            valid = np.logical_and(x[:, :, :, i] > 0.0,
                                   x[:, :, :, i] < 500.0)
            x_min = x[valid, i].min()
            x_max = x[valid, i].max()
            x_normed[:, :, :, i] = -0.9 + 1.9 * (x[:, :, :, i] - x_min)
            x_normed /= (x_max - x_min)
            x_normed[~valid, i] = -1.0
            self.x_maxs[i] = x_max
            self.x_mins[i] = x_min

        self.x = np.transpose(x_normed, (0, 3, 1, 2))

        self.y = self.file["y"][:n, :, :]
        valid = self.y >= 0.0
        self.y[~valid] = -1.0

        n = self.y.shape[1] // 2
        self.y[:, :, :n - 10] = -1
        self.y[:, :, (n + 10):] = -1

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        x = torch.tensor(self.x[index, :, 3:-2, 3:-2])
        y = torch.tensor(self.y[index, 3:-2, 3:-2])
        y[:10, :] = -1
        y[-10:, :] = -1
        return (x, y)

    def plot_colocation(self):

        ind = np.random.randint(self.x.shape[0])

        f = plt.figure(figsize=(16, 4))
        gs = GridSpec(1, 4)

        for i, j in enumerate([5, 9, 12]):
            ax = plt.subplot(gs[i])
            ax.pcolormesh(self.x[ind, :, :, j])

        ax = plt.subplot(gs[-1])
        ax.pcolormesh(self.y[ind, :, :])
        return f
