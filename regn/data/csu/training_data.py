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
from quantnn.normalizer import Normalizer

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
                 shuffle=True):
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
        else:
            self.normalizer = normalizer

        if normalize:
            self.x = self.normalizer(self.x)

        if transform_zero_rain:
            self._transform_zero_rain()

        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)



    def _transform_zero_rain(self):
        """
        Transforms rain amounts smaller than 1e-4 to a random amount withing
        the range [1e-6, 1e-4], which helps to stabilize training of QRNNs.
        """
        indices = self.y < 1e-4
        self.y[indices] = 10.0 ** np.random.uniform(-6, -4, indices.sum())

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
            chunk_size = 1024
            v = dataset["brightness_temps"]
            while index_start < n:
                index_end = index_start + chunk_size
                bts[index_start: index_end, :] = v[index_start: index_end, :].data
                index_start += chunk_size

            invalid = (bts > 500.0) * (bts < 0.0)
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

    def __len__(self):
        """
        The number of samples in the dataset.
        """
        if self.batch_size:
            n = self.x.shape[0] // self.batch_size
            if (self.x.shape[0] % self.batch_size) > 0:
                n += 1
            return n
        else:
            return self.x.shape[0]
