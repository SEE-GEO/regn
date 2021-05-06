"""
===========================
regn.data.csu.training_data
===========================

This module provides interface class to load the training and evaluation
 data for the NN-based GPROF algorithms.
"""
from pathlib import Path
import logging

from netCDF4 import Dataset
import numpy as np
import torch
from tqdm import tqdm
import xarray as xr

import quantnn
from quantnn.normalizer import MinMaxNormalizer, Normalizer
from quantnn.drnn import _to_categorical
from quantnn.utils import apply
import quantnn.quantiles as qq
import quantnn.density as qd

from regn.data.augmentation import extract_subscene, mask_stripe
from regn.data.csu.preprocessor import PreprocessorFile

LOGGER = logging.getLogger(__name__)
_DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    _DEVICE = torch.device("cuda")


_THRESHOLDS = {
    "surface_precip": 1e-4,
    "convective_precip": 1e-4,
    "rain_water_path": 1e-4,
    "ice_water_path": 1e-4,
    "cloud_water_path": 1e-4,
    "total_column_water_vapor": 1e0,
    "rain_water_content": 1e-6,
    "cloud_water_content": 1e-6,
    "snow_water_content": 1e-6,
    "latent_heat": -99999
}

def _apply(f, y):
    """
    Helper function to apply function to single array or element-wise
    to a dict of arrays.
    """
    if isinstance(y, dict):
        return {k: f(y[k]) for k in y}
    else:
        return f(y)

def write_preprocessor_file(input_file,
                            output_file,
                            x=None,
                            n_samples=None,
                            template=None):
    """
    Extract sample from training data file and write to preprocessor format.

    Args:
        input_file: Path to the NetCDF4 file containing the training or test
            data.
        output_file: Path of the file to write the output to.
        n_samples: How many samples to extract from the training data file.
        template: Template preprocessor file use to determine the orbit header
             information. If not provided this data will be filled with dummy
             values.
    """
    data = xr.open_dataset(input_file)
    new_names = {
        "brightness_temps": "brightness_temperatures"
    }
    n_pixels = 2048
    if n_samples is None:
        n_samples = data.samples.size

    if n_samples < n_pixels:
        n_pixels = n_samples

    if n_samples < data.samples.size:
        indices = np.random.permutation(data.samples.size)[:n_samples]
    else:
        indices = slice(0, None)
    data = data[{"samples": indices}].rename(new_names)
    n_scans = data.samples.size // n_pixels
    n_scans += (data.samples.size % n_pixels) > 0

    new_dims = ["scans", "pixels", "channels"]
    new_dataset = {
        "scans": np.arange(n_scans),
        "pixels": np.arange(n_pixels),
        "channels": np.arange(15),
    }
    dims = ("scans", "pixels", "channels")
    shape = ((n_scans, n_pixels, 15))
    for k in data:
        da = data[k]
        n_dims = len(da.dims)
        s = (n_scans, n_pixels) + da.data.shape[1:]
        new_data = np.zeros(s)
        n = da.data.size
        new_data.ravel()[:n] = da.data.ravel()
        new_data.ravel()[n:] = np.nan
        dims = ("scans", "pixels") + da.dims[1:]
        new_dataset[k] = (dims, new_data)

    new_dataset["earth_incidence_angle"] = (
        ("scans", "pixels", "channels"),
        np.broadcast_to(data.attrs["nominal_eia"].reshape(1, 1, -1),
                        (n_scans, n_pixels, 15))
    )

    new_data = xr.Dataset(new_dataset)
    PreprocessorFile.write(output_file, new_data, template=template)


###############################################################################
# Single-pixel observations.
###############################################################################
class GPROF0DDataset:
    """
    Dataset class providing an interface for the single-pixel GPROF
    training dataset mapping TBs and ancillary data to surface precip
    values and other target variables.

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

    def __init__(
        self,
        filename,
        target="surface_precip",
        normalize=True,
        transform_zeros=True,
        batch_size=None,
        normalizer=None,
        shuffle=True,
        augment=True,
    ):
        """
        Create GPROF 0D dataset.

        Args:
            filename: Path to the NetCDF file containing the 0D training data
                to load.
            target: The variable to use as target (output) variable.
            normalize: Whether or not to normalize the input data.
            transform_zeros: Whether or not to replace very small
                values with random values.
            batch_size: Number of samples in each training batch.
            shuffle: Whether or not to shuffle the training data.
            augment: Whether or not to randomly mask high-frequency channels.
        """
        self.filename = Path(filename)
        self.target = target
        self.transform_zeros = transform_zeros
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self._load_data()

        indices_1h = list(range(17, 40))
        if normalizer is None:
            self.normalizer = Normalizer(self.x, exclude_indices=indices_1h)
        elif isinstance(normalizer, type):
            self.normalizer = normalizer(self.x, exclude_indices=indices_1h)
        else:
            self.normalizer = normalizer

        self.normalize = normalize
        if normalize:
            self.x = self.normalizer(self.x)

        if transform_zeros:
            self._transform_zeros()

        self.x = self.x.astype(np.float32)
        if isinstance(self.y, dict):
            self.y = {k: self.y[k].astype(np.float32) for k in self.y}
        else:
            self.y = self.y.astype(np.float32)

        self._shuffled = False
        if self.shuffle:
            self._shuffle()

    def __repr__(self):
        return f"GPROF0DDataset({self.filename.name}, n_batches={len(self)})"

    def __str__(self):
        return f"GPROF0DDataset({self.filename.name}, n_batches={len(self)})"

    def _transform_zeros(self):
        """
        Transforms target values that are zero to small, non-zero values.
        """
        if isinstance(self.y, dict):
            for k, y_k in self.y.items():
                threshold = _THRESHOLDS[k]
                indices = (y_k < threshold) * (y_k >= 0.0)
                y_k[indices] = np.random.uniform(threshold * 0.01,
                                                 threshold,
                                                 indices.sum())
        else:
            threshold = _THRESHOLDS[self.target]
            y = self.y
            indices = (y < threshold) * (y >= 0.0)
            y[indices] = np.random.uniform(threshold * 0.01,
                                           threshold,
                                           indices.sum())

    def _load_data(self):
        """
        Loads the data from the file into the classes ``x`` attribute.
        """
        with Dataset(self.filename, "r") as dataset:

            variables = dataset.variables
            n = dataset.dimensions["samples"].size

            #
            # Input data
            #

            # Brightness temperatures
            m = dataset.dimensions["channel"].size
            bts = dataset["brightness_temps"][:]

            invalid = (bts > 500.0) + (bts < 0.0)
            bts[invalid] = np.nan

            # Simulate missing high-frequency channels
            if self.augment:
                r = np.random.rand(bts.shape[0])
                bts[r > 0.8, 10:15] = np.nan

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

            n = dataset.dimensions["samples"].size
            if isinstance(self.target, list):
                self.y = {}
                for l in self.target:
                    y = variables[l][:]
                    y[y < -400] = -9999
                    self.y[l] = y
            else:
                y = variables[self.target][:]
                y[y < -400] = -9999
                self.y = y


            LOGGER.info("Loaded %s samples from %s",
                        self.x.shape[0],
                        self.filename.name)

    def save_data(self, filename):
        if self.normalize:
            x = self.normalizer.invert(self.x)
        else:
            x = self.x

        if self.binned:
            centers = 0.5 * (self.bins[1:] + self.bins[:-1])
            y = centers[self.y]
        else:
            y = self.y

        bts = x[:, :15]
        t2m = x[:, 15]
        tcwv = x[:, 16]
        st = np.where(x[:, 17:17+19])[1]
        at = np.where(x[:, 17+19:17+23])[1]

        dataset = xr.open_dataset(self.filename)

        dims = ("samples", "channel")
        new_dataset = {
            "brightness_temps": (dims, bts),
            "two_meter_temperature": (dims[:1], t2m),
            "total_column_water_vapor": (dims[:1], tcwv),
            "surface_type": (dims[:1], st),
            "airmass_type": (dims[:1], at),
            "surface_precip": (dims[:1], y)
        }
        new_dataset = xr.Dataset(new_dataset)
        new_dataset.attrs = dataset.attrs

        new_dataset.to_netcdf(filename)

    def _shuffle(self):
        if not self._shuffled:
            LOGGER.info("Shuffling dataset %s.", self.filename.name)
            indices = np.random.permutation(self.x.shape[0])
            self.x = self.x[indices, :]
            if isinstance(self.y, dict):
                self.y = {k: self.y[k][indices] for k in self.y}
            else:
                self.y = self.y[indices]
            self._shuffled = True

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if i >= len(self):
            LOGGER.info("Finished iterating through dataset %s.",
                        self.filename.name)
            raise IndexError()
        if i == 0:
            self._shuffle()
            if self.transform_zeros:
                self._transform_zeros()

        self._shuffled = False
        if self.batch_size is None:
            if self.isinstance(self.y, dict):
                return (torch.tensor(self.x[[i], :]),
                        {k: torch.tensor(self.y[k][[i]]) for k in self.y})

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)

        x = torch.tensor(self.x[i_start:i_end, :])
        if isinstance(self.y, dict):
            y = {k: torch.tensor(self.y[k][i_start:i_end]) for k in self.y}
        else:
            y = torch.tensor(self.y[i_start:i_end])

        return x, y

    def __len__(self):
        """
        The number of samples in the dataset.
        """
        if self.batch_size:
            n = self.x.shape[0] // self.batch_size
            return n
        else:
            return self.x.shape[0]

    def evaluate(self, model, batch_size=16384, device=_DEVICE):
        """
        Run retrieval on test dataset and returns results as
        xarray Dataset.

        Args:
            model: The QRNN or DRNN model to evaluate.
            batch_size: The batch size to use for the evaluation.
            device: On which device to run the evaluation.

        Return:
            ``xarray.Dataset`` containing the predicted and reference values
            for the data in this dataset.
        """
        n_samples = self.x.shape[0]
        y_means = []
        y_medians = []
        dy_means = []
        dy_medians = []
        pops = []
        y_trues = []
        surfaces = []
        airmasses = []
        y_samples = []

        st_indices = torch.arange(19).reshape(1, -1).to(device)
        am_indices = torch.arange(4).reshape(1, -1).to(device)
        i_start = 0
        model.model.to(device)

        with torch.no_grad():
            for i in tqdm(range(n_samples // batch_size + 1)):
                i_start = i * batch_size
                i_end = i_start + batch_size
                if i_start >= n_samples:
                    break

                x = torch.tensor(self.x[i_start:i_end]).float().to(device)
                y = torch.tensor(self.y[i_start:i_end]).float().to(device)
                i_start += batch_size

                y_pred = model.predict(x)
                y_mean = model.posterior_mean(y_pred=y_pred).reshape(-1)
                dy_mean = y_mean - y
                y_median = model.posterior_quantiles(
                    y_pred=y_pred, quantiles=[0.5]
                ).squeeze(1)
                y_sample = model.sample_posterior(y_pred=y_pred).squeeze(1)
                dy_median = y_median - y

                y_samples.append(y_sample.cpu())
                y_means.append(y_mean.cpu())
                dy_means.append(dy_mean.cpu())
                y_medians.append(y_median.cpu())
                dy_medians.append(dy_median.cpu())

                pops.append(model.probability_larger_than(y_pred=y_pred, y=1e-2).cpu())
                y_trues.append(y.cpu())

                surfaces += [(x[:, 17:36] * st_indices).sum(1).cpu()]
                airmasses += [(x[:, 36:] * am_indices).sum(1).cpu()]

        y_means = torch.cat(y_means, 0).detach().numpy()
        y_medians = torch.cat(y_medians, 0).detach().numpy()
        y_samples = torch.cat(y_samples, 0).detach().numpy()
        dy_means = torch.cat(dy_means, 0).detach().numpy()
        dy_medians = torch.cat(dy_medians, 0).detach().numpy()
        pops = torch.cat(pops, 0).detach().numpy()
        y_trues = torch.cat(y_trues, 0).detach().numpy()
        surfaces = torch.cat(surfaces, 0).detach().numpy()
        airmasses = torch.cat(airmasses, 0).detach().numpy()

        dims = ["samples"]

        data = {
            "y_mean": (dims, y_means),
            "y_sampled": (dims, y_samples),
            "y_median": (dims, y_medians),
            "dy_mean": (dims, dy_means),
            "dy_median": (dims, dy_medians), "y": (dims, y_trues),
            "pop": (dims, pops),
            "surface_type": (dims, surfaces),
            "airmass_type": (dims, airmasses),
        }
        return xr.Dataset(data)

    def evaluate_sensitivity(self, model, batch_size=512, device=_DEVICE):
        """
        Run retrieval on dataset.
        """
        n_samples = self.x.shape[0]
        y_means = []
        y_trues = []
        grads = []
        surfaces = []
        airmasses = []
        dydxs = []

        st_indices = torch.arange(19).reshape(1, -1).to(device)
        am_indices = torch.arange(4).reshape(1, -1).to(device)
        i_start = 0
        model.model.to(device)

        loss = torch.nn.MSELoss()

        model.model.eval()

        for i in tqdm(range(n_samples // batch_size + 1)):
            i_start = i * batch_size
            i_end = i_start + batch_size
            if i_start >= n_samples:
                break

            model.model.zero_grad()

            x = torch.tensor(self.x[i_start:i_end]).float().to(device)
            x_l = x.clone()
            x_l[:, 0] -= 0.01
            x_r = x.clone()
            x_r[:, 0] += 0.01
            y = torch.tensor(self.y[i_start:i_end]).float().to(device)

            x.requires_grad = True
            y.requires_grad = True
            i_start += batch_size

            y_pred = model.predict(x)
            y_mean = model.posterior_mean(y_pred=y_pred).reshape(-1)
            y_mean_l = model.posterior_mean(x_l).reshape(-1)
            y_mean_r = model.posterior_mean(x_r).reshape(-1)
            dydx = (y_mean_r - y_mean_l) / 0.02
            torch.sum(y_mean).backward()

            y_means.append(y_mean.detach().cpu())
            y_trues.append(y.detach().cpu())
            grads.append(x.grad[:, :15].cpu())
            surfaces += [(x[:, 17:36] * st_indices).sum(1).cpu()]
            airmasses += [(x[:, 36:] * am_indices).sum(1).cpu()]
            dydxs += [dydx.cpu()]

        y_means = torch.cat(y_means, 0).detach().numpy()
        y_trues = torch.cat(y_trues, 0).detach().numpy()
        grads = torch.cat(grads, 0).detach().numpy()
        surfaces = torch.cat(surfaces, 0).detach().numpy()
        airmasses = torch.cat(airmasses, 0).detach().numpy()
        dydxs = torch.cat(dydxs, 0).detach().numpy()

        dims = ["samples"]

        data = {
            "gradients": (dims + ["channels",], grads),
            "surface_type": (dims, surfaces),
            "airmass_type": (dims, airmasses),
            "y_mean": (dims, y_means),
            "y_true": (dims, y_trues),
            "dydxs": (dims, dydxs)
        }
        return xr.Dataset(data)

class GPROFValidationDataset(GPROF0DDataset):
    """
    Specialization of the GPROF single-pixel dataset to be used for
    validation. This class will neither shuffle the data nor replace
    zero values by zero and will add geolocation information to the
    evaluation results.


    Attributes:
        lats: Vector containing the latitude of each sample in the
             validation data set.
        lons: Vector containing the longitude of each sample in the
             validation data set.
    """
    def __init__(
        self,
        filename,
        target="surface_precip",
        normalize=True,
        batch_size=None,
        normalizer=None,
    ):
        super().__init__(filename,
                         target=target,
                         normalize=normalize,
                         transform_zeros=False,
                         batch_size=batch_size,
                         normalizer=normalizer,
                         shuffle=False,
                         bins=None)


class GPROF0DDatasetLazy:
    """
    Dataset class providing an interface for the single-pixel GPROF
    training dataset mapping TBs and ancillary data to surface precip
    values and other target variables.

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

    def __init__(
        self,
        filename,
        target="surface_precip",
        normalizer=None,
        transform_zeros=True,
        batch_size=None,
        shuffle=True,
        bins=None,
    ):
        """
        Create GPROF 0D dataset.

        Args:
            filename: Path to the NetCDF file containing the data to
                 0D training data toload.
            target: The variable to use as target (output) variable.
            transform_zeros: Whether or not to replace very small
                 and zero rain with random amounts in the range [1e-6, 1e-4]
            batch_size: Number of samples in each training batch.
            shuffle: Whether or not to shuffle the training data.
            bins: If given, used to transform the training data to categorical
                 variables by binning using the bin boundaries in ``bins``.
        """
        self.filename = Path(filename)
        self.target = target
        self.shuffle = shuffle
        self.bins = bins

        # Determine numbers of samples in dataset.
        with Dataset(self.filename, "r") as dataset:
            self.n_samples = dataset.dimensions["samples"].size

        if batch_size is None:
            self.batch_size = self.n_samples
        else:
            self.batch_size = batch_size

        if normalizer is None:
            self.normalizer = quantnn.normalizer.Identity()
        else:
            self.normalizer = normalizer

        if bins is not None:
            self.binned = True
        else:
            self.binned = False

        if self.shuffle:
            self.indices = np.random.permutation(self.n_samples)
        else:
            self.indices = np.arange(self.n_samples)
        self._shuffled = False


    def transform_zeros(self, y):
        """
        Transforms target values that are zero to small, non-zero values.
        """
        thresh = 1e-4

        def transform(y):
            indices = (y < thresh) * (y >= 0.0)
            y[indices] = np.random.uniform(thresh * 0.1,
                                           thresh,
                                           indices.sum())
            return y

        return _apply(transform, y)

    def transform_categorical(self, y):
        """
        Transforms target values that are zero to small, non-zero values.
        """
        def transform(y):
            return _to_categorical(y, self.bins)

        return _apply(transform, y)


    def load_batch(self, i):
        """
        Loads the data from the file into the classes ``x`` attribute.
        """
        i_start = i * self.batch_size
        i_end = i_start + self.batch_size
        indices = self.indices[i_start:i_end]

        with Dataset(self.filename, "r") as dataset:

            variables = dataset.variables
            n = self.batch_size

            #
            # Input data
            #


            bts = dataset["brightness_temps"][indices]
            invalid = (bts > 500.0) + (bts < 0.0)
            bts[invalid] = -1.0

            # 2m temperature
            t2m = variables["two_meter_temperature"][indices].reshape(-1, 1)
            # Total precitable water.
            tcwv = variables["total_column_water_vapor"][indices].reshape(-1, 1)
            # Surface type
            st = variables["surface_type"][indices]
            n_types = 19
            st_1h = np.zeros((n, n_types), dtype=np.float32)
            st_1h[np.arange(n), st.ravel()] = 1.0
            # Airmass type
            am = variables["airmass_type"][indices]
            n_types = 4
            am_1h = np.zeros((n, n_types), dtype=np.float32)
            am_1h[np.arange(n), am.ravel()] = 1.0

            x = np.concatenate([bts, t2m, tcwv, st_1h, am_1h], axis=1)

            #
            # Output data
            #

            if isinstance(self.target, list):
                y = {l: variables[l][indices] for l in self.target}
            else:
                y = variables[self.target][indices]
        return x, y

    def save_data(self, filename):
        if self.normalize:
            x = self.normalizer.invert(self.x)
        else:
            x = self.x

        if self.binned:
            centers = 0.5 * (self.bins[1:] + self.bins[:-1])
            y = centers[self.y]
        else:
            y = self.y

        bts = x[:, :15]
        t2m = x[:, 15]
        tcwv = x[:, 16]
        st = np.where(x[:, 17:17+19])[1]
        at = np.where(x[:, 17+19:17+23])[1]

        dataset = xr.open_dataset(self.filename)

        dims = ("samples", "channel")
        new_dataset = {
            "brightness_temps": (dims, bts),
            "two_meter_temperature": (dims[:1], t2m),
            "total_column_water_vapor": (dims[:1], tcwv),
            "surface_type": (dims[:1], st),
            "airmass_type": (dims[:1], at),
            "surface_precip": (dims[:1], y)
        }
        new_dataset = xr.Dataset(new_dataset)
        new_dataset.attrs = dataset.attrs

        new_dataset.to_netcdf(filename)



    def _shuffle(self):
        if self.shuffle and not self._shuffled:
            indices = np.random.permutation(self.n_samples)
            self._shuffled = True

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if i >= len(self):
            raise IndexError()
        if i == 0:
            self._shuffle()
            if self.transform_zeros:
                self.transform_zeros()
        self._shuffled = False

        x, y = self.load_batch(i)
        x = self.normalizer(x)

        if self.transform_zeros:
            y = self.transform_zeros(y)
        if self.bins:
            y = self.transform_categorical(y)

        def to_tensor(x):
            return torch.tensor(x.astype(np.float32))

        x = to_tensor(x)
        y = _apply(to_tensor, y)

        return x, y

    def __len__(self):
        """
        The number of samples in the dataset.
        """
        return self.n_samples // self.batch_size

    def evaluate(self, model, batch_size=16384, device=_DEVICE):
        """
        Run retrieval on test dataset and returns results as
        xarray Dataset.

        Args:
            model: The QRNN or DRNN model to evaluate.
            batch_size: The batch size to use for the evaluation.
            device: On which device to run the evaluation.

        Return:
            ``xarray.Dataset`` containing the predicted and reference values
            for the data in this dataset.
        """
        n_samples = self.x.shape[0]
        y_means = []
        y_medians = []
        dy_means = []
        dy_medians = []
        pops = []
        y_trues = []
        surfaces = []
        airmasses = []
        y_samples = []

        st_indices = torch.arange(19).reshape(1, -1).to(device)
        am_indices = torch.arange(4).reshape(1, -1).to(device)
        i_start = 0
        model.model.to(device)

        with torch.no_grad():
            for i in tqdm(range(n_samples // batch_size + 1)):
                i_start = i * batch_size
                i_end = i_start + batch_size
                if i_start >= n_samples:
                    break

                x = torch.tensor(self.x[i_start:i_end]).float().to(device)
                y = torch.tensor(self.y[i_start:i_end]).float().to(device)
                i_start += batch_size

                y_pred = model.predict(x)
                y_mean = model.posterior_mean(y_pred=y_pred).reshape(-1)
                dy_mean = y_mean - y
                y_median = model.posterior_quantiles(
                    y_pred=y_pred, quantiles=[0.5]
                ).squeeze(1)
                y_sample = model.sample_posterior(y_pred=y_pred).squeeze(1)
                dy_median = y_median - y

                y_samples.append(y_sample.cpu())
                y_means.append(y_mean.cpu())
                dy_means.append(dy_mean.cpu())
                y_medians.append(y_median.cpu())
                dy_medians.append(dy_median.cpu())

                pops.append(model.probability_larger_than(y_pred=y_pred, y=1e-2).cpu())
                y_trues.append(y.cpu())

                surfaces += [(x[:, 17:36] * st_indices).sum(1).cpu()]
                airmasses += [(x[:, 36:] * am_indices).sum(1).cpu()]

        y_means = torch.cat(y_means, 0).detach().numpy()
        y_medians = torch.cat(y_medians, 0).detach().numpy()
        y_samples = torch.cat(y_samples, 0).detach().numpy()
        dy_means = torch.cat(dy_means, 0).detach().numpy()
        dy_medians = torch.cat(dy_medians, 0).detach().numpy()
        pops = torch.cat(pops, 0).detach().numpy()
        y_trues = torch.cat(y_trues, 0).detach().numpy()
        surfaces = torch.cat(surfaces, 0).detach().numpy()
        airmasses = torch.cat(airmasses, 0).detach().numpy()

        dims = ["samples"]

        data = {
            "y_mean": (dims, y_means),
            "y_sampled": (dims, y_samples),
            "y_median": (dims, y_medians),
            "dy_mean": (dims, dy_means),
            "dy_median": (dims, dy_medians), "y": (dims, y_trues),
            "pop": (dims, pops),
            "surface_type": (dims, surfaces),
            "airmass_type": (dims, airmasses),
        }
        return xr.Dataset(data)

    def evaluate_sensitivity(self, model, batch_size=512, device=_DEVICE):
        """
        Run retrieval on dataset.
        """
        n_samples = self.x.shape[0]
        y_means = []
        y_trues = []
        grads = []
        surfaces = []
        airmasses = []
        dydxs = []

        st_indices = torch.arange(19).reshape(1, -1).to(device)
        am_indices = torch.arange(4).reshape(1, -1).to(device)
        i_start = 0
        model.model.to(device)

        loss = torch.nn.MSELoss()

        model.model.eval()

        for i in tqdm(range(n_samples // batch_size + 1)):
            i_start = i * batch_size
            i_end = i_start + batch_size
            if i_start >= n_samples:
                break

            model.model.zero_grad()

            x = torch.tensor(self.x[i_start:i_end]).float().to(device)
            x_l = x.clone()
            x_l[:, 0] -= 0.01
            x_r = x.clone()
            x_r[:, 0] += 0.01
            y = torch.tensor(self.y[i_start:i_end]).float().to(device)

            x.requires_grad = True
            y.requires_grad = True
            i_start += batch_size

            y_pred = model.predict(x)
            y_mean = model.posterior_mean(y_pred=y_pred).reshape(-1)
            y_mean_l = model.posterior_mean(x_l).reshape(-1)
            y_mean_r = model.posterior_mean(x_r).reshape(-1)
            dydx = (y_mean_r - y_mean_l) / 0.02
            torch.sum(y_mean).backward()

            y_means.append(y_mean.detach().cpu())
            y_trues.append(y.detach().cpu())
            grads.append(x.grad[:, :15].cpu())
            surfaces += [(x[:, 17:36] * st_indices).sum(1).cpu()]
            airmasses += [(x[:, 36:] * am_indices).sum(1).cpu()]
            dydxs += [dydx.cpu()]

        y_means = torch.cat(y_means, 0).detach().numpy()
        y_trues = torch.cat(y_trues, 0).detach().numpy()
        grads = torch.cat(grads, 0).detach().numpy()
        surfaces = torch.cat(surfaces, 0).detach().numpy()
        airmasses = torch.cat(airmasses, 0).detach().numpy()
        dydxs = torch.cat(dydxs, 0).detach().numpy()

        dims = ["samples"]

        data = {
            "gradients": (dims + ["channels",], grads),
            "surface_type": (dims, surfaces),
            "airmass_type": (dims, airmasses),
            "y_mean": (dims, y_means),
            "y_true": (dims, y_trues),
            "dydxs": (dims, dydxs)
        }
        return xr.Dataset(data)

###############################################################################
# Convolutional dataset
###############################################################################


class GPROFConvDataset:
    """
    Dataset interface to load  GPROF training data for a convolutional
    network.

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

    def __init__(
        self,
        filename,
        target="surface_precip",
        normalize=True,
        transform_zeros=True,
        transform_log=False,
        batch_size=None,
        normalizer=None,
        shuffle=True,
        bins=None,
        augment=True,
    ):
        """
        Create GPROF dataset.

        Args:
            filename: Path to the NetCDF file containing the data to
                 load.
            target: The variable to use as target (output) variable.
            normalize: Whether or not to normalize the input data.
            transform_zeros: Whether or not to replace very small
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

        if transform_zeros:
            self._transform_zeros()

        if transform_log:
            self._transform_log()

        self.x = self.x.astype(np.float32)
        if isinstance(self.y, dict):
            self.y = {k: self.y[k].astype(np.float32) for k in self.y}
        else:
            self.y = self.y.astype(np.float32)

        if bins is not None:
            self.binned = True
            self.y = _to_categorical(self.y, bins)
        else:
            self.binned = False

        self._shuffled = False
        if self.shuffle:
            self._shuffle()

    def _transform_zeros(self):
        """
        Transforms target values that are zero to small, non-zero values.
        """
        if isinstance(self.y, dict):
            for k, y_k in self.y.items():
                if np.any(y_k > 0.0):
                    non_zero = np.min(y_k[y_k > 0.0])
                else:
                    non_zero = 0.0
                indices = (y_k < non_zero) * (y_k >= 0.0)
                y_k[indices] = np.random.uniform(non_zero * 0.1,
                                                 non_zero,
                                                 indices.sum())
        else:
            y = self.y
            if np.any(y > 0.0):
                non_zero = np.min(y[y > 0.0])
            else:
                non_zero = 0.0
            indices = (y < non_zero) * (y >= 0.0)
            y[indices] = np.random.uniform(non_zero * 0.1,
                                           non_zero,
                                           indices.sum())

    def _transform_log(self):
        indices = self.y < 0.0
        self.y = np.log10(self.y)
        self.y[indices] = -10

    def _load_data(self):
        """
        Loads the data from the file into the classes ``x`` attribute.
        """

        with Dataset(self.filename, "r") as dataset:

            LOGGER.info("Loading data from file: %s",
                        self.filename.name)

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
                bts = v_bt[index_start:index_end].data
                bt[index_start:index_end] = bts
                sp[index_start:index_end] = v_sp[index_start:index_end].data
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
                    self.x[i] = np.transpose(
                        extract_subscene(bt[i], 0.0, 0.0), [2, 0, 1]
                    )
                    self.y[i] = extract_subscene(sp[i], 0.0, 0.0)
                    continue

                p_in = np.random.uniform(-1, 1)
                p_out = np.random.uniform(-1, 1)
                s_o = np.random.uniform(-1, 1)

                self.x[i] = np.transpose(
                    extract_subscene(bt[i], p_in, p_out, s_o), [2, 0, 1]
                )
                self.y[i] = extract_subscene(sp[i], p_in, p_out, s_o)

                r = np.random.rand()
                if r < 0.2:
                    mask_stripe(self.x[i], p_out)

                r = np.random.rand()
                if r > 0.5:
                    self.x[i] = np.flip(self.x[i], axis=2)
                    self.y[i] = np.flip(self.y[i], axis=1)
                if r > 0.5:
                    self.x[i] = np.flip(self.x[i], axis=1)
                    self.y[i] = np.flip(self.y[i], axis=0)
            self.y[np.isnan(self.y)] = -1.0

    def get_surface_types(self):
        """
        Get surface types for non-augmented (validation) data.
        """
        with Dataset(self.filename, "r") as dataset:

            variables = dataset.variables
            n = dataset.dimensions["samples"].size
            h = dataset.dimensions["scans"].size
            w = dataset.dimensions["pixels"].size
            c = dataset.dimensions["channels"].size

            #
            # Input data
            #

            # Brightness temperatures
            st = np.zeros((n, 128, 128), np.int8)

            index_start = 0
            chunk_size = 128
            v = dataset["surface_type"]
            while index_start < n:
                index_end = index_start + chunk_size
                st[index_start:index_end] = v[
                    index_start:index_end, 110 - 64 : 110 + 64, 110 - 64 : 110 + 64
                ].data
                index_start += chunk_size
            return st

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
            return (torch.tensor(self.x[i]), torch.tensor(self.y[i]))

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)

        x = torch.tensor(self.x[i_start:i_end])
        y = torch.tensor(self.y[i_start:i_end])

        if i + 1 == len(self):
            self._shuffle()
            if not self.binned:
                self._transform_zeros()

        if i >= len(self):
            raise IndexError()
        return x, y

    def __len__(self):
        """
        The number of samples in the dataset.
        """
        if self.batch_size:
            n = self.x.shape[0] // self.batch_size
            return n
        else:
            return self.x.shape[0]

    def evaluate(
        self,
        model,
        surface_types,
        batch_size=128,
        device=_DEVICE,
        log=False,
    ):
        """
        Run retrieval on test dataset and returns results as
        xarray Dataset.

        Args:
            model: The QRNN or DRNN model to evaluate.
            surface_types: The surface types for all samples in
                this dataset. They are provided as external argument
                because using the original data may not be possible if
                the samples in the dataset have been shuffled.
            batch_size: The batch size to use for the evaluation.
            device: On which device to run the evaluation.

        Return:
            ``xarray.Dataset`` containing the predicted and reference values
            for the data in this dataset.
        """
        n_samples = self.x.shape[0]
        y_means = []
        y_medians = []
        y_samples = []
        dy_means = []
        dy_medians = []
        pops = []
        y_trues = []
        surfaces = []
        airmasses = []

        st_indices = torch.arange(19).reshape(1, -1).to(device)
        am_indices = torch.arange(4).reshape(1, -1).to(device)
        i_start = 0
        model.model.to(device)

        with torch.no_grad():
            for i in tqdm(range(n_samples // batch_size + 1)):
                i_start = i * batch_size
                i_end = i_start + batch_size
                sts = surface_types[i_start:i_end]
                if i_start >= n_samples:
                    break

                x = torch.tensor(self.x[i_start:i_end]).float().to(device)
                y = torch.tensor(self.y[i_start:i_end]).float().to(device)
                i_start += batch_size

                y_pred = model.predict(x)
                if log:
                    y_pred = torch.exp(np.log(10) * y_pred)
                y_mean = model.posterior_mean(y_pred=y_pred)
                y_sample = model.sample_posterior(y_pred=y_pred).squeeze(1)
                dy_mean = y_mean - y
                y_median = model.posterior_quantiles(
                    y_pred=y_pred, quantiles=[0.5]
                ).squeeze(1)
                dy_median = y_median - y

                y_mean = y_mean.cpu().numpy()
                y_sample = y_sample.cpu().numpy()
                dy_mean = dy_mean.cpu().numpy()
                y_median = y_median.cpu().numpy()
                dy_median = dy_median.cpu().numpy()
                y_true = y.cpu().numpy()
                pop = model.probability_larger_than(y_pred=y_pred, y=1e-2).cpu().numpy()

                indices = y_true[:, :, :] >= 0.0
                y_means.append(y_mean[indices])
                y_medians.append(y_median[indices])
                y_samples.append(y_sample[indices])
                dy_means.append(dy_mean[indices])
                dy_medians.append(dy_median[indices])
                y_trues.append(y_true[indices])
                pops.append(pop[indices])
                surfaces.append(sts[indices])

        y_means = np.concatenate(y_means, 0)
        y_samples = np.concatenate(y_samples, 0)
        y_medians = np.concatenate(y_medians, 0)
        dy_means = np.concatenate(dy_means, 0)
        dy_medians = np.concatenate(dy_medians, 0)
        pop = np.concatenate(pops, 0)
        y_trues = np.concatenate(y_trues, 0)
        surfaces = np.concatenate(surfaces, 0)

        dims = ["samples"]

        data = {
            "y_mean": (dims, y_means),
            "y_samples": (dims, y_samples),
            "y_median": (dims, y_medians),
            "dy_mean": (dims, dy_means),
            "dy_median": (dims, dy_medians),
            "y": (dims, y_trues),
            "surface_type": (dims, surfaces),
            "pop": (dims, pop),
        }
        return xr.Dataset(data)

class GPROFConvValidationDataset(GPROFConvDataset):
    """
    Specialization of the GPROF convolutional dataset to be used for
    validation. This class will neither shuffle the data nor replace
    zero values by zero.
    """
    def __init__(
            self,
            filename,
            target="surface_precip",
            normalize=True,
            batch_size=None,
            normalizer=None,
    ):
        super().__init__(filename,
                         target=target,
                         normalize=normalize,
                         transform_zeros=False,
                         batch_size=batch_size,
                         normalizer=normalizer,
                         shuffle=False,
                         bins=None)

    def evaluate(
        self,
        model,
        batch_size=32,
        device=_DEVICE,
        log=False,
    ):
        """
        Run retrieval on test dataset and returns results as
        xarray Dataset.

        Args:
            model: The QRNN or DRNN model to evaluate.
            batch_size: The batch size to use for the evaluation.
            device: On which device to run the evaluation.

        Return:
            ``xarray.Dataset`` containing the predicted and reference values
            for the data in this dataset.
        """
        surface_types = self.get_surface_types()
        return GPROFConvDataset.evaluate(
            self,
            model,
            surface_types,
            batch_size=batch_size,
            device=device,
            log=log
        )
