import numpy as np
import scipy as sp
from scipy.signal import convolve
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

class BinaryClassification(Dataset):
    """
    A test dataset to test using QRNNs as classifiers.

    This dataset represents a simple binary classification problem. Samples
    consists of points in a two-dimensional space and points from each class
    follow Gaussian distributions with different mean vector and unit covariance
    matrices.

    Attributes:
        x(:code:`np.ndarray`): Numpy array of shape :code:`(n, 1, 64, 64)`
            containing the training data.
        y(:code:`np.ndarray`): Numpy array of shape :code:`(n, 1, 64, 64)`
            containing the output data.
    """
    @staticmethod
    def create_data(n):
        """
        Create samples.

        Args:
            n(:code:`int`): The number of samples to create.
        Returns:
            Tuple :code:`(x, y)` containing the input data :code:`x`
            and output data :code:`y` as numpy arrays.
        """
        n1 = n // 2
        n2 = n - n1

        x1 = np.random.normal(size=(n1, 2)) + np.array([[-1.0, -1.0]])
        y1 = np.zeros((n1, 1))
        x2 = np.random.normal(size=(n2, 2)) + np.array([[1.0, 1.0]])
        y2 = np.ones((n2, 1))

        x = np.concatenate([x1, x2], axis=0)
        y = np.concatenate([y1, y2], axis=0)

        inds = np.random.permutation(np.arange(n))
        x = x[inds, :]
        y = y[inds, :]

        return x, y

    def __init__(self,
                 n,
                 batch_size = 8,
                 shuffle = True):
        """
        Create instance of the dataset.

        Args:
            n: Number of samples in the dataset.
            batch_size: Batch size to use to 
        """
        x, y = BinaryClassification.create_data(n)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self.inds = np.random.permutation(np.arange(n))
        else:
            self.inds = np.arange(n)

    def __len__(self):
        """
        The number of batches the data. This is part of the
        pytorch interface for datasets.

        Return:
            int: The number of samples in the data set
        """
        return self.x.shape[0] // self.batch_size

    def __getitem__(self, i):
        """
        Return element from the dataset. This is part of the
        pytorch interface for datasets.

        Args:
            i(int): The index of the sample to return
        """
        if i >= len(self) and self.shuffle:
            n = self.x.shape[0]
            self.inds = np.random.permutation(np.arange(n))
            raise IndexError()

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)
        return (torch.tensor(self.x[i_start : i_end]).float(),
                torch.tensor(self.y[i_start : i_end]).float())

    def plot_data(self):
        x1 = np.linspace(-2, 2, 101)
        x2 = np.linspace(-2, 2, 101)
        X1, X2 = np.meshgrid(x1, x2, indexing="ij")
        X = np.stack([X1, X2], axis = -1)
        m1 = -np.ones((1, 1, 2))
        m2 = np.ones((1, 1, 2))

        e1 = np.exp(-0.5 * np.sum((X - m2)**2, axis=-1))
        e2 = np.exp(-0.5 * np.sum((X - m1)**2, axis=-1))
        p =  e1 / (e1 + e2)

        f = plt.figure(figsize=(5, 5))
        gs = GridSpec(2, 2, height_ratios=[1.0, 0.05], width_ratios=[1.0, 0.1], figure=f)

        ax = plt.subplot(gs[0, 0])
        img = ax.contourf(X1, X2, p, norm=Normalize(0, 1),
                         levels = np.linspace(0, 1, 11),
                          cmap="coolwarm", alpha=0.5)

        handles = []
        inds = self.y[:, 0] == 0.0
        handles += [ax.scatter(self.x[inds, 0], self.x[inds, 1],
                               c="navy", marker="o", s=4)]
        inds = self.y[:, 0] == 1.0
        handles += [ax.scatter(self.x[inds, 0], self.x[inds, 1],
                               c="firebrick", marker="o", s=5)]
        labels = ["Class 1", "Class 2"]

        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_aspect(1.0)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

        # Colorbar
        ax = plt.subplot(gs[0, 1])
        plt.colorbar(img, cax=ax, label="P(Class 2)")
        ax.set_aspect(10)

        # Legend
        ax = plt.subplot(gs[1, 0])
        ax.set_axis_off()
        ax.legend(handles=handles, labels=labels, loc="upper center", ncol=2)
        ax.set_xlim()

        plt.tight_layout()
        return f

bc = BinaryClassification(10)

import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{lmodern}\usepackage{units}')
f = bc.plot_data()

f, ax = plt.subplots(1, 1)
ax.set_ylabel(r"Latitude [$\unit{^\circ\ N}$]")
