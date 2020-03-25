import numpy as np
import scipy as sp
from scipy.signal import convolve
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

class ImageGradients(Dataset):
    """
    A test dataset to test convolutional QRNNs.

    Input data consists of grey-scale images of smoothed Gaussian
    noise. Output data is the magnitude of the image gradients
    corrupted with Gaussian noise whose standard deviation is 
    intensity of the corresponding input pixel scaled by 0.1.

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
        w = 64
        kw = 21

        # Input data: Smooth random field.
        k = np.linspace(-3, 3, kw)
        k = np.exp(-np.sqrt((k ** 2).reshape(1, 1, -1, 1) + (k ** 2).reshape(1, 1, 1, -1)))
        k /= k.sum()

        x = np.random.normal(size=(n, 1, w + kw + 1, w + kw + 1))
        x = convolve(x, k, mode="valid")

        # Compute gradients
        kx = 0.5 * np.array([-1, 0, 1]).reshape(1, 1, 1, 3)
        ky = 0.5 * np.array([-1, 0, 1]).reshape(1, 1, 3, 1)
        dx = convolve(x, kx, mode="valid")[:, :, 1:-1, :]
        dy = convolve(x, ky, mode="valid")[:, :, :, 1:-1]
        m =  np.sqrt(dx ** 2 + dy ** 2)

        # Add noise
        x = x[:, :, 1:-1, 1:-1]
        y = m + 0.1 * x * np.random.normal(size=m.shape)
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
        x, y = ImageGradients.create_data(n)
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
        if i > len(self) and self.shuffle:
            self.inds = np.random.permutation(np.arange(n))

        i_start = self.batch_size * i
        i_end = self.batch_size * (i + 1)
        return (torch.tensor(self.x[i_start : i_end]),
                torch.tensor(self.y[i_start : i_end]))

    def plot_data(self, indices = None):
        """
        Plot samples from dataset.

        Args:
            indices: If provided, list containing indices of samples to plot.
                If None, a random sample from the dataset is plotted.
        Returns:
            Matplotlib figure containing the plotted samples.
        """
        if indices is None:
            indices = [np.random.randint(self.x.shape[0])]
        elif type(indices) == int:
            indices = np.random.randint(self.x.shape[0], size=indices)

        n = len(indices)

        f = plt.figure(figsize = (8, n * 5))
        gs = GridSpec(n + 1, 2, height_ratios = [1.0] * n + [0.1])
        norm_x = Normalize(-0.2, 0.2)
        norm_y = Normalize(-0.05, 0.05)

        for i in range(n):
            ind = indices[i]
            ax1 = plt.subplot(gs[i, 0])
            img1 = ax1.pcolormesh(self.x[ind, 0], norm=norm_x, cmap="cividis")
            ax1.set_aspect(1)
            ax1.set_ylabel("y")

            ax2 = plt.subplot(gs[i, 1])
            img2 = ax2.pcolormesh(self.y[ind, 0], norm=norm_y, cmap="cividis")
            ax2.set_aspect(1)
            ax2.set_yticks([])

            if i == 0:
                ax1.set_title("(a) Input", pad=20, loc="left")
                ax2.set_title("(b) Output", pad=20, loc="left")

            if i == n - 1:
                ax1.set_xlabel("x")
                ax2.set_xlabel("x")

        cax1 = plt.subplot(gs[-1, 0])
        plt.colorbar(img1, cax=cax1, orientation="horizontal", label="Image value")
        cax1.set_aspect(0.1)

        cax2 = plt.subplot(gs[-1, 1])
        plt.colorbar(img2, cax=cax2, orientation="horizontal", label="Gradient")
        cax2.set_aspect(0.1)

        plt.tight_layout()
        return f
