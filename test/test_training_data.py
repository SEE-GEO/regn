"""
Tests to test the extraction of training data from the GPROF
retrieval database.
"""
from pathlib import Path

import numpy as np
import torch

from regn.data.csu.training_data import (GPROFDataset,
                                         GPROFConvDataset)


def test_gprof_dataset():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "dataset_simple.nc"
    dataset = GPROFDataset(input_file, batch_size=2)

    xs = []
    ys = []

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = dataset.y.sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = ys.sum(dim=0).detach().numpy()

    assert np.all(np.isclose(x_mean, x_mean_ref, atol=1e-6))
    assert np.all(np.isclose(y_mean, y_mean_ref, atol=1e-6))


def test_gprof_conv_dataset():
    """
    Ensure that iterating over single-pixel dataset conserves
    statistics.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "dataset_conv.nc"
    dataset = GPROFConvDataset(input_file, batch_size=1)

    xs = []
    ys = []

    x_mean_ref = dataset.x.sum(axis=0)
    y_mean_ref = dataset.y.sum(axis=0)

    for x, y in dataset:
        xs.append(x)
        ys.append(y)

    xs = torch.cat(xs, dim=0)
    ys = torch.cat(ys, dim=0)

    x_mean = xs.sum(dim=0).detach().numpy()
    y_mean = ys.sum(dim=0).detach().numpy()

    assert np.all(np.isclose(x_mean, x_mean_ref, atol=1e-6))
    assert np.all(np.isclose(y_mean, y_mean_ref, atol=1e-6))



