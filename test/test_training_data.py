"""
Tests to test the extraction of training data from the GPROF
retrieval database.
"""
from pathlib import Path

import numpy as np
import torch
from quantnn.qrnn import QRNN
from quantnn.models.pytorch.xception import XceptionFpn

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


def test_evaluate_simple():
    """
    Ensure that evaluating a dummy model yields the expected
    results.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "dataset_simple.nc"
    dataset = GPROFDataset(input_file, batch_size=2)
    quantiles = np.linspace(0.01, 0.99, 99)
    qrnn = QRNN(quantiles, n_inputs=40)
    results = dataset.evaluate(qrnn)

    assert "y_mean" in results
    assert "y_median" in results
    assert "dy_mean" in results
    assert "dy_median" in results
    assert "y" in results
    assert "surface_type" in results
    assert "airmass_type" in results

def test_evaluate_gradients():
    """
    Ensure that evaluating a dummy model yields the expected
    results.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "dataset_simple.nc"
    dataset = GPROFDataset(input_file, batch_size=2)
    quantiles = np.linspace(0.01, 0.99, 99)
    qrnn = QRNN(quantiles, n_inputs=40)
    results = dataset.evaluate_sensitivity(qrnn)

    assert "gradients" in results
    assert "surface_type" in results
    assert "airmass_type" in results

def test_gprof_conv_dataset():
    """
    Ensure that iterating over convolutional dataset conserves
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


def test_evaluate_conv():
    """
    Ensure that evaluating a dummy model yields the expected
    results.
    """
    path = Path(__file__).parent
    input_file = path / "data" / "dataset_conv.nc"
    dataset = GPROFConvDataset(input_file, batch_size=1)
    quantiles = np.linspace(0.01, 0.99, 99)
    model = XceptionFpn(15, 99)
    qrnn = QRNN(quantiles, model=model)
    surface_types = dataset.get_surface_types()
    results = dataset.evaluate(qrnn, surface_types)

    assert "y_mean" in results
    assert "y_median" in results
    assert "dy_mean" in results
    assert "dy_median" in results
    assert "y" in results
    assert "surface_type" in results

