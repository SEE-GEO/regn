"""
Training script for the GPROF-NN-0D retrieval.
"""
import argparse
from pathlib import Path

import numpy as np
from regn.data.csu.training_data import (GPROF0DDataset,
                                         GPROF0DDatasetLazy)
from regn.models.pytorch import GPROFNN0D
from quantnn import QRNN
from quantnn.data import DataFolder, LazyDataFolder
from quantnn.normalizer import Normalizer
from quantnn.models.pytorch.logging import TensorBoardLogger
from quantnn.metrics import ScatterPlot
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(
        description='Training script for the GPROF-NN-0D algorithm.')
parser.add_argument('training_data', metavar='training_data', type=str, nargs=1,
                    help='Path to training data.')
parser.add_argument('validation_data', metavar='validation_data', type=str, nargs=1,
                    help='Path to validation data.')
parser.add_argument('model_path', metavar='model_path', type=str, nargs=1,
                    help='Where to store the model.')
parser.add_argument('--n_layers', metavar='n_layers', type=int, nargs=1,
                    help='How many fully connected layers.')
parser.add_argument('--n_neurons', metavar='n_neurons', type=int, nargs=1,
                    help='How many neurons per fully-connected layer.')
parser.add_argument('--device', metavar="device", type=str, nargs=1,
                    help="The name of the device on which to run the training")
parser.add_argument('--targets', metavar="target_1 target_2", type=str, nargs="+",
                    help="The target on which to train the network")

args = parser.parse_args()
training_data = args.training_data[0]
validation_data = args.validation_data[0]
n_layers = args.n_layers[0]
n_neurons = args.n_neurons[0]
device = args.device[0]
targets = args.targets

#
# Prepare output
#

model_path = Path(args.model_path[0])
model_path.mkdir(parents=False, exist_ok=True)

network_name = f"gprof_nn_0d_gmi_{n_layers}_{n_neurons}.pt"

#
# Load the data.
#

dataset_factory = GPROF0DDataset

normalizer = Normalizer.load(
        "sftp://129.16.35.202/mnt/array1/share/MLDatasets"
        "/gprof/simple/gprof_gmi_normalizer.pckl"
        )
kwargs = {"batch_size": 2048,
          "normalizer": normalizer,
          "target": targets}
training_data = DataFolder(
        training_data,
        dataset_factory,
        kwargs=kwargs,
        n_workers=8)
kwargs = {"batch_size": 16 * 4096,
          "normalizer": normalizer,
          "target": targets}
validation_data = DataFolder(
        validation_data,
        dataset_factory,
        kwargs=kwargs,
        n_workers=1)

#
# Create model
#

quantiles = np.linspace(0.001, 0.999, 128)
model = GPROFNN0D(n_layers,
                  n_neurons,
                  quantiles.size,
                  target=targets,
                  exp_activation=False)
qrnn = QRNN(quantiles, model=model)

n_epochs=20
logger = TensorBoardLogger(n_epochs)
logger.set_attributes({
    "n_layers": n_layers,
    "n_neurons": n_neurons,
    "targets": ", ".join(targets),
    "activation": "none",
    "optimizer": "adam"
    })

#
# Run the training
#
metrics = ["MeanSquaredError", "Bias", "CRPS", "CalibrationPlot"]
scatter_plot = ScatterPlot(np.logspace(-2, 2, 41), log_scale=True)
metrics.append(scatter_plot)


optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device)
qrnn.save(model_path / network_name)
n_epochs=20
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.9)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device)
qrnn.save(model_path / network_name)
