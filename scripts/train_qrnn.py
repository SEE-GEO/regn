import argparse
from pathlib import Path

import numpy as np
from regn.data.csu.training_data import GPROFDataset
from quantnn.models.pytorch.fully_connected import FullyConnected
from quantnn import QRNN
from quantnn.data import DataFolder
from quantnn.normalizer import Normalizer
from quantnn.models.pytorch.logging import TensorBoardLogger
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
parser.add_argument('--sensor', metavar='sensor', type=str, nargs=1,
                    help='The sensor type corresponding to the data (mhs or gmi).',
                    default="gmi")
parser.add_argument('--batch_norm', action='store_true',
                    help='How many neurons per fully-connected layer.')
parser.add_argument('--skip_connections', action='store_true',
                    help='How many neurons per fully-connected layer.')

args = parser.parse_args()
training_data = args.training_data[0]
validation_data = args.validation_data[0]
sensor = args.sensor[0].lower()
batch_norm = args.batch_norm
skip_connections = args.skip_connections

model_path = Path(args.model_path[0])
model_path.mkdir(parents=False, exist_ok=True)


n_layers = args.n_layers[0]
n_neurons = args.n_neurons[0]

network_name = f"gprof_nn_0d_{sensor}_{n_layers}_{n_neurons}.pt"

#
# Load the data.
#

training_path = "/gdata/simon/gprof/gmi/simple/training_data"
validation_path = "/gdata/simon/gprof/gmi/simple/validation_data"

dataset_factory = GPROFDataset

normalizer = Normalizer.load("sftp://129.16.35.202/mnt/array1/share/MLDatasets/gprof/simple/gprof_gmi_normalizer.pckl")
print(normalizer.means)
kwargs = {"batch_size": 512,
          "normalizer": normalizer}

training_data = DataFolder(training_path, dataset_factory, kwargs=kwargs, n_workers=5)
validation_data = DataFolder(validation_path, dataset_factory, kwargs=kwargs, n_workers=1)

#
# Create model
#

quantiles = np.linspace(0.001, 0.999, 128)
model = FullyConnected(40,
                       quantiles.size,
                       n_layers,
                       n_neurons,
                       skip_connections=skip_connections,
                       batch_norm=batch_norm)
qrnn = QRNN(quantiles, model=model)

n_epochs=10
logger = TensorBoardLogger(n_epochs)
logger.set_attributes({
    "n_layers": n_layers,
    "n_neurons": n_neurons
    })


optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           device="gpu")
qrnn.save(model_path / network_name)

n_epochs=10
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           device="gpu")
qrnn.save(model_path / network_name)

n_epochs=20
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           device="gpu")
qrnn.save(model_path / network_name)

n_epochs=40
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           device="gpu")
qrnn.save(model_path / network_name)
