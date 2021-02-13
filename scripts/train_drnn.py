import argparse
from pathlib import Path

import numpy as np
from regn.data.csu.training_data import GPROFDataset
from quantnn.models.pytorch.fully_connected import FullyConnected
from quantnn import DRNN
from quantnn.data import DataFolder
from quantnn.normalizer import Normalizer
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(description='Train fully-connected DRNN')
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

if skip_connections:
    network_name = f"drnn_{sensor}_{n_layers}_{n_neurons}_relu_sc.pt"
    results_name = f"drnn_{sensor}_{n_layers}_{n_neurons}_relu_sc.dat"
else:
    network_name = f"drnn_{sensor}_{n_layers}_{n_neurons}_relu.pt"
    results_name = f"drnn_{sensor}_{n_layers}_{n_neurons}_relu.dat"

#
# Load the data.
#

host = "129.16.35.202"
training_path = "/mnt/array1/share/MLDatasets/gprof/simple/training_data"
validation_path = "/mnt/array1/share/MLDatasets/gprof/simple/validation_data"
dataset_factory = GPROFDataset

normalizer = Normalizer.load("sftp://129.16.35.202/mnt/array1/share/MLDatasets/gprof/simple/gprof_gmi_normalizer.pckl")
print(normalizer.means)
bins = np.logspace(-4, 2.5, 257)
kwargs = {"batch_size": 512,
          "normalizer": normalizer,
          "bins": bins}

path = "sftp://" + host + "/" + training_path
training_data = DataFolder(path, dataset_factory, kwargs=kwargs, n_workers=5)
path = "sftp://" + host + "/" + validation_path
validation_data = DataFolder(path, dataset_factory, kwargs=kwargs, n_workers=1)
#training_data = DataLoader(training_data, batch_size=None, num_workers=1, pin_memory=True)
#validation_data = DataLoader(validation_data, batch_size=None, num_workers=1, pin_memory=True)

#
# Create model
#

model = FullyConnected(40,
                       bins.size - 1,
                       n_layers,
                       n_neurons,
                       skip_connections=skip_connections,
                       batch_norm=batch_norm)
drnn = DRNN(bins, model=model)

n_epochs=5
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
drnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device="gpu")
drnn.save(model_path / network_name)
n_epochs=10
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
drnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device="gpu")
drnn.save(model_path / network_name)
n_epochs=20
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
drnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device="gpu")
drnn.save(model_path / network_name)
n_epochs=40
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
drnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device="gpu")
drnn.save(model_path / network_name)
