import argparse
from pathlib import Path

import numpy as np
from regn.data.csu.training_data import GPROFDataset
from regn.models.torch import FullyConnectedWithSkips
from quantnn import QRNN
from quantnn.data import SFTPStream
from quantnn.normalizer import Normalizer
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(description='Train fully-connected QRNN')
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

network_name = f"qrnn_{sensor}_{n_layers}_{n_neurons}_relu.pt"
results_name = f"qrnn_{sensor}_{n_layers}_{n_neurons}_relu.dat"

#
# Load the data.
#

host = "129.16.35.202"
training_path = "array1/share/Datasets/gprof/simple/training_data"
validation_path = "array1/share/Datasets/gprof/simple/validation_data"
dataset_factory = GPROFDataset

normalizer = Normalizer.load("sftp://129.16.35.202/mnt/array1/share/Datasets/gprof/simple/gprof_gmi_normalizer.pckl")
print(normalizer.means)
kwargs = {"batch_size": 512,
          "normalizer": normalizer}

training_data = SFTPStream(host, training_path, dataset_factory, kwargs=kwargs, n_workers=5, n_files=1)
validation_data = SFTPStream(host, validation_path, dataset_factory, kwargs=kwargs, n_workers=1)
#training_data = DataLoader(training_data, batch_size=None, num_workers=1, pin_memory=True)
#validation_data = DataLoader(validation_data, batch_size=None, num_workers=1, pin_memory=True)

#
# Create model
#

quantiles = np.linspace(0.01, 0.99, 99)
model = FullyConnectedWithSkips(40,
        quantiles.size,
                       n_layers,
                       n_neurons,
                       batch_norm=batch_norm)
qrnn = QRNN(quantiles, model=model)

n_epochs=10
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.5e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device="gpu")
qrnn.save(model_path / network_name)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device="gpu")
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
losses = qrnn.train(training_data=training_data,
                    validation_data=validation_data,
                    n_epochs=n_epochs,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device="gpu")


#
# Store results
#

qrnn.save(model_path / network_name)
training_errors = losses["training_errors"]
validation_errors = losses["validation_errors"]
np.savetxt(results_name, np.stack((training_errors, validation_errors)))
