import argparse
from pathlib import Path

import numpy as np
from regn.data.gprof import MHSDataset, GMIDataset
from regn.models.torch import FullyConnected
from quantnn import set_backend, QRNN
from torch.utils.data import DataLoader
import torch
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
parser.add_argument('--log', action='store_true',
                    help='Whether or not to train on logarithmic rain rates.')

args = parser.parse_args()
training_data = args.training_data[0]
validation_data = args.validation_data[0]
sensor = args.sensor[0].lower()
batch_norm = args.batch_norm
skip_connections = args.skip_connections
log = args.log

model_path = Path(args.model_path[0])
model_path.mkdir(parents=False, exist_ok=True)


n_layers = args.n_layers[0]
n_neurons = args.n_neurons[0]

network_name = f"qrnn_{sensor}_{n_layers}_{n_neurons}_relu_{batch_norm}_{skip_connections}_{log}.pt"
results_name = f"qrnn_{sensor}_{n_layers}_{n_neurons}_relu_{batch_norm}_{skip_connections}_{log}.dat"

#
# Load the data.
#

data_classes = {"gmi": GMIDataset,
                "mhs": MHSDataset}

data_class = data_classes[sensor]
training_data = data_class(training_data,
                        log_rain_rates=log,
                        batch_size=512)
validation_data = data_class(validation_data,
                          normalizer=training_data.normalizer,
                          log_rain_rates=log,
                          batch_size=512)
training_data = DataLoader(training_data, batch_size=None, num_workers=4, pin_memory=True)
validation_data = DataLoader(validation_data, batch_size=None, num_workers=4, pin_memory=True)

#
# Create model
#

set_backend("pytorch")
quantiles = np.array([0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95, 0.99])
model = FullyConnected(training_data.dataset.input_features,
                       quantiles,
                       n_layers,
                       n_neurons,
                       batch_norm=batch_norm,
                       log=False,
                       skip_connections=skip_connections)
model.quantiles = quantiles
model.backend = "quantnn.models.pytorch"
qrnn = QRNN(training_data.dataset.input_features, model=model)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           convergence_epochs=0,
           delta_at=1e-3,
           maximum_epochs=20,
           optimizer=optimizer,
           learning_rate_scheduler=scheduler,
           gpu=True)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           convergence_epochs=0,
           delta_at=1e-3,
           maximum_epochs=20,
           optimizer=optimizer,
           learning_rate_scheduler=scheduler,
           gpu=True)
optimizer = optim.SGD(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 20)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           convergence_epochs=0,
           delta_at=1e-3,
           maximum_epochs=20,
           optimizer=optimizer,
           learning_rate_scheduler=scheduler,
           gpu=True)



#
# Store results
#

qrnn.save(model_path / network_name)
training_errors = losses["training_errors"]
validation_errors = losses["validation_errors"]
np.savetxt(results_name, np.stack((training_errors, validation_errors)))
