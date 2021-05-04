"""
Training script for the GPROF-NN-0D retrieval.
"""
import argparse
from pathlib import Path

from torch import nn
from torch import optim
import numpy as np
from quantnn import QRNN
from quantnn.drnn import DRNN
from quantnn.data import DataFolder
from quantnn.normalizer import Normalizer
from quantnn.models.pytorch.logging import TensorBoardLogger
from quantnn.metrics import ScatterPlot
from quantnn.transformations import Log

from regn.data.csu.training_data import GPROF0DDataset
from regn.models import GPROFNN0D, BINS, QUANTILES

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
parser.add_argument('--n_layers_body', metavar='n_layers', type=int, nargs=1,
                    help='How many layers in body of network.')
parser.add_argument('--n_layers_head', metavar='n_layers', type=int, nargs=1,
                    help='How many layers in head of network.')
parser.add_argument('--n_neurons', metavar='n_neurons', type=int, nargs=1,
                    help='How many neurons in each hidden layer.')
parser.add_argument('--device', metavar="device", type=str, nargs=1,
                    help="The name of the device on which to run the training")
parser.add_argument('--type', metavar="target_1 target_2", type=str, nargs=1,
                    help="The type of network: drnn, qrnn or qrnn_exp")
parser.add_argument('--targets', metavar="target_1 target_2", type=str, nargs="+",
                    help="The target on which to train the network")
parser.add_argument('--batch_size', metavar="n", type=int, nargs=1,
                    help="The batch size to use for training.")

args = parser.parse_args()
training_data = args.training_data[0]
validation_data = args.validation_data[0]
n_layers_body = args.n_layers_body[0]
n_layers_head = args.n_layers_head[0]
n_neurons = args.n_neurons[0]
device = args.device[0]
targets = args.targets
network_type = args.type[0]
batch_size = args.batch_size[0]

#
# Prepare output
#

model_path = Path(args.model_path[0])
model_path.mkdir(parents=False, exist_ok=True)
network_name = f"gprof_nn_0d_gmi_{network_type}_{n_layers_body}_{n_layers_head}_{n_neurons}.pt"

#
# Load the data.
#

dataset_factory = GPROF0DDataset
normalizer = Normalizer.load("../data/normalizer_gprof_0d_gmi.pckl")
kwargs = {
    "batch_size": batch_size,
    "normalizer": normalizer,
    "target": targets
}
training_data = DataFolder(
    training_data,
    dataset_factory,
    kwargs=kwargs,
    n_workers=1)

kwargs = {
    "batch_size": 8 * batch_size,
    "normalizer": normalizer,
    "target": targets,
    "augment": False
}
validation_data = DataFolder(
    validation_data,
    dataset_factory,
    kwargs=kwargs,
    n_workers=1
)

#
# Create neural network model
#

if network_type == "drnn":
    model = GPROFNN0D(n_layers_body,
                      n_layers_head,
                      n_neurons,
                      128,
                      target=targets,
                      exp_activation=False)
    xrnn = DRNN(BINS, model=model)
elif network_type == "qrnn_exp":
    model = GPROFNN0D(n_layers_body,
                      n_layers_head,
                      n_neurons,
                      128,
                      target=targets,
                      exp_activation=False)
    xrnn = QRNN(QUANTILES, model=model)
else:
    model = GPROFNN0D(n_layers_body,
                      n_layers_head,
                      n_neurons,
                      128,
                      target=targets,
                      exp_activation=False)
    xrnn = QRNN(QUANTILES, model=model)

#
# Run training
#

n_epochs = 50
logger = TensorBoardLogger(n_epochs)
logger.set_attributes({
    "n_layers_body": n_layers_body,
    "n_layers_head": n_layers_head,
    "n_neurons": n_neurons,
    "targets": ", ".join(targets),
    "type": network_type,
    "optimizer": "adam"
    })

metrics = ["MeanSquaredError", "Bias", "CalibrationPlot", "CRPS"]
scatter_plot = ScatterPlot(log_scale=True)
metrics.append(scatter_plot)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5)
xrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           logger=logger,
           metrics=metrics,
           device=device,
           mask=-1)
xrnn.save(model_path / network_name)
