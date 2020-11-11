import argparse
from pathlib import Path

import numpy as np
from regn.data.gprof import MHSDataset, GMIDataset
from regn.models.torch import FullyConnected
from typhon.retrieval.qrnn import set_backend, QRNN
import typhon.retrieval.qrnn.qrnn
from torch.utils.data import DataLoader

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

args = parser.parse_args()
training_data = args.training_data[0]
validation_data = args.validation_data[0]
sensor = args.sensor[0].lower()

model_path = Path(args.model_path[0])
model_path.mkdir(parents=False, exist_ok=True)


n_layers = args.n_layers[0]
n_neurons = args.n_neurons[0]

network_name = f"qrnn_{n_layers}_{n_neurons}_relu_bn.pt"
results_name = f"qrnn_{n_layers}_{n_neurons}_relu_bn.dat"

#
# Load the data.
#

data_classes = {"gmi": GMIDataset,
                "mhs": MHSDataset}

data_class = data_classes[sensor]
training_data = data_class(training_data,
                        batch_size=512)
validation_data = data_class(validation_data,
                          normalizer=training_data.normalizer,
                          batch_size=512)
training_data = DataLoader(training_data, batch_size=None, num_workers=4, pin_memory=True)
validation_data = DataLoader(validation_data, batch_size=None, num_workers=4, pin_memory=True)

#
# Create model
#

set_backend("pytorch")
quantiles = np.array([0.01, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99])
model = FullyConnected(training_data.dataset.input_features,
                       quantiles,
                       n_layers,
                       n_neurons,
                       batch_norm=True,
                       skip_connections=False)
model.quantiles = quantiles
model.backend = "typhon.retrieval.qrnn.models.pytorch"
qrnn = QRNN(20, model=model)

losses = qrnn.train(training_data=training_data,
                    validation_data=validation_data,
                    initial_learning_rate=1.0,
                    convergence_epochs=2,
                    delta_at=1e-3,
                    maximum_epochs=100,
gpu=True)

#
# Store results
#

qrnn.save(model_path / network_name)
training_errors = losses["training_errors"]
validation_errors = losses["validation_errors"]
np.savetxt(losses_name, np.stack((training_errors, validation_errors)))
