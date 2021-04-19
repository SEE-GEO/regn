
import argparse
from pathlib import Path

import numpy as np
from regn.data.csu.training_data import GPROFConvDataset
from quantnn.models.pytorch.unet import UNet
from quantnn.models.pytorch.resnet import ResNet
from quantnn.models.pytorch.xception import XceptionFpn
from quantnn import QRNN
from quantnn.data import DataFolder
from quantnn.normalizer import Normalizer
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

###############################################################################
# Command line arguments.
###############################################################################

parser = argparse.ArgumentParser(description='Train convolutional QRNN')
parser.add_argument('training_data_path', metavar='training_data',
                    type=str,
                    nargs=1,
                    help='Path to training data.')
parser.add_argument('validation_data_path',
                    metavar='validation_data',
                    type=str,
                    nargs=1,
                    help='Path to validation data.')
parser.add_argument('model_path', metavar='model_path', type=str, nargs=1,
                    help='Where to store the model.')
parser.add_argument('--device', metavar='device', type=str, default="cuda", nargs=1)
args = parser.parse_args()
training_path = args.training_data_path[0]
validation_path = args.validation_data_path[0]
model_path = Path(args.model_path[0])
device = args.device[0]

model_path.mkdir(parents=False, exist_ok=True)
network_name = f"qrnn_xception_8_256_only_log_adam_2.pt"
results_name = f"qrnn_xception_8_256_only_log_adam_2.dat"

#
# Load the data.
#

host = "129.16.35.202"
dataset_factory = GPROFConvDataset

normalizer = Normalizer.load("sftp://129.16.35.202/mnt/array1/share/MLDatasets/gprof/conv/normalizer_gprof_gmi_conv_bt_only.pckl")
kwargs = {"batch_size": 4,
          "normalizer": normalizer,
          "transform_log": True}

training_data = DataFolder(training_path, dataset_factory, kwargs=kwargs, n_workers=5)
path = "sftp://" + host + "/" + validation_path
validation_data = DataFolder(validation_path, dataset_factory, kwargs=kwargs, n_workers=1)

#
# Create model
#

quantiles = np.linspace(1e-3, 1.0 - 1e-3, 128)
model = XceptionFpn(15, quantiles.size, n_features=256, blocks=8)
qrnn = QRNN(quantiles, model=model)
model = qrnn.model

n_epochs=20
optimizer = optim.Adam(qrnn.model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer, 20)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-10.0)
n_epochs=20
optimizer = optim.Adam(qrnn.model.parameters(), lr=0.01)
scheduler = CosineAnnealingLR(optimizer, 20)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-10.0)
n_epochs=20
optimizer = optim.Adam(qrnn.model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, 20)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-10.0)
n_epochs=20
optimizer = optim.Adam(qrnn.model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, 20)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-10.0)


qrnn.save(model_path / network_name)

qrnn.save(model_path / network_name)
np.savetxt(results_name, np.stack((training_errors, validation_errors)))
