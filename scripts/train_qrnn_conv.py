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
print("device: ", device)

model_path.mkdir(parents=False, exist_ok=True)

network_name = f"qrnn_xception_8_256_only_log.pt"
results_name = f"qrnn_xception_8_256_only_log.dat"

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
#model = XceptionFpn(15, quantiles.size, n_features=256, blocks=8)
#qrnn = QRNN(quantiles, model=model)
qrnn = QRNN.load(model_path / network_name)
model = qrnn.model

#qrnn = QRNN.load(model_path / network_name)
#for m in qrnn.model.modules():
#    m.training = False
#qrnn.model.head.training = True
#for p in qrnn.model.parameters():
#    p.requires_grad = False
#for p in qrnn.model.head.parameters():
#        p.requires_grad = True
#
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from quantnn.models.keras import CosineAnnealing

#n_epochs=40
#optimizer = optim.SGD(qrnn.model.parameters(), lr=0.1, momentum=0.9)
#scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.001)
###scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
#qrnn.train(training_data=training_data,
#           validation_data=validation_data,
#           n_epochs=n_epochs,
#           optimizer=optimizer,
#           scheduler=scheduler,
#           device=device,
#           mask=-10.0)
#qrnn.save(model_path / network_name)
n_epochs=40
optimizer = optim.SGD(qrnn.model.parameters(), lr=0.01, momentum=0.9)
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.0001)
#scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-10.0)
qrnn.save(model_path / network_name)
#

n_epochs=40
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.00001)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-10.0)

qrnn.save(model_path / network_name)
n_epochs=40
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.00001)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-10.0)

qrnn.save(model_path / network_name)
#n_epochs=20
#optimizer = optim.SGD(model.parameters(), lr=0.0010.9)
##scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
#qrnn.train(training_data=training_data,
#           validation_data=validation_data,
#           n_epochs=n_epochs,
#           optimizer=optimizer,
#           scheduler=scheduler,
#           device=device,
#           mask=-1.0)
#qrnn.save(model_path / network_name)
#
#
# Store results
#

qrnn.save(model_path / network_name)
np.savetxt(results_name, np.stack((training_errors, validation_errors)))
