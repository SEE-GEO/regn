import argparse
from pathlib import Path

import numpy as np
from regn.data.csu.training_data import GPROFConvDataset
from quantnn.models.pytorch.resnet import ResNet
from quantnn.models.pytorch.xception import XceptionFpn
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

parser = argparse.ArgumentParser(description='Train convolutional DRNN')
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

network_name = f"drnn_xception_8_256_bts_only.pt"
results_name = f"drnn_xception_8_256_bts_only.dat"

#
# Load the data.
#

host = "129.16.35.202"
dataset_factory = GPROFConvDataset

normalizer = Normalizer.load("sftp://129.16.35.202/mnt/array1/share/MLDatasets/gprof/conv/normalizer_gprof_gmi_conv_bt_only.pckl")
bins = np.logspace(-4, 2.5, 257)
kwargs = {"batch_size": 8,
          "normalizer": normalizer,
          "bins": bins}

training_data = DataFolder(training_path, dataset_factory, kwargs=kwargs, n_workers=5)
path = "sftp://" + host + "/" + validation_path
validation_data = DataFolder(validation_path, dataset_factory, kwargs=kwargs, n_workers=1)

#
# Create model
#

model = XceptionFpn(15, bins.size - 1, n_features=256, blocks=8)
drnn = DRNN(bins, model=model)
#drnn = DRNN.load(model_path / network_name)
model = drnn.model
#for m in model.modules():
#    m.eval()
#model.head.train(False)
#for p in model.parameters():
#    p.requires_grad = False
#for p in model.head.parameters():
#    p.requires_grad = True
#

n_epochs=40
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
drnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-1.0)
drnn.save(model_path / network_name)
n_epochs=40
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
drnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-1.0)
drnn.save(model_path / network_name)
n_epochs=40
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
drnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-1.0)
drnn.save(model_path / network_name)
n_epochs=40
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
drnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
           device=device,
           mask=-1.0)
drnn.save(model_path / network_name)


#
# Store results
#

drnn.save(model_path / network_name)
np.savetxt(results_name, np.stack((training_errors, validation_errors)))
