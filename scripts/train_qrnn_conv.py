import argparse
from pathlib import Path

import numpy as np
from regn.data.csu.training_data import GPROFConvDataset
from regn.models.unet import UNet
from quantnn import QRNN
from quantnn.data import DataFolder
from quantnn.normalizer import Normalizer
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

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
<<<<<<< HEAD
parser.add_argument('--device', metavar='device', type=str, default="cuda")
=======
parser.add_argument('--device', metavar='device', type=str, default="cuda", nargs=1)
>>>>>>> 649864b4495ae4c64b1fac6263d9348fa1aab54f

args = parser.parse_args()
training_path = args.training_data_path[0]
validation_path = args.validation_data_path[0]
model_path = Path(args.model_path[0])
device = args.device[0]

model_path.mkdir(parents=False, exist_ok=True)

network_name = f"uqrnn_bts_only.pt"
results_name = f"uqrnn_bts_only.dat"

#
# Load the data.
#

host = "129.16.35.202"
dataset_factory = GPROFConvDataset

normalizer = Normalizer.load("sftp://129.16.35.202/mnt/array1/share/MLDatasets/gprof/conv/normalizer_gprof_gmi_conv_bt_only.pckl")
kwargs = {"batch_size": 1,
          "normalizer": normalizer}

training_data = DataFolder(training_path, dataset_factory, kwargs=kwargs, n_workers=5)
path = "sftp://" + host + "/" + validation_path
validation_data = DataFolder(validation_path, dataset_factory, kwargs=kwargs, n_workers=1)

#
# Create model
#

quantiles = np.linspace(0.01, 0.99, 99)
model = UNet(15, quantiles.size)
qrnn = QRNN(quantiles, model=model)

n_epochs=5
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
<<<<<<< HEAD
           device=device)
=======
           device=device,
           mask=-1.0)
>>>>>>> 649864b4495ae4c64b1fac6263d9348fa1aab54f
qrnn.save(model_path / network_name)

n_epochs=10
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
<<<<<<< HEAD
           device=device)
=======
           device=device,
           mask=-1.0)
>>>>>>> 649864b4495ae4c64b1fac6263d9348fa1aab54f
qrnn.save(model_path / network_name)

n_epochs=20
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
<<<<<<< HEAD
           device=device)
=======
           device=device,
           mask=-1.0)
>>>>>>> 649864b4495ae4c64b1fac6263d9348fa1aab54f
qrnn.save(model_path / network_name)
n_epochs=20
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
qrnn.train(training_data=training_data,
           validation_data=validation_data,
           n_epochs=n_epochs,
           optimizer=optimizer,
           scheduler=scheduler,
<<<<<<< HEAD
           device=device)
=======
           device=device,
           mask=-1.0)
>>>>>>> 649864b4495ae4c64b1fac6263d9348fa1aab54f
qrnn.save(model_path / network_name)


#
# Store results
#

qrnn.save(model_path / network_name)
np.savetxt(results_name, np.stack((training_errors, validation_errors)))
