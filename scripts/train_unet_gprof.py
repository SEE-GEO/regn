import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from regn.data import GpmData
from typhon.retrieval.qrnn import QRNN
from typhon.retrieval.qrnn.models.pytorch import UNet

################################################################################
# Parse arguments
################################################################################

import argparse

parser = argparse.ArgumentParser(description='Train unet.')
parser.add_argument("training_data", type=str, nargs=1, help="The training data.")
parser.add_argument("levels", type=int, nargs=1, help="Number of downscaling blocks.")
parser.add_argument("n_features", type=int, nargs=1, help="Number of features in network.")

args = parser.parse_args()
training_data = args.training_data[0]
level = args.levels[0]
n_features = args.n_features[0]

################################################################################
# Train network
################################################################################

data = GpmData(training_data)
n = len(data)
training_data, validation_data = torch.utils.data.random_split(data, [int(0.9 * n), n - int(0.9 * n)])
training_loader = DataLoader(training_data, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True)

skip_connection = "all"
quantiles = np.array([0.01, 0.05, 0.15, 0.25, 0.35, 0.45, 0.5,
                      0.55, 0.65, 0.75, 0.85, 0.95, 0.99])
unet = UNet(13, n_features=128, quantiles, skip_connection=skip_connection)
qrnn = QRNN(13, model=unet)

qrnn.train(training_loader, gpu = True, lr = 1e-2,  momentum=0.99)
qrnn.save("unet.pt")
