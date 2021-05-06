"""
regn.models
===========



"""
import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus
from regn.data.csu.bin import PROFILE_NAMES

BINS = {
    "surface_precip": np.logspace(-5.0, 2.5, 129),
    "convective_precip": np.logspace(-5.0, 2.5, 129),
    "rain_water_path": np.logspace(-5.0, 2.5, 129),
    "ice_water_path": np.logspace(-5.0, 2.5, 129),
    "cloud_water_path": np.logspace(-5.0, 2.5, 129),
    "total_column_water_vapor": np.logspace(-4.5, 2.5, 129),
    "cloud_water_content": np.logspace(-6.0, 1.5, 129),
    "snow_water_content": np.logspace(-6.0, 1.5, 129),
    "rain_water_content": np.logspace(-6.0, 1.5, 129),
    "latent_heat": np.linspace(-100, 300, 129)
}

for k in BINS:
    if k != "latent_heat":
        BINS[k][0] == 0.0

QUANTILES = np.linspace(1e-3, 1.0 - 1e-3, 128)

class ClampedExp(nn.Module):
    """
    Clamped version of the exponential function that avoids exploding values.
    """
    def forward(self, x):
        return torch.exp(torch.clamp(x, -20, 20))

class ResNetFC(nn.Module):
    """
    Fully-connected residual network used as building
    block in the GPROF-NN-0D algorithm.
    """
    def __init__(self,
                 n_inputs,
                 n_neurons,
                 n_outputs,
                 n_layers,
                 output_activation=None,
                 internal=False):
        """
        Create network object.

        Args:
            n_inputs: Number of features in the input vector.
            n_outputs: Number of output values.
            n_neurons: Number of neurons in hidden layers.
            n_layers: Number of layers including the output layer.
            output_activation: Layer class to use as output activation.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.internal = internal
        for i in range(n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(n_inputs, n_neurons, bias=False),
                    nn.LayerNorm(n_neurons),
                    nn.GELU()
                )
            )
            n_inputs = n_neurons
        self.output_layer = nn.Linear(n_inputs, n_outputs, bias=False)
        if output_activation is not None:
            self.output_activation = nn.Sequential(
                    nn.LayerNorm(n_outputs),
                    output_activation()
                    )
        else:
            self.output_activation = None

    def forward(self, x, acc, li=1):
        """
        Forward input through network.
        """
        for l in self.layers:
            y = l(x)
            y[:, :x.shape[1]] += x
            if li > 1:
                y[:, :acc.shape[1]] += (1.0 / (li - 1)) * acc
            acc[:, :x.shape[1]] += x
            li += 1
            x = y
        y = self.output_layer(x)
        if self.internal:
            y[:, :x.shape[1]] += x
            y[:, :acc.shape[1]] += (1.0 / (li - 1)) * acc
        if self.output_activation:
            return self.output_activation(y), acc
        return y, acc


class GPROFNN0D(nn.Module):
    """
    Pytorch neural network model for the GPROF 0D retrieval.

    The model is a fully-connected residual network model with multiple heads for each
    of the retrieval targets.

    Attributes:
         n_layers: The total number of layers in the network.
         n_neurons: The number of neurons in the hidden layers of the network.
         n_outputs: How many quantiles to predict for each retrieval target.
         target: Single string or list containing the retrieval targets
               to predict.
    """
    def __init__(self,
                 n_layers_body,
                 n_layers_head,
                 n_neurons,
                 n_outputs,
                 target="surface_precip",
                 exp_activation=False):
        self.target = target
        self.profile_shape = (-1, n_outputs, 28)
        self.n_layers_body = n_layers_body
        self.n_neurons = n_neurons

        super().__init__()
        self.body = ResNetFC(
                40,
                n_neurons,
                n_neurons,
                n_layers_body,
                nn.GELU,
                internal=True)
        self.heads = nn.ModuleDict()
        if exp_activation:
            activation = ClampedExp
        else:
            activation = None

        if not isinstance(self.target, list):
            targets = [self.target]
        else:
            targets = self.target

        for t in targets:
            if t in PROFILE_NAMES:
                self.heads[t] = ResNetFC(n_neurons,
                                         n_neurons,
                                         28 * n_outputs,
                                         n_layers_head,
                                         output_activation=activation)
            else:
                self.heads[t] = ResNetFC(n_neurons,
                                         n_neurons,
                                         n_outputs,
                                         n_layers_head,
                                         output_activation=activation)

    def forward(self, x):
        """
        Forward the input x through the network.

        Args:
             x: Rank-2 tensor with the 40 input elements along the
                 second dimension and the batch sample along the first.

        Return:
            In the case of a single-target network a single tensor. In
            the case of a multi-target network a dictionary of tensors.
        """
        if not isinstance(self.target, list):
            targets = [self.target]
        else:
            targets = self.target

        acc = torch.zeros(x.shape[0], self.n_neurons, device=x.device, dtype=x.dtype)
        y, acc = self.body(x, acc)
        results = {}
        for k in targets:
            results[k], _ = self.heads[k](y, acc, self.n_layers_body + 1)
            if k in PROFILE_NAMES:
                results[k] = results[k].reshape(self.profile_shape)
        if not isinstance(self.target, list):
            return results[self.traget]
        return results
