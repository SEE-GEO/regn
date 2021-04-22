import torch
from torch import nn
from torch.nn.functional import softplus
from regn.data.csu.bin import PROFILE_NAMES

class QuantileHead(nn.Module):
    def __init__(self, n_inputs, n_quantiles):
        self.upper = nn.Linear(n_inputs, n_quantiles // 2)
        self.median = nn.Linear(n_inputs, 1)
        self.lower = nn.Linear(n_inputs, n_quantiles // 2)

    def forward(self, x):
        m = self.median(x)
        upper = m + torch.cumsum(softplus(self.upper(x)), 1)
        lower = m - torch.cumsum(softplus(self.lower(x)), 1)
        if self.odd:
            return torch.cat([lower, m, upper], 1)
        return torch.cat([lower, upper], 1)


class GPROFNN0D(nn.Module):
    """
    Pytorch neural network model for the GPROF 0D retrieval.

    The model is a simple fully-connected model with multiple heads for each
    of the retrieval targets.

    Attributes:
         n_layers: The total number of layers in the network.
         n_neurons: The number of neurons in the hidden layers of the network.
         n_quantiles: How many quantiles to predict for each retrieval target.
         target: Single string or dictionary containing the retrieval targets
               to predict.
    """
    def __init__(self,
                 n_layers,
                 n_neurons,
                 n_quantiles,
                 target="surface_precip",
                 exp_activation=False,
                 quantile_heads=False):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.n_quantiles = n_quantiles
        self.target = target
        self.exp_activation = exp_activation
        self.quantile_heads = quantiles_heads

        super().__init__()
        self.layers = nn.Sequential(*(
            [nn.Linear(40, n_neurons), nn.ReLU()] +
            [nn.Linear(n_neurons, n_neurons), nn.ReLU()] * (n_layers - 2)
            )
        )

        if isinstance(self.target, list):
            self.heads = {}
            for k in self.target:
                if k in PROFILE_NAMES:
                    n_outputs = 28 * n_quantiles
                else:
                    n_outputs = n_quantiles
                if self.quantile_heads:
                    l = QuantileHead(n_neurons, n_outputs)
                else:
                    l = nn.Linear(n_neurons, n_outputs)
                setattr(self, "head_" + k, l)
                self.heads[k] = l
        else:
            self.head = nn.Linear(n_neurons, n_quantiles)

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
        y = self.layers(x)
        if isinstance(self.target, list):
            results = {}
            for k in self.target:
                if self.exp_activation:
                    results[k] = torch.exp(self.heads[k](y))
                else:
                    results[k] = self.heads[k](y)

                if k in PROFILE_NAMES:
                    shape = (-1, self.n_quantiles, 28)
                    if self.exp_activation:
                        results[k] = torch.exp(results[k].reshape(shape))
                    else:
                        results[k] = results[k].reshape(shape)
            return results
        else:
            if self.exp_activation:
                return torch.exp(self.head(y))
            else:
                return self.head(y)
