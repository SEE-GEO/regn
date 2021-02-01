import torch
from torch import nn

class Block(nn.Sequential):
    def __init__(self,
                 in_features,
                 out_features,
                 activation,
                 skip_connections=False,
                 batch_norm=True):
        self.skip_connections = skip_connections
        if self.skip_connections:
            in_features = 2 * in_features
        modules = [nn.Linear(in_features, out_features)]
        if batch_norm:
            modules.append(nn.BatchNorm1d(out_features))
        modules.append(activation())
        super().__init__(*modules)

    def forward(self, x):
        n = x.shape[-1] // 2
        y = nn.Sequential.forward(self, x)
        if self.skip_connections:
            y = torch.cat((y, x[:, n:]), -1)
        return y

class FullyConnected(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 layers,
                 width,
                 activation=nn.ReLU,
                 log=False,
                 skip_connections=False,
                 batch_norm=True):

        self.log = log
        self.skip_connections = skip_connections
        nn.Module.__init__(self)
        self.mods = nn.ModuleList([Block(input_features,
                                         width - input_features,
                                         activation,
                                         skip_connections=False,
                                         batch_norm=batch_norm)])
        for i in range(layers):
            self.mods.append(Block(width,
                                   width,
                                   activation,
                                   skip_connections=skip_connections,
                                   batch_norm=batch_norm))
        if skip_connections:
            self.mods.append(nn.Linear(2 * width, output_features))
        else:
            self.mods.append(nn.Linear(width, output_features))

    def forward(self, x):
        l0 = self.mods[0]
        y = l0.forward(x)
        if self.skip_connections:
            y = torch.cat((y, y), -1)
        for l in self.mods[1:]:
            y = l.forward(y)
        if self.log:
            y = torch.exp(y)
        return y

class FullyConnectedWithSkips(nn.Module):
    def __init__(self,
                 input_features,
                 output_features,
                 layers,
                 width,
                 activation=nn.ReLU,
                 log=False,
                 batch_norm=True):
        nn.Module.__init__(self)

        n_out = (width - input_features) // 2
        n_in = 2 * n_out + input_features

        self.mods = nn.ModuleList([Block(input_features,
                                         n_out,
                                         activation,
                                         skip_connections=False,
                                         batch_norm=batch_norm)])
        self.mods.append(Block(n_out + input_features,
                               n_out,
                               activation,
                               skip_connections=False,
                               batch_norm=batch_norm))
        for i in range(layers - 1):
            self.mods.append(Block(n_in,
                                   n_out,
                                   activation,
                                   skip_connections=False,
                                   batch_norm=batch_norm))

        self.mods.append(nn.Linear(n_in,
                                   output_features))

    def forward(self, x):


        l = self.mods[0]
        ly = l(x)
        y = torch.cat([ly, x], -1)

        ly_p = ly


        for l in self.mods[1:-1]:
            ly = l(y)
            y = torch.cat([ly, ly_p, x], -1)

        y = self.mods[-1](y)
        return y
