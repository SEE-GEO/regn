import torch
from torch import nn

class Block(nn.Sequential):
    def __init__(self,
                 in_features,
                 out_features,
                 activation,
                 skip_connections=True,
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
                 skip_connections=True,
                 batch_norm=True):

        self.log = log
        self.skip_connections = skip_connections
        nn.Module.__init__(self)
        self.mods = nn.ModuleList([Block(input_features,
                                         width,
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
