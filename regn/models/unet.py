import torch
import torch.nn as nn

def _conv2(channels_in, channels_out, kernel_size):
    return nn.Conv2d(channels_in,
                     channels_out,
                     kernel_size=kernel_size,
                     padding=kernel_size // 2,
                     padding_mode="reflect")


class ConvolutionBlock(nn.Module):
    """
    A convolution block consisting of a pair of 2x2
    convolutions followed by batch norm and ReLU activaitons.
    """
    def __init__(self,
                 channels_in,
                 channels_out):
        """
        Create new convolution block.

        Args:
            channels_in: The number of input channels.
            channels_out: The number of output channels.
        """
        super().__init__()
        self.block = nn.Sequential(
            _conv2(channels_in, channels_out, 3),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            _conv2(channels_out, channels_out, 3),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownSamplingBlock(nn.Module):
    """
    UNet downsampling block consisting of 2x2 max-pooling followed
    by a convolution block.
    """
    def __init__(self,
                 channels_in,
                 channels_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvolutionBlock(channels_in, channels_out)
        )

    def forward(self, x):
        return self.block(x)


class UpSamplingBlock(nn.Module):
    """
    UNet downsampling block consisting of 2x2 max-pooling followed
    by a convolution block.
    """
    def __init__(self, channels_in, channels_out, bilinear=True):
        super().__init__()
        self.upscaling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.reduce = _conv2(channels_in, channels_in // 2, 3)
        self.conv = ConvolutionBlock(channels_in, channels_out)


    def forward(self, x, x_skip):
        x = self.reduce(self.upscaling(x))
        x = torch.cat([x, x_skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs):

        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.in_block = ConvolutionBlock(n_inputs, 64)

        self.down_block_1 = DownSamplingBlock(64, 128)
        self.down_block_2 = DownSamplingBlock(128, 256)
        self.down_block_3 = DownSamplingBlock(256, 512)
        self.down_block_4 = DownSamplingBlock(512, 1024)

        self.up_block_1 = UpSamplingBlock(1024, 512)
        self.up_block_2 = UpSamplingBlock(512, 256)
        self.up_block_3 = UpSamplingBlock(256, 128)
        self.up_block_4 = UpSamplingBlock(128, n_outputs)

        self.out_block = _conv2(n_outputs, n_outputs, 1)


    def forward(self, x):

        d_64 = self.in_block(x)
        d_128 = self.down_block_1(d_64)
        d_256 = self.down_block_2(d_128)
        d_512 = self.down_block_3(d_256)
        d_1024 = self.down_block_4(d_512)

        u_512 = self.up_block_1(d_1024, d_512)
        u_256 = self.up_block_2(u_512, d_256)
        u_128 = self.up_block_3(u_256, d_128)
        u_out = self.up_block_4(u_128, d_64)

        return self.out_block(u_out)
