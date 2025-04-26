import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

ACTIVATIONS = {
    'mish': Mish(),
    'linear': nn.Identity()
}

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, activation='mish'):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            ACTIVATIONS[activation]
        )

    def forward(self, x):
        return self.conv(x)

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, residual_activation='linear'):
        super(CSPBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        self.block = nn.Sequential(
            Conv(in_channels, hidden_channels, 1),
            Conv(hidden_channels, out_channels, 3)
        )

        self.activation = ACTIVATIONS[residual_activation]

    def forward(self, x):
        return self.activation(x+self.block(x))

class CSPFirstStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPFirstStage, self).__init__()

        self.downsample_conv = Conv(in_channels, out_channels, 1, stride=1)

        self.split_conv0 = Conv(out_channels, out_channels, 1)
        self.split_conv1 = Conv(out_channels, out_channels, 1)

        self.blocks_conv = nn.Sequential(
            CSPBlock(out_channels, out_channels, in_channels),
            Conv(out_channels, out_channels, 1)
        )

        self.concat_conv = Conv(out_channels*2, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)

        return x

class CSPStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super(CSPStage, self).__init__()

        self.downsample_conv = Conv(in_channels, out_channels, 3, stride=2)

        self.split_conv0 = Conv(out_channels, out_channels//2, 1)
        self.split_conv1 = Conv(out_channels, out_channels//2, 1)

        self.blocks_conv = nn.Sequential(
            *[CSPBlock(out_channels//2, out_channels//2) for _ in range(num_blocks)],
            Conv(out_channels//2, out_channels//2, 1)
        )

        self.concat_conv = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)

        x1 = self.blocks_conv(x1)

        x = torch.cat([x0, x1], dim=1)
        x = self.concat_conv(x)
        return x
    
class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride=1, cardinality=2, base_width=16, widen_factor=2):
        """ Constructor

        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / (widen_factor * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)