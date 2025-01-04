import torch
from torch import nn
from torch.nn.parameter import Parameter
 
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(nn.Conv2d(channel, channel*2, kernel_size=1, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(channel*2,channel,kernel_size=1,stride=1,bias=False))
        self.sigmoid = nn.Sigmoid()
        self.out_conv = nn.Conv2d(channel,channel,kernel_size=1,bias=False)
 
    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
 
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
 
        # Two different branches of ECA module
        y = self.conv(y)
 
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return y*x + self.out_conv(x)
class eca_layer_fuse(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer_fuse, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.fc1 = nn.Conv2d(in_channels=channel, out_channels=channel//4, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=channel//4, out_channels=channel, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,low,high):
        y = self.avg_pool(high)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        global_attention = self.fc1(self.max_pool(low))
        global_attention = self.fc2(global_attention)
        y =self.sigmoid(global_attention+y)
        return  y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc1(self.avg_pool(x))
        # max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out 
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7,channel=2):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.weight = nn.Parameter(torch.zeros([1,channel,1,1]))
        self.inp_attn = ChannelAttention(in_planes=channel)
    def forward(self, xin):
        xin = self.inp_attn(xin)
        avg_out = torch.mean(xin, dim=1, keepdim=True)
        max_out, _ = torch.max(xin, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)*xin