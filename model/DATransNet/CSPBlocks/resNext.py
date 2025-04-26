import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from functools import partial
class PartialConv2d(nn.Module):
    r""" 
    Conduct convolution on partial channels.
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
                 conv_ratio=1.0,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, **kwargs,
                 ):
        super().__init__()
        in_chs = int(in_channels * conv_ratio)
        out_chs = int(out_channels * conv_ratio)
        gps = int(groups * conv_ratio) or 1 # groups should be at least 1
        self.conv = nn.Conv2d(in_chs, out_chs, 
                              kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, 
                              groups=gps, bias=bias,
                              **kwargs,
                              )
        self.split_indices = (in_channels - in_chs, in_chs)

    def forward(self, x):
        identity, conv = torch.split(x, self.split_indices, dim=1)
        return torch.cat(
            (identity, self.conv(conv)),
            dim=1,
        )
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ResNextBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                drop_path=0., layer_scale_init_value=1e-6,
                conv_fn=nn.Conv2d,
                ):
        super().__init__()
        dim = out_channels
        self.dwconv = conv_fn(in_channels, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=in_channels) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.dwconv(x)
        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x
    
