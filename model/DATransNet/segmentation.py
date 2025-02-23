import torch
import torch.nn as nn
from torch.nn import Flatten
import torch.nn.functional as F
from .Gradient_attention.contrast_and_atrous import AttnContrastLayer
from .CDCNs.Gradient_model import ExpansionContrastModule
from .AttentionModule import *
from .AttentionModule import _NonLocalBlockND
from .Non_local.TGMandTRM import TGMandTRM
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
class GFEM(nn.Module):
    def __init__(self,channels,size):
        super().__init__()
        self.down = nn.MaxPool2d((2,2))
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.ca = ChannelAttention(in_planes=channels)
        self.sp = TGMandTRM(h=size,c=channels)
        self.tra_conv_1 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=1)
        self.tra_conv_2 = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1)
        self.out_conv = nn.Conv2d(in_channels=channels,out_channels=channels,kernel_size=3,stride=1,padding=1)
    def forward(self,inps):
        spat = self.sp(inps)
        down = self.down(inps)
        down = self.ca(spat)*down
        down = self.up(down)
        spat = self.tra_conv_1(spat)
        down = self.tra_conv_2(down)
        out = spat + down
        out = self.out_conv(out)
        return out
        
        
        
class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU',kernel_size=3):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding='same')
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)
def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)
class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear')
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        self.sattn = nn.Sequential(nn.Conv2d(in_channels//2,in_channels//2,kernel_size=1),
                                   nn.Sigmoid())
    def forward(self,d,c,xin):
        d = self.up(d)
        d = self.sattn(xin)*d
        x = torch.cat([c, d], dim=1)
        x = self.nConvs(x)
        return x
class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class DATransNet(nn.Module):
    def __init__(self,  n_channels=1, n_classes=1, img_size=256, vis=False, mode='train', deepsuper=True):
        super().__init__()
        self.vis = vis
        self.deepsuper = deepsuper
        self.mode = mode
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = 16  # basic channel 64
        block = Res_block
        self.pool = nn.MaxPool2d(2, 2)
        self.inc = self._make_layer(block, n_channels, in_channels)
        self.encoder1 = self._make_layer(block, in_channels, in_channels * 2, 1)  
        self.encoder2 = self._make_layer(block, in_channels * 2, in_channels * 4, 1) 
        self.encoder3 = self._make_layer(block, in_channels * 4, in_channels * 8, 1)  
        self.encoder4 = self._make_layer(block, in_channels * 8,  in_channels * 8, 1)  
        self.encoder5 = self._make_layer(block, in_channels*8 , in_channels *8  ,1)
        # self.encoder6 = self._make_layer(block, in_channels*4 , in_channels *4  ,1)
        self.contras1 = ExpansionContrastModule(in_channels=in_channels*1,out_channels=in_channels*1,tra_channels=in_channels*1,width=img_size//1,height=img_size//1,shifts=[1,3])
        self.contras2 = ExpansionContrastModule(in_channels=in_channels*2,out_channels=in_channels*2,tra_channels=in_channels*2,width=img_size//2,height=img_size//2,shifts=[1,3])
        self.contras3 = ExpansionContrastModule(in_channels=in_channels*4,out_channels=in_channels*4,tra_channels=in_channels*2,width=img_size//4,height=img_size//4,shifts=[1,3])
        self.contras4 = ExpansionContrastModule(in_channels=in_channels*8,out_channels=in_channels*8,tra_channels=in_channels*2,width=img_size//8,height=img_size//8,shifts=[1])
        self.GFEM = TGMandTRM(h=img_size//16,c=in_channels*8,rank_num=32)
        # self.decoder6 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels*4,out_channels=in_channels*4,kernel_size=2,stride=2),CBN(in_channels*4,in_channels*4,kernel_size=1))
        self.decoder5 = UpBlock_attention(in_channels * 16, in_channels * 8, nb_Conv=2)
        self.decoder4 = UpBlock_attention(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.decoder3 = UpBlock_attention(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.decoder2 = UpBlock_attention(in_channels * 4, in_channels, nb_Conv=2)
        self.decoder1 = UpBlock_attention(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #encoder
        x1 = self.inc(x) 
        x2 = self.encoder1(self.pool(x1)) 
        x3 = self.encoder2(self.pool(x2))  
        x4 = self.encoder3(self.pool(x3))  
        d5 = self.encoder4(self.pool(x4))  
        # Transfor_layer
        c1 = self.contras1(x1)
        c2 = self.contras2(x2)
        c3 = self.contras3(x3)
        c4 = self.contras4(x4)
        d5 = self.GFEM(d5)
        # decoder
        d4 = self.decoder4(d5, c4, x4)
        d3 = self.decoder3(d4, c3, x3)
        d2 = self.decoder2(d3, c2, x2)
        out = self.outc(self.decoder1(d2, c1, x1))
        return out.sigmoid()