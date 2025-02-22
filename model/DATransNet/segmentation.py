import torch
import torch.nn as nn
from torch.nn import Flatten
import torch.nn.functional as F
from .Gradient_attention.contrast_and_atrous import AttnContrastLayer
from .CDCNs.Gradient_model import ExpansionContrastModule
from .AttentionModule import *
from .AttentionModule import _NonLocalBlockND
from .CDCNs.Global_view import External_attention
from .UIUNet_module.modules import *
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
class GFEM(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=4)
        self.up = nn.Upsample(scale_factor=4,mode='bilinear')
        self.ca = ChannelAttention(in_planes=channels)
        self.sp = _NonLocalBlockND(in_channels=channels,inter_channels=channels//8)
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
        # d = self.sattn(xin)*d
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
    
def _upsample_like(src,tar):

    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src

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
        self.pool = nn.MaxPool2d(2, 2,ceil_mode=True)
        self.inc = RSU7(n_channels,in_channels//2,in_channels)
        self.encoder1 = RSU6(in_channels,in_channels//2,in_channels*2)
        self.encoder2 = RSU5(in_channels*2,in_channels*1,in_channels*4)
        self.encoder3 = RSU4(in_channels*4,in_channels*2,in_channels*8)
        self.encoder4 = RSU4F(in_channels*8,in_channels*4,in_channels*8)
        self.encoder5 = RSU4F(in_channels*8,in_channels*4,in_channels*8)
        # self.encoder6 = self._make_layer(block, in_channels*4 , in_channels *4  ,1)
        self.contras1 = ExpansionContrastModule(in_channels=in_channels*1,out_channels=in_channels*1,width=img_size//1,height=img_size//1,shifts=[1,3])
        self.contras2 = ExpansionContrastModule(in_channels=in_channels*2,out_channels=in_channels*2,width=img_size//2,height=img_size//2,shifts=[1,3])
        self.contras3 = ExpansionContrastModule(in_channels=in_channels*4,out_channels=in_channels*4,width=img_size//4,height=img_size//4,shifts=[1,3])
        self.contras4 = ExpansionContrastModule(in_channels=in_channels*8,out_channels=in_channels*8,width=img_size//4,height=img_size//4,shifts=[1,3])
        self.GFEM = GFEM(channels=in_channels*8)

        self.fuse4 = self._fuse_layer(in_channels*8, in_channels*8, in_channels*8, fuse_mode='AsymBi')
        self.fuse3 = self._fuse_layer(in_channels*4, in_channels*4, in_channels*4, fuse_mode='AsymBi')
        self.fuse2 = self._fuse_layer(in_channels*2, in_channels*2, in_channels*2, fuse_mode='AsymBi')
        # self.fuse2 = self._fuse_layer(in_channels, in_channels, in_channels, fuse_mode='AsymBi')

        # self.stage5d = RSU4F(1024,256,512)
        self.stage4d = RSU4(in_channels*16,in_channels*4,in_channels*4)
        self.stage3d = RSU5(in_channels*8,in_channels*2,in_channels*2)
        self.stage2d = RSU6(in_channels*4,in_channels*1,in_channels)
        self.stage1d = RSU7(in_channels*2,in_channels//2,in_channels)

        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1, 1), stride=(1, 1))
    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for _ in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)
    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels,fuse_mode='AsymBi'):#fuse_mode='AsymBi'
        # assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        # if fuse_mode == 'BiLocal':
        #     fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # el
        if fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        # elif fuse_mode == 'BiGlobal':
        #     fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer
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
        hx5dup = _upsample_like(d5,c4)

        #-------------------- decoder --------------------

        fusec41,fusec42 = self.fuse4(hx5dup, c4)
        hx4d = self.stage4d(torch.cat((fusec41,fusec42),1))
        hx4dup = _upsample_like(hx4d,c3)


        fusec31,fusec32 = self.fuse3(hx4dup, c3)
        hx3d = self.stage3d(torch.cat((fusec31,fusec32),1))
        hx3dup = _upsample_like(hx3d,c2)

        fusec21, fusec22 = self.fuse2(hx3dup, c2)
        hx2d = self.stage2d(torch.cat((fusec21, fusec22), 1))
        hx2dup = _upsample_like(hx2d,c1)


        out = self.stage1d(torch.cat((hx2dup,c1),1))
        out = self.outc(out)
        return out.sigmoid()