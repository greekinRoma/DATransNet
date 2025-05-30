import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
from .contrast_and_atrous import AttnContrastLayer
class DATrans(nn.Module):
    def __init__(self,in_channels,out_channels,tra_channels,width,height,shifts):
        super().__init__()
        #The hyper parameters settting
        self.convs_list=nn.ModuleList()
        self.in_channels = max(in_channels,1)
        self.out_channels = out_channels
        self.tra_channels = tra_channels
        self.shifts =shifts
        self.num_heads = len(shifts)
        self.num_layer= 8
        self.width = width
        self.height = height
        self.area = width* height
        self.psi = nn.InstanceNorm2d(len(self.shifts))
        self.softmax_layer = nn.Softmax(dim=-1)
        self.query_convs=nn.ModuleList()
        self.key_convs=nn.ModuleList()
        self.value_convs=nn.ModuleList()
        self.sur_weight_layers = nn.ModuleList()
        self.sum_layers = nn.ModuleList()
        self.hidden_channels = self.tra_channels//len(self.shifts)
        for i in range(len(self.shifts)):
            self.query_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False))
            self.key_convs.append(nn.Conv2d(in_channels=self.in_channels*self.num_layer,out_channels=self.hidden_channels*self.num_layer,kernel_size=1,stride=1,bias=False))
            self.value_convs.append(nn.Conv2d(in_channels=self.in_channels*self.num_layer,out_channels=self.hidden_channels*self.num_layer,kernel_size=1,stride=1,bias=False)) 
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.tra_channels,out_channels=self.out_channels,kernel_size=1,stride=1,bias=False),
                                      nn.BatchNorm2d(self.out_channels),
                                      nn.ReLU())
    def feature_padding(self, input_feature, dilation_ratio):

        B, C, H, W = input_feature.size()
        in_feat = input_feature.clone()

        # top pad
        left_top_pad = nn.ReflectionPad2d((dilation_ratio, 0, dilation_ratio, 0))
        x0 = left_top_pad(in_feat)
        x0 = x0[:, :, 0:H, 0:W]

        center_top_pad = nn.ReflectionPad2d((0, 0, dilation_ratio, 0))
        x1 = center_top_pad(in_feat)
        x1 = x1[:, :, 0:H, :]

        right_top_pad = nn.ReflectionPad2d((0, dilation_ratio, dilation_ratio, 0))
        x2 = right_top_pad(in_feat)
        x2 = x2[:, :, 0:H, dilation_ratio:]

        # center pad
        left_center_pad = nn.ReflectionPad2d((dilation_ratio, 0, 0, 0))
        x3 = left_center_pad(in_feat)
        x3 = x3[:, :, :, 0:W]

        right_center_pad = nn.ReflectionPad2d((0, dilation_ratio, 0, 0))
        x4 = right_center_pad(in_feat)
        x4 = x4[:, :, :, dilation_ratio:]

        # bottom pad
        left_bottom_pad = nn.ReflectionPad2d((dilation_ratio, 0, 0, dilation_ratio))
        x5 = left_bottom_pad(in_feat)
        x5 = x5[:, :, dilation_ratio:, 0:W]

        center_bottom_pad = nn.ReflectionPad2d((0, 0, 0, dilation_ratio))
        x6 = center_bottom_pad(in_feat)
        x6 = x6[:, :, dilation_ratio:, :]

        right_bottm_pad = nn.ReflectionPad2d((0, dilation_ratio, 0, dilation_ratio))
        x7 = right_bottm_pad(in_feat)
        x7 = x7[:, :, dilation_ratio:, dilation_ratio:]

        return x0, x1, x2, x3, x4, x5, x6, x7
    def Extract_layer(self,cen,b,w,h):
        surrounds_keys = []
        surrounds_querys = []
        surrounds_values = []
        for i in range(len(self.shifts)):
            x0, x1, x2, x3, x4, x5, x6, x7 = self.feature_padding(cen,self.shifts[i])
            surround1 = x0 - cen
            surround2 = x1 - cen
            surround3 = x2 - cen
            surround4 = x3 - cen
            surround5 = x4 - cen
            surround6 = x5 - cen
            surround7 = x6 - cen
            surround8 = x7 - cen
            surrounds = torch.cat([surround1,surround2,surround3,surround4,surround5,surround6,surround7,surround8],1)
            surrounds_keys.append(self.key_convs[i](surrounds))
            surrounds_querys.append(self.query_convs[i](cen))
            surrounds_values.append(self.value_convs[i](surrounds))
        surrounds_keys = torch.stack(surrounds_keys,dim=2).view(b,self.num_heads,-1,w*h)
        surrounds_querys = torch.stack(surrounds_querys,dim=2).view(b,self.num_heads,-1,w*h)
        surrounds_values = torch.stack(surrounds_values,dim=2).view(b,self.num_heads,-1,w*h)
        return surrounds_keys,surrounds_querys,surrounds_values
    def forward(self,cen):
        b,_,w,h= cen.shape
        deltas_keys,deltas_querys,deltas_values = self.Extract_layer(cen,b,w,h)
        deltas_keys = torch.nn.functional.normalize(deltas_keys,dim=-1).transpose(-2,-1)
        deltas_querys = torch.nn.functional.normalize(deltas_querys,dim=-1)
        weight_score = torch.matmul(deltas_querys,deltas_keys)
        weight_score = self.softmax_layer(self.psi(weight_score/math.sqrt(self.area)))
        out = torch.matmul(weight_score,deltas_values)
        out = out.view(b,self.tra_channels,w,h)
        out = self.out_conv(out)
        return out