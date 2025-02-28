import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import nn
import math
from .contrast_and_atrous import AttnContrastLayer
class ExpansionContrastModule(nn.Module):
    def __init__(self,in_channels,out_channels,width,height,shifts):
        super().__init__()
        #The hyper parameters settting
        self.convs_list=nn.ModuleList()
        delta1=np.array([[[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                         [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
                         [[0, 0, 0], [0, 1, -1], [0, 0, 0]]])
        delta1=delta1.reshape(4,1,3,3)
        delta2=delta1[:,:,::-1,::-1].copy()
        delta=np.concatenate([delta1,delta2],axis=0)
        w1,w2,w3,w4,w5,w6,w7,w8=np.array_split(delta,8)
        self.in_channels = max(in_channels,1)
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
        self.down_convs = nn.ModuleList()
        self.kernel1 = torch.Tensor(w1).cuda()
        self.kernel2 = torch.Tensor(w2).cuda()
        self.kernel3 = torch.Tensor(w3).cuda()
        self.kernel4 = torch.Tensor(w4).cuda()
        self.kernel5 = torch.Tensor(w5).cuda()
        self.kernel6 = torch.Tensor(w6).cuda()
        self.kernel7 = torch.Tensor(w7).cuda()
        self.kernel8 = torch.Tensor(w8).cuda()
        self.kernel1 = self.kernel1.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel2 = self.kernel2.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel3 = self.kernel3.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel4 = self.kernel4.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel5 = self.kernel5.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel6 = self.kernel6.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel7 = self.kernel7.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel8 = self.kernel8.repeat(self.in_channels, 1, 1, 1).contiguous()
        self.kernel0 = (self.kernel1 + self.kernel2 + self.kernel3 + self.kernel4 + self.kernel5 + self.kernel6 + self.kernel7 + self.kernel8)/8
        self.hidden_channels = self.in_channels//len(self.shifts)
        self.sum_weights = nn.ParameterList()
        self.pad_layers = nn.ModuleList()
        for i in range(len(self.shifts)):
            self.query_convs.append(nn.Conv2d(in_channels=self.in_channels,out_channels=self.hidden_channels,kernel_size=1,stride=1,bias=False))
            self.key_convs.append(nn.Conv2d(in_channels=self.in_channels*self.num_layer,out_channels=self.hidden_channels*self.num_layer,kernel_size=1,stride=1,bias=False))
            self.value_convs.append(nn.Conv2d(in_channels=self.in_channels*self.num_layer,out_channels=self.hidden_channels*self.num_layer,kernel_size=1,stride=1,bias=False)) 
            self.sum_weights.append(nn.Parameter(torch.randn((in_channels,1,1,1,2))))
            self.pad_layers.append(nn.ReflectionPad2d((self.shifts[i],self.shifts[i],self.shifts[i],self.shifts[i])))
        self.out_conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels,kernel_size=1,stride=1,bias=False),
                                      nn.BatchNorm2d(self.in_channels),
                                      nn.ReLU())
    def Extract_layer(self,cen,b,w,h):
        surrounds_keys = []
        surrounds_querys = []
        surrounds_values = []
        for i in range(len(self.shifts)):
            sum_weight = torch.nn.functional.softmax(self.sum_weights[i],dim=-1)
            surround1 = torch.nn.functional.conv2d(weight=self.kernel1*sum_weight[...,0]+self.kernel0*sum_weight[...,1], stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround2 = torch.nn.functional.conv2d(weight=self.kernel2*sum_weight[...,0]+self.kernel0*sum_weight[...,1], stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround3 = torch.nn.functional.conv2d(weight=self.kernel3*sum_weight[...,0]+self.kernel0*sum_weight[...,1], stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround4 = torch.nn.functional.conv2d(weight=self.kernel4*sum_weight[...,0]+self.kernel0*sum_weight[...,1], stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround5 = torch.nn.functional.conv2d(weight=self.kernel5*sum_weight[...,0]+self.kernel0*sum_weight[...,1], stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround6 = torch.nn.functional.conv2d(weight=self.kernel6*sum_weight[...,0]+self.kernel0*sum_weight[...,1], stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround7 = torch.nn.functional.conv2d(weight=self.kernel7*sum_weight[...,0]+self.kernel0*sum_weight[...,1], stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
            surround8 = torch.nn.functional.conv2d(weight=self.kernel8*sum_weight[...,0]+self.kernel0*sum_weight[...,1], stride=1, padding="same", input=cen,groups=self.in_channels,dilation=self.shifts[i])
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
        weight_score = self.softmax_layer(self.psi(torch.matmul(deltas_querys,deltas_keys)/math.sqrt(self.area)))
        out = torch.matmul(weight_score,deltas_values)
        out = out.view(b,self.in_channels,w,h)
        out = self.out_conv(out)
        return out