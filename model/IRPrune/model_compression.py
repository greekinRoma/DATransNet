import torch
from nni.compression.pytorch.utils.counter import count_flops_params
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = torch.load('E:\研究生作业\汇报/acm-pytorch-main/result/2022-05-06-15-27-24_UNet_BiGlobal\checkpoint\Epoch-195_IoU-0.7263_nIoU-0.7138.pkl')

input = torch.randn(8, 3, 512, 512).to(device)

flops, params = profile(net, inputs=(input, ))
print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
print("params=", str(params/1e6)+'{}'.format("M"))
""""
flops = FlopCountAnalysis(net, input)
print("FLOPs: ", flops.total())
# 分析parameters
#print(parameter_count_table(model))
"""""

net_pretrained = torch.load('E:\研一下\模型压缩\红外目标检测/acm-pytorch-pruning/result/2023-03-05-17-26-01_UNet_BiGlobal\checkpoint\Epoch-280_IoU-0.6764_nIoU-0.7144.pkl')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.randn(8, 3, 512, 512).to(device)

flops, params, results = count_flops_params( net_pretrained, input)
print(f"Model FLOPs {flops/1e6:.2f}M, Params {params/1e6:.2f}M")