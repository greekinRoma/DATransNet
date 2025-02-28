from torch import nn
import torch
import torch.nn.functional as F
class TGMandTRM(nn.Module):
    def __init__(self, h,c,rank_num=32, norm_layer=None):
        super(TGMandTRM, self).__init__()
        self.rank = rank_num
        self.ps = [1, 1, 1, 1]
        self.h = h
        conv1_1, conv1_2, conv1_3 = self.ConvGeneration(self.rank, h,c)

        self.conv1_1 = conv1_1
        self.conv1_2 = conv1_2
        self.conv1_3 = conv1_3

        self.lam = torch.nn.Parameter(torch.ones(self.rank, requires_grad=True)).cuda()
        self.out_conv = nn.Conv2d(in_channels=c*2,out_channels=c,kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(self.ps[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')


    def forward(self, x):
        b, c, height, width = x.size()
        C = self.pool(x)
        H = self.pool(x.permute(0, 3, 1, 2).contiguous())
        W = self.pool(x.permute(0, 2, 3, 1).contiguous())
        # self.lam = F.softmax(self.lam,-1)
        lam = torch.chunk(self.lam, dim=0, chunks=self.rank)
        list = []
        for i in range(0, self.rank):
            list.append(lam[i]*self.TukerReconstruction(b, self.h , self.ps[0], self.conv1_1[i](C), self.conv1_2[i](H), self.conv1_3[i](W)))
        tensor1 = sum(list)
        tensor1 = torch.cat((x , F.relu_(x * tensor1)), 1)
        tensor1 = self.out_conv(tensor1)
        return tensor1

    def ConvGeneration(self, rank, h, c):
        conv1 = []
        n = 1
        for _ in range(0, rank):
                conv1.append(nn.Sequential(
                nn.Conv2d(c, c // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv1 = nn.ModuleList(conv1)

        conv2 = []
        for _ in range(0, rank):
                conv2.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv2 = nn.ModuleList(conv2)

        conv3 = []
        for _ in range(0, rank):
                conv3.append(nn.Sequential(
                nn.Conv2d(h, h // n, kernel_size=1, bias=False),
                nn.Sigmoid(),
            ))
        conv3 = nn.ModuleList(conv3)

        return conv1, conv2, conv3

    def TukerReconstruction(self, batch_size, h, ps, feat, feat2, feat3):
        b = batch_size
        C = feat.view(b, -1, ps)
        H = feat2.view(b, ps, -1)
        W = feat3.view(b, ps * ps, -1)
        CHW = torch.bmm(torch.bmm(C, H).view(b, -1, ps * ps), W).view(b, -1, h, h)
        return CHW