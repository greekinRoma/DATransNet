from torch import nn

import os
from loss import SoftIoULoss, ISNetLoss
from model import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, model_name, mode='test',size=256):
        super(Net, self).__init__()

        self.model_name = model_name
        self.cal_loss = SoftIoULoss()

        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'AGPCNet':
            self.model = AGPCNet()
        # elif model_name == 'ISNet':
        #     if mode == 'train':
        #         self.model = ISNet(mode='train')
        #     else:
        #         self.model = ISNet(mode='test')
        #     self.cal_loss = ISNetLoss()
        # elif model_name == 'RISTDnet':
        #     self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name == 'ISTDU_Net':
            self.model = ISTDU_Net()
        elif model_name == 'DATransNet':
            self.model = DATransNet(img_size=size)
        elif model_name == 'res_UNet':
            self.model = res_UNet()
    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
