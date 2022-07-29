# Code modified from https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/networks/pyramidnet.py

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from model.Net_DDplus import DDplus_bottleneck
from model.pyramid import aa_PyramidNet


class Net_DDplus(nn.Module):
    def __init__(self, Feature_Net, width, num_classes, nd, blk):
        super(Net_DDplus, self).__init__()
        self.Feature_Net = Feature_Net
        if blk == 1:
            o_r = 4
        else:
            o_r = 1

        channal_add = width//3
        nChannels = [16, (16+1*channal_add)*o_r, (16+2*channal_add)*o_r, (16+3*channal_add)*o_r, (16+4*channal_add)*o_r]

        self.DD1 = DDplus_bottleneck(nChannels[0], nChannels[1], depth=1, dropRate=0.0, stride=1, r=16)
        self.DD2 = DDplus_bottleneck(nChannels[1], nChannels[2], depth=1, dropRate=0.0, stride=2, r=16)
        self.DD3 = DDplus_bottleneck(nChannels[2], nChannels[3], depth=1, dropRate=0.0, stride=2, r=16)
        self.bn_final = nn.BatchNorm2d(nChannels[3])
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(nChannels[3], num_classes)
        if nd == 2:
            self.CDD4 = DDplus_bottleneck(nChannels[3], nChannels[4], depth=1, dropRate=0.0, stride=2, r=16)
            #self.linear = nn.Linear(nChannels[4], num_classes)
        if nd == 1:
            self.CDD4 = lambda out_CDD,out: out_CDD  # 恒等映射
            #self.linear = nn.Linear(p_in*4, num_classes)

    def forward(self, x):
        x = self.Feature_Net.conv1(x)
        x = self.Feature_Net.bn1(x)
        feature0 = F.relu(x)
        feature1 = self.Feature_Net.layer1(feature0)
        feature2 = self.Feature_Net.layer2(feature1)
        feature3 = self.Feature_Net.layer3(feature2)
        
        x = self.DD3(self.DD2(self.DD1(feature0, feature1), feature2), feature3)
        x = self.bn_final(feature3)
        x = self.relu_final(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    
def Pyramid_DDplus(dataset, layers, widen_factor, num_classes, pretrained, nd, blk, path):
    widen_factor = widen_factor//3*3
    if blk == 1:
        Feature_Net = aa_PyramidNet(dataset=dataset, depth=layers, alpha=widen_factor, num_classes=num_classes, bottleneck=True)
    else:
        Feature_Net = aa_PyramidNet(dataset=dataset, depth=layers, alpha=widen_factor, num_classes=num_classes, bottleneck=False)
    if pretrained:
        model_path = path
        print('Loading weights into state dict...')
        state_dict = torch.load(model_path)
        Feature_Net.load_state_dict(state_dict, strict=False)
        print('Finished!')
        print('Turn off weight update of feature network!')
        for param in Feature_Net.parameters():  #冻结特征网络的参数
            param.requires_grad = False
    return Net_DDplus(Feature_Net, width=widen_factor, num_classes=num_classes, nd=nd, blk=blk)
