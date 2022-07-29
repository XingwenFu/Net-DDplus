import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor, ceil
from model.Net_DDplus import DDplus_basicblock
from model.wideresnet import WideResNet


class Net_DDplus(nn.Module):
    def __init__(self, Feature_Net, width, num_classes, nd):
        super(Net_DDplus, self).__init__()
        self.Feature_Net = Feature_Net
        nChannels = [16, 16*width, 32*width, 64*width, 128*width]
        self.DD1 = DDplus_basicblock(nChannels[0], nChannels[1], depth=1, dropRate=0.0, stride=1)
        self.DD2 = DDplus_basicblock(nChannels[1], nChannels[2], depth=1, dropRate=0.0, stride=2)
        self.DD3 = DDplus_basicblock(nChannels[2], nChannels[3], depth=1, dropRate=0.0, stride=2)
        if nd == 2:
            self.DD4 = DDplus_basicblock(nChannels[3], nChannels[4], depth=1, dropRate=0.0, stride=2)
            # self.linear = nn.Linear(p_in*8, num_classes)
        if nd == 1:
            self.DD4 = lambda out_DD,out: out_DD  # 恒等映射
            # self.linear = nn.Linear(p_in*4, num_classes)

    def forward(self, x):
        x = self.Feature_Net.conv1(x)
        x = self.Feature_Net.bn0(x)
        feature0 = F.relu(x)
        feature1 = self.Feature_Net.block1(feature0)
        feature2 = self.Feature_Net.block2(feature1)
        feature3 = self.Feature_Net.block3(feature2)
        feature4 = self.Feature_Net.block4(feature3)
        out = self.DD4(self.DD3(self.DD2(self.DD1(feature0, feature1), feature2), feature3), feature4)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.Feature_Net.fc(out)
        return out

def WideResNet_DDplus(layers=20, num_classes=10, width=1, dropRate=0.0, nc=3, nd=1, pretrained=False, path=''):
    Feature_Net = WideResNet(depth=layers, num_classes=num_classes, widen_factor=width, dropRate=dropRate, nc=nc, nd=nd)
    if pretrained:
        model_path = path
        print('Loading weights into state dict...')
        state_dict = torch.load(model_path)
        Feature_Net.load_state_dict(state_dict, strict=False)
        print('Finished!')
        print('Turn off weight update of feature network!')
        for param in Feature_Net.parameters():  #冻结特征网络的参数
            param.requires_grad = False
        print('Finished!')
        #print('Turn off weight update of feature network!')
        #for param in Feature_Net.parameters():  #冻结特征网络的参数
            #param.requires_grad = False
    return Net_DDplus(Feature_Net, width=width, num_classes=num_classes, nd=nd)