# Code modified from https://github.com/kuangliu/pytorch-cifar

'''ResNet in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np

from model.Net_DDplus import DDplus_basicblock
from model.resnet import ResNet, Bottleneck, BasicBlock


class Net_DDplus(nn.Module):
    def __init__(self, Feature_Net, width, num_classes, nd):
        super(Net_DDplus, self).__init__()
        self.Feature_Net = Feature_Net
        p_in = 16*width
        self.DD1 = DDplus_basicblock(p_in, p_in, depth=1, dropRate=0.0, stride=1)
        self.DD2 = DDplus_basicblock(p_in, p_in*2, depth=1, dropRate=0.0, stride=2)
        self.DD3 = DDplus_basicblock(p_in*2, p_in*4, depth=1, dropRate=0.0, stride=2)
        if nd == 2:
            self.DD4 = DDplus_basicblock(p_in*4, p_in*8, depth=1, dropRate=0.0, stride=2)
            # self.linear = nn.Linear(p_in*8, num_classes)
        if nd == 1:
            self.DD4 = lambda out_DD,out: out_DD  # 恒等映射
            # self.linear = nn.Linear(p_in*4, num_classes)

    def forward(self, x):
        x = self.Feature_Net.conv1(x)
        x = self.Feature_Net.bn1(x)
        feature0 = F.relu(x)
        feature1 = self.Feature_Net.layer1(feature0)
        feature2 = self.Feature_Net.layer2(feature1)
        feature3 = self.Feature_Net.layer3(feature2)
        feature4 = self.Feature_Net.layer4(feature3)
        out = self.DD4(self.DD3(self.DD2(self.DD1(feature0, feature1), feature2), feature3), feature4)
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.Feature_Net.linear(out)
        return out

def Resnet_DDplus(layers=20, num_classes=10, width=1, nc=3, nd=1, pretrained=False, path=''):
    if layers == 18 or layers == 34:
        if layers==18:
            block_num = [2, 2, 2, 2]
        if layers==34:
            block_num = [3, 4, 6, 3]
        Feature_Net = ResNet(BasicBlock, block_num, num_classes=num_classes, nc=nc, nd=nd, width=width)
    if layers == 50 or layers == 101 or layers == 152:
        if layers==50:
            block_num = [3, 4, 6, 3]
        if layers==101:
            block_num = [3, 4, 23, 3]
        if layers==152:
            block_num = [3, 8, 36, 3]
        Feature_Net = ResNet(Bottleneck, block_num, num_classes=num_classes, nc=nc, nd=nd, width=width)
    if (layers-2)%6 == 0:
        b_n= (layers-2)//6
        block_num = [b_n, b_n, b_n, b_n]
        Feature_Net = ResNet(BasicBlock, block_num, num_classes=num_classes, nc=nc, nd=nd, width=width)

    if pretrained:
        model_path = path
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = Feature_Net.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        Feature_Net.load_state_dict(model_dict)
        print('Finished!')
        #print('Turn off weight update of feature network!')
        #for param in Feature_Net.parameters():  #冻结特征网络的参数
            #param.requires_grad = False
    return Net_DDplus(Feature_Net, width=width, num_classes=num_classes, nd=nd)

