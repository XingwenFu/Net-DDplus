import torch
import torch.nn as nn
import torch.nn.functional as F

class DDplus_bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, depth=1, dropRate=0.0, stride=1, r=4):
        super(DDplus_bottleneck, self).__init__()
        self.depth = depth
        self.drop = nn.Dropout(dropRate)
        mplans = out_planes // r
        self.DD1 = nn.Conv2d(in_planes, mplans, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mplans)
        self.DD2 = nn.Conv2d(mplans, mplans, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mplans)
        self.DD3 = nn.Conv2d(mplans, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, out_DD, out_B):
        for i in range(self.depth):
            out_DD = self.bn1(self.DD1(out_DD))
            out_DD = self.bn2(self.DD2(out_DD))
            out_DD = self.DD3(out_DD)
            out_DD = self.drop(out_DD)
            out_DD = F.relu(out_DD*out_B)
            out_DD = self.bn3(out_DD)
        return out_DD

class DDplus_basicblock(nn.Module):
    def __init__(self, in_planes, out_planes, depth=1, dropRate=0.0, stride=1):
        super(DDplus_basicblock, self).__init__()
        self.depth = depth
        self.drop = nn.Dropout(dropRate)
        self.bn = nn.BatchNorm2d(out_planes)
        self.DD = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False)

    def forward(self, out_DD, out_B):
        for i in range(self.depth):
            out_DD = self.DD(out_DD)
            out_DD = F.relu(out_DD*out_B)
            out_DD = self.drop(out_DD)
            out_DD = self.bn(out_DD)
        return out_DD