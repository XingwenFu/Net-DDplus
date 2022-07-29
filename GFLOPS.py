#pytorch 计算模型的Para和GFLOPs
# 导入所有的网络架构
from model.wideresnet import WideResNet
from model.wideresnet_DDplus import WideResNet_DDplus
from model.resnet import Resnet
from model.resnet_DDplus import Resnet_DDplus
from model.se_resnet import Se_net
from model.se_resnet_DDplus import Se_net_DDplus
from model.densenet3 import DenseNet
from model.densenet3_DDplus import DenseNet_DDplus
from model.shake_shake import Shake_shake
from model.shake_shake_DDplus import Shake_shake_DDplus
from model.pyramid import Pyramid
from model.pyramid_DDplus import Pyramid_DDplus

import torch
from thop import profile

model = Resnet_DDplus(32, 100, 3, 1, 4)
# model = Se_net(32, 100, 3, 1, 4)
# model = WideResNet(40, 100, 12, dropRate=0, nc=3, nd=1)
# model = DenseNet(190, 100, 40, nc=3)
# model = Pyramid('cifar100', 272, 100)
# model = Resnet(56, 100, 3, 1, 4)

input = torch.randn(128, 3, 32, 32) #模型输入的形状,batch_size=1
if torch.cuda.is_available():
    model = model.cuda()
    input = input.cuda()
flops, params = profile(model, inputs=(input, ))
print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops/1e9,params/1e6)) #flops单位G，para单位M