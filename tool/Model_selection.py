
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
def Model_selection(model, layers, dataset, widen_factor, droprate, blk):
    # Select parameters
    if dataset == 'cifar10':
        nc = 3
        nd = 1
        num_class = 10;
    if dataset == 'cifar100':
        nc = 3
        nd = 1
        num_class = 100;
    if dataset == 'FashionMNIST':
        nc = 1
        nd = 1
        num_class = 10
    if dataset == 'mini_imagenet':
        nc = 3
        nd = 2
        num_class = 100;
    if dataset == 'imagenet':
        nc = 3
        nd = 2
        num_class = 1000;
    # create model
    if model == 'WideResNet':
        model = WideResNet(layers, num_class, widen_factor, dropRate=droprate, nc=nc, nd=nd)
    if model == 'WideResNet_DDplus':
        model = WideResNet_DDplus(layers, num_class, widen_factor, droprate, nc=nc, nd=nd, pretrained=False, path = '')
    if model == 'Resnet':
        model = Resnet(layers, num_class, nc=nc, nd=nd, width=widen_factor)
    if model == 'Resnet_DDplus':
        model = Resnet_DDplus(layers, num_class, widen_factor, nc=nc, nd=nd, pretrained=False, path = '')
    if model == 'Se_net':
        model = Se_net(layers, num_class, nc=nc, nd=nd, width=widen_factor)
    if model == 'Se_net_DDplus':
        model = Se_net_DDplus(layers, num_class, widen_factor, nc=nc, nd=nd, pretrained=False, path = '')
    if model == 'DenseNet':
        model = DenseNet(layers, num_class, widen_factor, nc=nc)
    if model == 'DenseNet_DDplus':
        model = DenseNet_DDplus(layers, num_class, widen_factor, nc=nc)
    if model == 'Shake_shake':
        model = Shake_shake(widen_factor, num_class, nc=nc, nd=nd)
    if model == 'Shake_shake_DDplus':
        model = Shake_shake_DDplus(widen_factor, num_class, nc=nc, nd=nd)
    if model == 'Pyramid':
        model = Pyramid(dataset, layers, width=widen_factor, num_classes=num_class, nc=nc, nd=nd, blk=blk)
    if model == 'Pyramid_DDplus':
        model = Pyramid_DDplus(dataset, layers, widen_factor, num_class, pretrained=False, nd=1, blk=blk, path = './result/cifar100_Pyramid_272_1_accuracy_200/model.pth')
    return model