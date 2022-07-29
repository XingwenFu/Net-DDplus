# 基础包
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tool.random_erasing import RandomErasing
import torchvision.datasets as datasets
import numpy as np
# 架构
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
# 运行设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)

# In[1] 定义测试函数
def test(model,test_loader):
    
    model.eval() #切换到评估模式
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs= model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc=100 * correct / total
    
        # print('Accuracy of the model on the test images: {} %'.format(acc))
    return acc

def best_prec(best_prec, path, name):
    best_prec = np.array(best_prec)
    try:
        precmax = np.load(path+'/'+name+'.npy')
        
    except:
        precmax = 0
        precmax = np.array(precmax)
    
    if precmax < best_prec:
        precmax = best_prec
        np.save(path+'/'+name+'.npy', precmax)
    print('the best Prediction accuracy of',name,'is:', precmax)

def Robust_test(model, dataset, path):
    batch_size = 512
    if  dataset == 'cifar10' or dataset == 'cifar100':
        # 归一化，白化参数
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        # NO_normalize
        transform_test = transforms.Compose([
            transforms.ToTensor(),                # 张量
            ])
        kwargs = {'num_workers': 0, 'pin_memory': True}
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[dataset.upper()]('./data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        prec_NO_n = test(model,val_loader)
        best_prec(prec_NO_n, path, 'prec_NO_n')
        # print('NO_normalize：', prec_NO_n)
        # 干扰办法
        transform_test = transforms.Compose([
            transforms.ToTensor(),                # 张量
            normalize,                            # 归一化，白化
            RandomErasing(),                      # 随机遮挡
            ])
        kwargs = {'num_workers': 0, 'pin_memory': True}
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[dataset.upper()]('./data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        prec_RE = test(model,val_loader)
        best_prec(prec_RE, path, 'prec_RE')
        # print('随机遮挡的精度：', prec_RE)

        # 干扰办法
        transform_test = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5), # 随机改变图片的亮度brightness，色调hue，对比度contrast，饱和度saturation
            transforms.ToTensor(),                # 张量
            normalize,                            # 归一化，白化
            ])
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[dataset.upper()]('./data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        prec_CJ = test(model,val_loader)
        best_prec(prec_CJ, path, 'prec_CJ')
        # print('随机调节亮度brightness，色调hue，对比度contrast，饱和度saturation的精度：', prec_CJ)
        # 干扰办法
        transform_test = transforms.Compose([
            transforms.RandomRotation((0,360), resample=False,expand=False,center=None),
            transforms.ToTensor(),                # 张量
            normalize,                            # 归一化，白化
            ])

        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[dataset.upper()]('./data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        prec_RR = test(model,val_loader)
        best_prec(prec_RR, path, 'prec_RR')
        # print('随机水平和竖直翻转的精度：', prec_RR)
    if  dataset == 'FashionMNIST':
        # 归一化，白化参数
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        # NO_normalize
        transform_test = transforms.Compose([
            transforms.ToTensor(),                # 张量
            ])
        kwargs = {'num_workers': 0, 'pin_memory': True}
        val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        prec_NO_n = test(model, val_loader)
        best_prec(prec_NO_n, path, 'prec_NO_n')
        # print('NO_normalize：', prec_NO_n)
        # 干扰办法
        transform_test = transforms.Compose([
            transforms.ToTensor(),                # 张量
            transforms.Normalize((0.1307,), (0.3081,)),                            # 归一化，白化
            RandomErasing(probability=0.1, mean=[0.4914]),                      # 随机遮挡
            ])
        kwargs = {'num_workers': 0, 'pin_memory': True}
        val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        prec_RE = test(model, val_loader)
        best_prec(prec_RE, path, 'prec_RE')
        # print('随机遮挡的精度：', prec_RE)

        # 干扰办法
        transform_test = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5), # 随机改变图片的亮度brightness，色调hue，对比度contrast，饱和度saturation
            transforms.ToTensor(),                # 张量
            transforms.Normalize((0.1307,), (0.3081,)),                            # 归一化，白化
            ])
        val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        prec_CJ = test(model,val_loader)
        best_prec(prec_CJ, path, 'prec_CJ')
        # print('随机调节亮度brightness，色调hue，对比度contrast，饱和度saturation的精度：', prec_CJ)
        # 干扰办法
        transform_test = transforms.Compose([
            transforms.RandomRotation((0,360), resample=False,expand=False,center=None),
            transforms.ToTensor(),                # 张量
            transforms.Normalize((0.1307,), (0.3081,)),                            # 归一化，白化
            ])

        val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data', train=False, transform=transform_test),
            batch_size=batch_size, shuffle=True, **kwargs)
        prec_RR = test(model,val_loader)
        best_prec(prec_RR, path, 'prec_RR')
        # print('随机水平和竖直翻转的精度：', prec_RR)

def main():
    # 定义一个模型的架构
    model = WideResNet_DDplus(layers=20, num_classes=10, width=1, dropRate=0.0, nc=3, nd=1, pretrained=False, path='').to(device)  # 对应的是WRN40_4和cifar10数据集
    # 参数
    # 导入你想要测试的模型参数
    path = "./cifar10_WideResNet_CDD_40_4/model.pth"   # 模型参数的位置
    model.load_state_dict(torch.load(path))
    dataset = 'CIFAR10'
    # dataset = 'CIFAR100'
    Robust_test(model, dataset, path)

if __name__ == '__main__':
    main()