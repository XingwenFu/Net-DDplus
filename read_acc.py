# -*- coding: utf-8 -*-
"""
付幸文于2021.7.31
这是一个读结果的程序
"""

import numpy as np
import matplotlib.pyplot as plt

def read_results(dataset, model, layers, widths, Test):
    file = './result/'
    file_path = file+dataset+'_'+model+'_'+str(layers)+'_'+str(widths)+'_'+Test+'_'+str(200)
    
    print(file_path)
    test_acc = np.load(file_path+'/test_acc.npy')
    print(np.shape(test_acc))

    
    if Test == 'robust':
        prec_CJ = np.load(file_path+'/prec_CJ.npy')
        print('prec_CJ:', prec_CJ )
        prec_NO_n = np.load(file_path+'/prec_NO_n.npy')
        print('prec_NO_n:', prec_NO_n )
        prec_RE = np.load(file_path+'/prec_RE.npy')
        print('prec_RE:', prec_RE )
        prec_RR = np.load(file_path+'/prec_RR.npy')
        print('prec_RR:', prec_RR )

    if Test == 'accuracy':
        best_prec = np.load(file_path+'/best_prec.npy')
        print('best_prec:', best_prec )

def read_plot(dataset, model, layers, widths, Test):
    file = './result/'
    file_path = file+dataset+'_'+model+'_'+str(layers)+'_'+str(widths)+'_'+Test+'_'+str(200)
    
    print(file_path)
    test_acc = np.load(file_path+'/test_acc.npy')
    train_loss = np.load(file_path+'/train_loss.npy')
    print(np.shape(test_acc))
    return test_acc, train_loss
    
read_results(dataset='mini_imagenet', model='Resnet', layers=56, widths=4, Test='accuracy')
# read_results(dataset='mini_imagenet', model='Resnet_CDD', layers=56, widths=4, Test='accuracy')
test_acc1, train_loss1 = read_plot(dataset='mini_imagenet', model='Resnet', layers=56, widths=4, Test='accuracy')
test_acc2, train_loss2 = read_plot(dataset='mini_imagenet', model='Resnet_CDD', layers=56, widths=4, Test='accuracy')

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
ax.plot(test_acc2, label='ResNet-DDplus-74-4')
ax.plot(test_acc1, label='ResNet-74-4')
plt.legend(loc='lower right', frameon=True)
# plt.legend([p1, p2], ["ResNet-CDD-74-4", "ResNet-74-4"], loc='upper left')
plt.ylabel("Test Accuracy (%)", {'weight': 'normal', 'size': 12})  # 设置y坐标
plt.xlabel("epoch", {'weight': 'normal', 'size': 12})  # 设置y坐标
plt.grid(axis="y", ls=":", lw=1, color="gray", alpha=.4)  # 设置网格线
plt.savefig('testacc.jpeg', dpi=300)
plt.show()
