###
#这是一个下载数据的程序
#付幸文于2021.7.23
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import torch.utils.data.distributed
from tool.random_erasing import RandomErasing
import torch.nn.functional as F
def data_loader(dataset, Test, batch_size):
    if dataset == 'cifar10' or dataset == 'cifar100':
        # Data loading code
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        if Test == 'robust':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4,4,4,4),mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),            # 随机裁剪为指定大小
                transforms.RandomRotation((0,360), resample=False,expand=False,center=None),  #任意角度旋转
                transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5), # 随机改变图片的亮度brightness，色调hue，对比度contrast，饱和度saturation
                transforms.ToTensor(),                # 张量
                normalize,                            # 归一化，白化
                RandomErasing(),                      # 随机遮挡
                ])
        elif Test == 'accuracy':
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                  (4,4,4,4),mode='reflect').squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),            # 随机裁剪为指定大小
                transforms.RandomHorizontalFlip(),    # 随机水平翻转
                transforms.ToTensor(),                # 张量
                normalize,                            # 归一化，白化
                ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            # RandomErasing(),   # 随机遮挡
            ])

        kwargs = {'num_workers': 0, 'pin_memory': True}
        assert(dataset == 'cifar10' or dataset == 'cifar100')
        train_loader = torch.utils.data.DataLoader(
            datasets.__dict__[dataset.upper()]('./data', train=True, download=True,
                             transform=transform_train),
            batch_size= batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.__dict__[dataset.upper()]('./data', train=False, transform=transform_test),
            batch_size= batch_size, shuffle=True, **kwargs)
#  ====================================================================================================
    if dataset == 'FashionMNIST':
        # In[1] 加载数据
        if Test == 'robust':
            transform_train=transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomRotation((0,360), resample=False,expand=False,center=None), # 任意角度旋转
                transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5), # 随机改变图片的亮度brightness，色调hue，对比度contrast，饱和度saturation
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                RandomErasing(probability=0.1, mean=[0.4914]),
                ])
        if Test == 'accuracy':
            transform_train=transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                # RandomErasing(probability=0.1, mean=[0.4914]),
                ])
        transform_test=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data',train=True, download=True,transform=transform_train),
            batch_size=batch_size, shuffle=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data',train=False, download=True,transform=transform_test),
            batch_size=batch_size, shuffle=False)
#  ====================================================================================================
    if dataset == 'mini_imagenet':
        print('yes')
        transform_train = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
        ])
        transform_test = transforms.Compose([
            transforms.Resize((64,64)),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
        ])

        #读取数据
        dataset_train = datasets.ImageFolder('./data/mini_imagenet/train', transform_train)
        dataset_test = datasets.ImageFolder('./data/mini_imagenet/test', transform_test)
        #dataset_val = datasets.ImageFolder('data/val', transform)

        # 上面这一段是加载测试集的
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True) # 训练集
        val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True) # 测试集
        #val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True) # 验证集
        # 对应文件夹的label
        # print(dataset_train.class_to_idx)   # 这是一个字典，可以查看每个标签对应的文件夹，也就是你的类别。
                                            # 训练好模型后输入一张图片测试，比如输出是99，就可以用字典查询找到你的类别名称
        # print(dataset_test.class_to_idx)
        #print(dataset_val.class_to_idx)
    if dataset == 'imagenet':
        print('yes')
        transform_train = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
        ])
        transform_test = transforms.Compose([
            transforms.Resize((64,64)),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 归一化处理
        ])

        #读取数据
        dataset_train = datasets.ImageFolder('/root/train', transform_train)
        dataset_test = datasets.ImageFolder('/root/val', transform_test)
        #dataset_val = datasets.ImageFolder('data/val', transform)

        # 上面这一段是加载测试集的
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True) # 训练集
        val_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True) # 测试集
        #val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True) # 验证集
        # 对应文件夹的label
        # print(dataset_train.class_to_idx)   # 这是一个字典，可以查看每个标签对应的文件夹，也就是你的类别。
                                            # 训练好模型后输入一张图片测试，比如输出是99，就可以用字典查询找到你的类别名称
        # print(dataset_test.class_to_idx)
        #print(dataset_val.class_to_idx)
    return train_loader, val_loader
