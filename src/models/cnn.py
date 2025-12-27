import torch
#from torchinfo import summary

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from copy import deepcopy

__all__ = ['cnn','CNN_mnist', 'CNN_cifar']


class CNN_mnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sq1 = nn.Sequential(         
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),   # (16, 28, 28)                           #  output: (16, 28, 28)
            nn.ReLU(),                    
            nn.MaxPool2d(kernel_size=2),    # (16, 14, 14)
        )
        self.Sq2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # (32, 14, 14)
            nn.ReLU(),                      
            nn.MaxPool2d(2),                # (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   

    def forward(self, x):
        x = self.Sq1(x)
        x = self.Sq2(x)
        x = x.view(x.size(0), -1)          
        output = self.out(x)
        return output


# class CNN_cifar(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.Sq1 = nn.Sequential(         
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2),   # (16, 32, 32)                           #  output: (16, 28, 28)
#             nn.ReLU(),                    
#             nn.MaxPool2d(kernel_size=2),    # (16, 16, 16)
#         )
#         self.Sq2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # (32, 16, 16)
#             nn.ReLU(),                      
#             nn.MaxPool2d(2),                # (32, 8, 8)
#         )
#         self.out = nn.Linear(32 * 8 * 8, 10)   

#     def forward(self, x):
#         x = self.Sq1(x)
#         x = self.Sq2(x)
#         x = x.view(x.size(0), -1)          
#         output = self.out(x)
#         return output


class CNN_cifar(nn.Module):
    def __init__(self):
        super(CNN_cifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 输出为16*16*16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 输出为32*8*8
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)  # 防止过拟合

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        return output


class CNN_cifar100(nn.Module):
    def __init__(self):
        super(CNN_cifar100, self).__init__()
        # 增加卷积层的数量和宽度
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 输出为32*32*32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # 输出为64*16*16
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # 输出为128*8*8
        self.pool = nn.MaxPool2d(2, 2)
        
        # 增加全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 100)  # 输出100个类别

        # 激活函数和 dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # 增加 dropout 比率以防止过拟合

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def cnn(**kwargs):
    arch, dataset = map(
        kwargs.get, ['arch', 'dataset'])
    
    if dataset == "cifar10":
        return CNN_cifar()
    
    elif dataset == "cifar100":
        return CNN_cifar100()

    elif dataset == "mnist":
        return CNN_mnist()