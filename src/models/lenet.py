import torch
#from torchinfo import summary

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from copy import deepcopy


__all__ = ['lenet','LeNet_cifar','LeNet_mnist']


class LeNet_cifar(nn.Module):
    def __init__(self):
        """
        Build a LeNet5 pytorch Module.
        """
        super(LeNet_cifar, self).__init__()
        # feature extractor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # classifer
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Define the forward pass of a LeNet5 module.
        """
        # feature extraction
        x = self.relu(self.conv1(x))        # input(3, 32, 32) output(16, 28, 28)
        x = self.maxpool1(x)                # output(16, 14, 14)
        x = self.relu(self.conv2(x))        # output(32, 10, 10)
        x = self.maxpool2(x)                # output(32, 5, 5)
        # classification
        x = x.view(-1, 32*5*5)              # output(32*5*5)
        x = self.relu(self.fc1(x))          # output(120)
        x = self.relu(self.fc2(x))          # output(84)
        out = self.fc3(x)                   # output(10)
        return out
    

class LeNet_mnist(nn.Module):
    def __init__(self):
        """
        Build a LeNet5 pytorch Module.
        """
        super(LeNet_mnist, self).__init__()
        # feature extractor
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # classifer
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Define the forward pass of a LeNet5 module.
        """
        # feature extraction
        x = self.relu(self.conv1(x))        # input(1, 28, 28) output(6, 24, 24)
        x = self.maxpool1(x)                # output(6, 12, 12)
        x = self.relu(self.conv2(x))        # output(16, 8, 8)
        x = self.maxpool2(x)                # output(16, 4, 4)
        # classification
        x = x.view(-1, 16*4*4)              # output(16*4*4)
        x = self.relu(self.fc1(x))          # output(120)
        x = self.relu(self.fc2(x))          # output(84)
        out = self.fc3(x)                   # output(10)
        return out


def lenet(**kwargs):
    arch, dataset = map(
        kwargs.get, ['arch', 'dataset'])
    
    if dataset == "cifar10":
        return LeNet_cifar()

    elif dataset == "mnist":
        return LeNet_mnist()


