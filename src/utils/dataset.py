import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset, Dataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os
import argparse
from collections import Counter
import copy

class Dataset:
    def __init__(self, dataset_name, data_dir, data_npy_dir, n_clients,
                 sample_strategy, dirichlet_alpha, batch_size, trn_time, tst_time, balance):
        '''
        '''
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.data_npy_dir = data_npy_dir
        self.n_clients = n_clients
        # self.training_rate = training_rate
        # self.testing_rate = testing_rate
        self.sample_strategy = sample_strategy  # 是否使用iid
        self.dirichlet_alpha = dirichlet_alpha  # 狄利克雷分布参数
        self.batch_size = batch_size
        self.trn_time = trn_time
        self.tst_time = tst_time
        self.download = False
        self.balance = balance
        if self.balance:
            self.unbalance = 0.0
        else:
            self.unbalance = 0.5


    def get_data(self):

        '''
        相当于只处理对应的数据，根据数据集和对应的是否non-iid的方法生成需要的数据集
        但是没有对数据进行划分，另外需要一个partition的操作来讲数据划分到每个客户端
        '''
        # print(self.data_dir)

        if not os.path.exists('%s/%s' %(self.data_dir, self.dataset_name)):
            # print("Data is downloading")
            self.download = True
        else:
            # print("Data is already downloaded")
            pass

        if not os.path.exists('%s/%s/clnt_x_%s_%s_%s_%s.npy' %(self.data_npy_dir, self.dataset_name, self.sample_strategy, self.dirichlet_alpha, self.n_clients, self.balance)):
            
            # print("Data is generating") # 没有本地生成的数据

            # MNIST单通道，CIFAR三通道
            if self.dataset_name == 'MNIST':  
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                #归一化操作，它的含义是将图像值都转换到[-1,1]之间
                train_data = datasets.MNIST(root='%s' %(self.data_dir), download=self.download, train=True, transform=transform)
                test_data = datasets.MNIST(root='%s' %(self.data_dir), download=self.download, train=False, transform=transform)
                # 数据必须经过DataLoader读取，否则无法生成归一化的数据
                trn_load = torch.utils.data.DataLoader(train_data, batch_size=60000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.classes = 10;
            
            if self.dataset_name == 'EMNIST':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
                train_data = datasets.EMNIST(root='%s/%s' %(self.data_dir, self.dataset_name), download=self.download, train=True, transform=transform)
                test_data = datasets.EMNIST(root='%s/%s' %(self.data_dir, self.dataset_name),  download=self.download, train=False, transform=transform)
                trn_load = torch.utils.data.DataLoader(train_data, batch_size=self.trn_time*self.batch_size*self.n_clients, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(test_data, batch_size=self.tst_time*self.batch_size*self.n_clients, shuffle=False, num_workers=1)
                self.channels = 1; self.width = 28; self.height = 28; self.classes = 10;
            
            if self.dataset_name == 'CIFAR10':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                    mean=[0.491, 0.482, 0.447],std=[0.247, 0.243, 0.261])])
                train_data = datasets.CIFAR10(root='%s/%s' %(self.data_dir, self.dataset_name), download=self.download, train=True, transform=transform)
                test_data = datasets.CIFAR10(root='%s/%s' %(self.data_dir, self.dataset_name), download=self.download, train=False, transform=transform)
                trn_load = torch.utils.data.DataLoader(train_data, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.classes = 10;
            
            if self.dataset_name == 'CIFAR100':
                transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],std=[0.2675, 0.2565, 0.2761])])
                train_data = datasets.CIFAR100(root='%s/%s' %(self.data_dir, self.dataset_name), download=self.download, train=True, transform=transform)
                test_data = datasets.CIFAR100(root='%s/%s' %(self.data_dir, self.dataset_name), download=self.download, train=False, transform=transform)
                trn_load = torch.utils.data.DataLoader(train_data, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(test_data, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.classes = 100;
                
            trn_x, trn_y = next(iter(trn_load))  # next(iter())可以取出DataLoader中一个batch的数据
            tst_x, tst_y = next(iter(tst_load))

            trn_x = trn_x.numpy()
            trn_y = trn_y.numpy().reshape(-1, 1)
            tst_x = tst_x.numpy()
            tst_y = tst_y.numpy().reshape(-1, 1)

            # Shuffle操作
            permutation = np.random.permutation(len(trn_y))
            trn_x = trn_x[permutation]
            trn_y = trn_y[permutation]
            
            tst_x_copied = np.tile(tst_x, (self.n_clients,) + (1,) * (len(tst_x.shape) - 1))
            tst_y_copied = np.tile(tst_y, (self.n_clients,) + (1,) * (len(tst_y.shape) - 1))
            tst_x = np.array(np.split(tst_x_copied, self.n_clients))
            tst_y = np.array(np.split(tst_y_copied, self.n_clients))
            
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y

            n_classes = trn_y.max()+1

            if self.sample_strategy == 'dir':
                
                '''
                # 使用Dirichlet分布为每个客户端生成类别分布
                label_distribution = np.random.dirichlet([self.dirichlet_alpha] * self.n_clients, n_classes)

                # 记录每个类别对应的样本下标
                class_idcs = [np.where(trn_y == y)[0] for y in range(n_classes)]

                # 初始化一个空列表用于存储每个客户端的数据下标
                client_idcs = [[] for _ in range(self.n_clients)]

                # 根据Dirichlet分布将类别样本下标分配给各个客户端
                for c, fracs in zip(class_idcs, label_distribution):
                    for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                        client_idcs[i] += [idcs]

                # 将每个客户端的下标列表连接起来，形成最终的下标列表
                client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

                # 初始化客户端数据容器
                clnt_x = [np.zeros((len(client_idcs[i]), self.channels, self.height, self.width)).astype(np.float32) for i in range(self.n_clients)]
                clnt_y = [np.zeros((len(client_idcs[i]), 1)).astype(np.int64) for i in range(self.n_clients)]

                # 根据生成的下标为每个客户端分配数据
                for clnt_i, idcs in enumerate(client_idcs):
                    for j, idx in enumerate(idcs):
                        clnt_x[clnt_i][j] = trn_x[idx]
                        clnt_y[clnt_i][j] = trn_y[idx]
                '''

                # 每个客户端的数据数量
                n_data_per_clnt = len(trn_y) // self.n_clients
                clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalance, size=self.n_clients)
                clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(trn_y)).astype(int)

                # 使用狄利克雷分布生成类先验
                cls_priors = np.random.dirichlet(alpha=[self.dirichlet_alpha] * n_classes, size=self.n_clients)
                prior_cumsum = np.cumsum(cls_priors, axis=1)

                # 按类别记录样本索引
                idx_list = [np.where(trn_y == i)[0] for i in range(n_classes)]
                cls_amount = [len(idx_list[i]) for i in range(n_classes)]

                # 初始化客户端数据和标签
                clnt_x = [np.zeros((clnt_data_list[clnt], self.channels, self.height, self.width), dtype=np.float32) for clnt in range(self.n_clients)]
                clnt_y = [np.zeros(clnt_data_list[clnt], dtype=np.int64) for clnt in range(self.n_clients)]

                while np.sum(clnt_data_list) != 0:
                    curr_clnt = np.random.randint(self.n_clients)
                    print('\rRemaining Data: %d' %np.sum(clnt_data_list), end='', flush=True)
                    # 如果当前客户端的数据已经满了，则重新选择客户端
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]

                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # 如果该类别的样本已经分完了，则重新选择类别
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]
                        break

            else:
                '''
                data_per_client = int(n_sample / self.n_clients)
                clients = []
                for i in range(self.n_clients):
                    start_idx = i * data_per_client
                    end_idx = (i + 1) * data_per_client
                    data_subset = train_data[start_idx:end_idx]
                    data_loader = torch.utils.data.DataLoader(data_subset, batch_size=64, shuffle=True)
                    clients.append(data_loader)
                '''

                n_data_per_clnt = len(trn_y) // self.n_clients
                clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=self.n_clients)
                # 用于决定均匀分布的数据分配，sigma决定了数据分布是否条数相同
                clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(trn_y)).astype(int)

                # 初始化客户端数据容器
                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_clients) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_clients) ]
            
                clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(clnt_data_list)))
                for clnt_idx_ in range(self.n_clients):
                    clnt_x[clnt_idx_] = trn_x[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                    clnt_y[clnt_idx_] = trn_y[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
                
            clnt_x = np.asarray(clnt_x)
            clnt_y = np.asarray(clnt_y)

            self.clnt_x = clnt_x; self.clnt_y = clnt_y
            self.tst_x  = tst_x;  self.tst_y  = tst_y

            if not os.path.exists('%s/%s' %(self.data_dir, self.dataset_name)):
                os.mkdir('./data/%s' %(self.dataset_name))

            del train_data
            del test_data
            del trn_x
            del trn_y
            
            np.save('./data/%s/clnt_x_%s_%s_%s_%s.npy' %(self.dataset_name, self.sample_strategy, self.dirichlet_alpha, self.n_clients, self.balance), clnt_x)
            np.save('./data/%s/clnt_y_%s_%s_%s_%s.npy' %(self.dataset_name, self.sample_strategy, self.dirichlet_alpha, self.n_clients, self.balance), clnt_y)

            np.save('./data/%s/tst_x_%s_%s_%s_%s.npy' %(self.dataset_name, self.sample_strategy, self.dirichlet_alpha, self.n_clients, self.balance), tst_x)
            np.save('./data/%s/tst_y_%s_%s_%s_%s.npy' %(self.dataset_name, self.sample_strategy, self.dirichlet_alpha, self.n_clients, self.balance), tst_y)

        else:
            # print("Data is already generated")

            self.clnt_x = np.load('./data/%s/clnt_x_%s_%s_%s_%s.npy' %(self.dataset_name, self.sample_strategy, self.dirichlet_alpha, self.n_clients, self.balance),allow_pickle=True)
            self.clnt_y = np.load('./data/%s/clnt_y_%s_%s_%s_%s.npy' %(self.dataset_name, self.sample_strategy, self.dirichlet_alpha, self.n_clients, self.balance),allow_pickle=True)
            self.n_clients = len(self.clnt_x)

            self.tst_x  = np.load('./data/%s/tst_x_%s_%s_%s_%s.npy' %(self.dataset_name, self.sample_strategy, self.dirichlet_alpha, self.n_clients, self.balance),allow_pickle=True)
            self.tst_y  = np.load('./data/%s/tst_y_%s_%s_%s_%s.npy' %(self.dataset_name, self.sample_strategy, self.dirichlet_alpha, self.n_clients, self.balance),allow_pickle=True)
            
            if self.dataset_name == 'MNIST':
                self.channels = 1; self.width = 28; self.height = 28; self.classes = 10;
            if self.dataset_name == 'EMNIST':
                self.channels = 1; self.width = 28; self.height = 28; self.classes = 10;
            if self.dataset_name == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.classes = 10;
            if self.dataset_name == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.classes = 100;
        
        return self.clnt_x, self.clnt_y, self.tst_x, self.tst_y, self.classes
    
    def target_stats(self, y):

        for i in range(self.n_clients):
            y_per_client = copy.deepcopy(y[i])
            y_per_client_dic = dict(Counter(y_per_client.squeeze()))
            y_per_client_dic_order = {}
            for key in sorted(y_per_client_dic):
                y_per_client_dic_order[key] = y_per_client_dic[key]
            print(y_per_client_dic_order)


    def partition_data_dirichlet(train_data, trn_x, trn_y, n_clients, dirichlet_alpha, channels, height, width):
        
        n_sample = len(train_data.data)
        train_labels = np.array(train_data.targets)
        n_cls = train_labels.max() + 1

        # 每个客户端的数据数量
        n_data_per_clnt = len(trn_y) / n_clients
        clnt_data_list = np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=0, size=n_clients)
        clnt_data_list = (clnt_data_list / np.sum(clnt_data_list) * len(trn_y)).astype(int)

        # 使用狄利克雷分布生成类先验
        cls_priors = np.random.dirichlet(alpha=[dirichlet_alpha] * n_cls, size=n_clients)
        prior_cumsum = np.cumsum(cls_priors, axis=1)

        # 按类别记录样本索引
        idx_list = [np.where(trn_y == i)[0] for i in range(n_cls)]
        cls_amount = [len(idx_list[i]) for i in range(n_cls)]

        # 初始化客户端数据和标签
        clnt_x = [np.zeros((clnt_data_list[clnt], channels, height, width), dtype=np.float32) for clnt in range(n_clients)]
        clnt_y = [np.zeros(clnt_data_list[clnt], dtype=np.int64) for clnt in range(n_clients)]

        while np.sum(clnt_data_list) != 0:
            curr_clnt = np.random.randint(n_clients)
            # 如果当前客户端的数据已经满了，则重新选择客户端
            if clnt_data_list[curr_clnt] <= 0:
                continue
            clnt_data_list[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]

            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # 如果该类别的样本已经分完了，则重新选择类别
                if cls_amount[cls_label] <= 0:
                    continue
                cls_amount[cls_label] -= 1
                clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]
                break

        return np.asarray(clnt_x), np.asarray(clnt_y)

