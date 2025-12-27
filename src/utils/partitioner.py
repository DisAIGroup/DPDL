import random, math, sys
import torch
from torch.utils.data import DataLoader #, Sampler, DistributedSampler
import numpy as np
import matplotlib.pyplot as plt

COLOR_MAP = ['red', 'green', 'blue', 'black', 'brown', 'purple', 'yellow', 'pink', 'cyan', 'gray']

class Partition(object):
	
	def __init__(self, data, label, index):
		self.data = data
		self.label = label
		self.index = index
		
	def __len__(self):
		return len(self.index)
	
	def __getitem__(self, index):
		idx = self.index[index]
		data_ele = self.data[idx]
		label_ele = self.label[idx]
		return data_ele, label_ele


class Partition_Loader(object):
    """
    Partition the whole dataset into smaller sets for each rank.
    """ 
    def __init__(self, data, label, batch_size, train=True):
        self.data = data
        self.label = label
        self.batch_size = batch_size
        self.index = [i for i in range(len(self.data))]
        self.partitioner = Partition(data, label, self.index)
        if train:
            self.loader = self.trn_loader()
        else:
            self.loader = self.tst_loader()
        
    def trn_loader(self): 
        return DataLoader(self.partitioner, batch_size=self.batch_size, shuffle=True)
    def tst_loader(self): 
        return DataLoader(self.partitioner, batch_size=self.batch_size, shuffle=False)


def calculate_distribution(num_clients, num_class, data, labels):
    client_statis = []
    for client_id in range(num_clients):
        samples_distribution = [0] * num_class
        data_set, label_set = data[client_id], labels[client_id]
        train_loader = torch.utils.data.DataLoader(
            Partition(data_set, label_set, index=[i for i in range(len(data_set))]),
            batch_size=1,
            shuffle=True
        )
        for _, label in train_loader:
            samples_distribution[int(label.data.numpy())] += 1
        client_statis.append(samples_distribution)
    return client_statis


def plot_distribution(client_statis, num_clients, num_class, dataset, save_path, sample_strategy, dirichlet_alpha, phase):
    client_list = np.array(client_statis)
    client_name_list = ['Client' + str(client_id) for client_id in range(num_clients)]

    plt.figure(figsize=(10, 10))
    bot = np.zeros([num_class, num_clients])
    for i in range(num_class):
        for j in range(i):
            bot[i][:] += client_list[:, j]
    x = range(len(client_name_list))
    for label_id in range(num_class):
        plt.bar(x=x,
                height=client_list[:, label_id],
                width=0.8,
                alpha=0.8,
                color=COLOR_MAP[label_id % len(COLOR_MAP)],
                label=str(label_id),
                bottom=bot[label_id])

    plt.ylabel("Quantity")
    plt.xticks(x, client_name_list)
    plt.xlabel("Client")
    plt.title("Sample distribution")
    plt.legend()
    plt.savefig(f"{save_path}/{phase}_{dataset}_{num_clients}_{sample_strategy}_{dirichlet_alpha}.png")
    plt.close()


def show_statis(data_obj, num_clients, num_class, dataset, save_path='./distribute/', train=True):
    if train:
        client_statis = calculate_distribution(num_clients, num_class, data_obj.clnt_x, data_obj.clnt_y)
        phase = 'train'
    else:
        client_statis = calculate_distribution(num_clients, num_class, data_obj.tst_x, data_obj.tst_y)
        phase = 'test'
    
    plot_distribution(client_statis, num_clients, num_class, dataset, save_path, data_obj.sample_strategy, data_obj.dirichlet_alpha, phase)

