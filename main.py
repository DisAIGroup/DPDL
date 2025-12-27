import pickle
import os
import copy
import json
import numpy as np
from math import ceil
import random
from collections import Counter
from itertools import islice
import logging
import pandas as pd
import sys
import logging
import argparse
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim as optim
from torchsummary import summary
import torch.nn.functional as F
from src.utils.dataset import Dataset
from src.utils.partitioner import Partition, Partition_Loader, show_statis
from src.utils.graph_set import Graph
from src.models import *
from src.client import Client


def parse_arguments():
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser = argparse.ArgumentParser(description='PPDSL')
    # basic environment
    parser.add_argument('--use_cuda', action="store_true", help='Use CUDA if available')
    parser.add_argument('-d', '--devices', nargs='+', type=int, default=[0], help='list of gpus/devices')
    parser.add_argument('--port', dest='port',   help='between 3000 to 65000',default='25550' , type=str)
    parser.add_argument('--n_clients', type=int, default=10,
                        help='number of workers in a distributed cluster')
    parser.add_argument("--file", type=int, default=0)
    parser.add_argument("--mode", type=str, default="train")
    
    # paths
    parser.add_argument('--data-dir', dest='data_dir',    help='The directory used to save the trained models',   default='../data', type=str)
    parser.add_argument('--data_npy_dir', type=str, default='./data',
                            help='data pny file directory, please feel free to change the directory to the right place')
    parser.add_argument('--check_dir', type=str, default='./checkpoint', help='save checkpoints')
    
    # dataset and topology
    parser.add_argument('--dataset', dest='dataset', default='CIFAR10', type=str,
                        help ='CIFAR10, CIFAR100, MNIST,FMNIST...')
    parser.add_argument('--classes', type=int, default=10)
    parser.add_argument('--balance', type=bool, default=True)
    parser.add_argument('--avg', type=int, default=0)
    parser.add_argument('--reg', type=float, default=1e-2, help='neighbor weight difference')
    parser.add_argument('--sample_strategy', type=str, default='dir',
                            help="current supporting three types of data partition, one called 'dir' short for Dirichlet"
                                "one called 'n_cls' short for how many classes allocated for each client"
                                "and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution")
    parser.add_argument('--topology', type=str, default='FC', choices=("Ring", "Bipar", "FC", "Grid", "Random"),
                            help="graph structure")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.25,
                            help='The smaller,the deeper degree of non-iid. Available parameters for data partition method')
    
    # training settings
    parser.add_argument('--round', type=int, default=1, help = 'exe times of each comparsion group')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='cganet', 
                        help = 'cnn, lenet, cganet, mobilenet, resnet20, resnet32, vgg11, vgg13, vgg16, vgg19...' )
    parser.add_argument('--methods', type=str, default='ppdsl',
                        help='ppdsl, netfleet, dpcga, muffliato, adp2sgd')
    parser.add_argument('--normtype',   default='evonorm', help = 'none or batchnorm or groupnorm or evonorm' )
    parser.add_argument('--epochs', type=int, default=100,
                        help='local training epochs for each client')
    parser.add_argument('--trn_time', type=int, default=10,
                        help='local training times in one epoch')
    parser.add_argument('--tst_time', type=int, default=5,
                        help='local test times in one epoch')
    parser.add_argument('--trn_bsz', type=int, default=256, 
                        help='local batch size for training')
    parser.add_argument('--tst_bsz', type=int, default=64, 
                        help='local batch size for testing')
    parser.add_argument('--client_optimizer', type=str, help='SGD with momentum')
    parser.add_argument('--lr', type=float, default=0.1, 
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0,
                        help='learning rate decay')
    parser.add_argument('--inner_round', type=int, default=3)
    parser.add_argument('--wd', help='weight decay parameter', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--qgm', action='store_true', help='quasi global momentum')
    parser.add_argument('--print_freq', '-p', default=20, type=int,  help='print frequency (default: 50)') 
    
    # hyperparams
    parser.add_argument('--is_dp', action='store_true', help='whether add privacy')
    parser.add_argument('--dp_eps', type=float, default=5, help='privacy epsilon')
    parser.add_argument('--dp_budget', type=float, default=1e-3, help='privacy budget')
    parser.add_argument('--ct', type=float, default=1, help='clipping threshold')
    parser.add_argument('--sigma', type=float, default=5, help='noise degree')
    parser.add_argument('--epsilon', type=float, default=1, help='dp budget')
    
    return parser.parse_args()


def configure_environment(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    devices = args.devices
    if isinstance(args.devices, str):
        devices = args.devices.split(',')
    devices = [torch.device('cuda:{}'.format(d)) for d in devices]
    def distribute_clients(n_clients, devices):
        num_devices = len(devices)
        avg_clients_per_device = n_clients // num_devices
        remainder = n_clients % num_devices 
        num_clients_per_gpu = [avg_clients_per_device] * num_devices
        for i in range(remainder):
            num_clients_per_gpu[i] += 1
        return num_clients_per_gpu
    num_clients_per_gpu = distribute_clients(args.n_clients, args.devices)
    return devices, num_clients_per_gpu


def initialize_dataset(args):
    # 读取划分数据集
    dataset = Dataset(args.dataset, args.data_dir, args.data_npy_dir, args.n_clients, 
                      args.sample_strategy, args.dirichlet_alpha, args.trn_bsz, 
                      args.trn_time, args.tst_time, args.balance)
    clnt_x, clnt_y, tst_x, tst_y, classes = dataset.get_data()
    args.classes = classes

    # for idx in range(len(clnt_y)):
    #     permutation = np.random.permutation(len(clnt_y[idx]))
    #     print(permutation)
    #     clnt_x[idx] = clnt_x[idx][permutation]
    #     clnt_y[idx] = clnt_y[idx][permutation]
    
    def show_statistics(dataset, args):
        num_clients, num_classes, dataset_name = args.n_clients, args.classes, args.dataset
        show_statis(dataset, num_clients, num_classes, dataset_name, train=True)
        show_statis(dataset, num_clients, num_classes, dataset_name, train=False)
    
    # show_statis(dataset, args.n_clients, args.classes, args.dataset, train=True)
    # show_statis(dataset, args.n_clients, args.classes, args.dataset, train=False)
    return dataset, {"clnt_x": clnt_x, "clnt_y": clnt_y, "tst_x": tst_x, "tst_y": tst_y}


def initialize_single_client(args, data, devices, graph):
    
    rank = int(os.environ['LOCAL_RANK'])
    clnt_x_rank, clnt_y_rank = data['clnt_x'][rank], data['clnt_y'][rank]
    tst_x_rank, tst_y_rank = data['tst_x'][rank], data['tst_y'][rank]
 
    trn_bs_dict = {10: 256, 20: 144, 30: 72}
    if args.balance == False:
        trn_bs = len(clnt_x_rank) // args.trn_time; tst_bs = len(tst_x_rank) // args.tst_time
    else:
        trn_bs = trn_bs_dict[args.n_clients]; tst_bs = args.tst_bsz
    train_loader = Partition_Loader(clnt_x_rank, clnt_y_rank, trn_bs, train=True).loader
    test_loader = Partition_Loader(tst_x_rank, tst_y_rank, tst_bs, train=False).loader
    
    dist.init_process_group(backend="gloo", rank=rank, world_size=args.n_clients)
    client = Client(args, rank, devices, data, graph)
    client.run(train_loader, test_loader, trn_bs)


def main():
    args = parse_arguments()    # load paramters
    devices, _ = configure_environment(args)  # set divices
    _, data = initialize_dataset(args)    # initialize_dataset
    graph = Graph(args.topology, args.n_clients, args.avg, args.reg)  # get network graph
    initialize_single_client(args, data, devices, graph)    # set clients and run


if __name__ == "__main__":
    main()


