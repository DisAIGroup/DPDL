import os
import time
import numpy as np
import pandas as pd
import torch
import random
import copy
import pickle
import logging
import argparse
import json
import math
from math import ceil
from scipy import sparse
from collections import Counter
from collections import defaultdict
from typing import List, Any, Dict, Tuple, Optional
import datetime
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
import torch.nn.functional as F
from torch.optim import LBFGS
import torch.distributed as dist
from torch.autograd import Variable
from torch.multiprocessing import Process
from torch.multiprocessing import spawn
import torch.multiprocessing as mp
from itertools import islice
from models import *
from utils.dataset import Dataset
from utils.partitioner import Partition, Partition_Loader, show_statis
from utils.graph_set import Graph
from utils.util import accuracy, precision, flatten_tensors, unflatten_tensors
from dpdl_trainer import Trainer_dpdl


DEFAULT_CONFIG = dict(lr_decay=True, scoring_choice='loss')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Client(object):
    def __init__(self, args, rank, devices, data, graph):
        self.args = args
        self.rank = rank
        self.data = data
        self.graph = graph
        self.device = devices[rank % len(devices)]  # Assign device based on rank

    def _validate_config(config):
        for key in DEFAULT_CONFIG.keys():
            if config.get(key) is None:
                config[key] = DEFAULT_CONFIG[key]
        for key in config.keys():
            if DEFAULT_CONFIG.get(key) is None:
                raise ValueError(f'Deprecated key in config dict: {key}!')
        return config


    def setup_logger(self):
        """Setup a logger for the client."""
        logger = logging.getLogger(f"client_{self.rank}")
        logger.setLevel(logging.INFO)
        log_dir = f"./log/"
        if not os.path.exists(log_dir) and self.rank == 0:
            os.makedirs(log_dir)
        dist.barrier()
        log_file = log_dir + f"client_{self.rank}.log"
        with open(log_file, 'w'):
            pass
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger


    def load_model(self):
        """Load the model based on the architecture."""
        if 'resnet' in self.args.arch.lower():
            model = resnet(num_classes=self.args.classes, arch=self.args.arch, dataset=self.args.dataset.lower(), 
                           norm_type=self.args.normtype, groups=2)
        elif 'vgg' in self.args.arch.lower():
            model = vgg(num_classes=self.args.classes, arch=self.args.arch, dataset=self.args.dataset.lower(), 
                        norm_type=self.args.normtype, groups=2)
        elif self.args.arch.lower() == 'mobilenet':
            model = mobilenet(num_classes=self.args.classes, dataset=self.args.dataset.lower(), 
                              norm_type=self.args.normtype, groups=2)
        elif self.args.arch.lower() == 'lenet':
            model = lenet(arch=self.args.arch, dataset=self.args.dataset.lower())
        elif self.args.arch.lower() == 'cnn':
            model = cnn(arch=self.args.arch, dataset=self.args.dataset.lower())
        elif self.args.arch.lower() == 'vit':
            model = vit(num_classes=self.args.classes)
        else:
            raise NotImplementedError(f"model {self.args.arch.lower()} is not implemented")
        return model.to(self.device)

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        trn_bs: int,
        epoch: int
    ) -> Tuple[List[float], List[float], float]:
        self.model.train()
        batch_loss = []
        time_meter = AverageMeter()
        end = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > self.args.trn_time  - 1:
                break
            data, target = data.to(self.device), target.to(self.device)
            loss = self.trainer.train(data, target, epoch)
            batch_loss.append(round(loss.item(), 4))
            time_meter.update(time.time() - end)
            end = time.time()
            del data
            del target
            torch.cuda.empty_cache()
        return batch_loss, time_meter.avg


    def run(
        self,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        trn_bs: int
    ) -> None:
        self.logger = self.setup_logger()
        self.model = self.load_model()
        self.trainer = Trainer_dpdl(self.args, self.rank, self.graph, self.model, trn_bs, self.device, self.args.lr, self.args.ct, self.args.epsilon, self.args.momentum)
        self.logger.info(f"Starting")
        trn_loss: List[List[float]] = []
        for epoch in range(self.args.epochs):
            trn_loss_epoch, trn_time_epoch = self.train(train_loader, self.args.trn_bsz, epoch)
            trn_loss.append(trn_loss_epoch)
            self.trainer.lr_scheduler.step()
            self.logger.info(
                f'''Train Epoch: {epoch}/{self.args.epochs} | Avg Training Loss: {np.mean(trn_loss_epoch):.4f} | Batch avg time: {trn_time_epoch:.3f}'''
            )
        data_dict = self.combine_loss(trn_loss)
        self.save_loss(data_dict)
        self.save_checkpoint()


    def combine_loss(
        self,
        lcl_loss: List[List[float]]
    ) -> Dict[str, List[float]]:
        """Collect and aggregate training metrics across clients."""
        def gather_2d_list(data: List[List[float]]) -> List[List[float]]:
            tensor_data = torch.tensor(data, dtype=torch.float64).to(self.device)
            all_data = [torch.zeros_like(tensor_data) for _ in range(self.args.n_clients)]
            dist.all_gather(all_data, tensor_data)
            return [client_data.cpu().tolist() for client_data in all_data]
        
        def gather_1d_list(data: List[float]) -> List[List[float]]:
            tensor_data = torch.tensor(data).to(self.device)
            all_data = [torch.zeros_like(tensor_data) for _ in range(self.args.n_clients)]
            dist.all_gather(all_data, tensor_data)
            return [client_data.cpu().tolist() for client_data in all_data]

        # Collect client data
        glb_loss = gather_2d_list(lcl_loss)
        dist.barrier()
        if self.rank != 0:
            return {}

        # Build result dictionary
        data_dict = {
            'dataset': self.args.dataset,
            'methods': self.args.methods,
            'arch': self.args.arch,
            'nodes': self.args.n_clients,
            'topology': self.args.topology,
            'dirichlet_alpha': self.args.dirichlet_alpha,
            'lr': self.args.lr,
            'epsilon': self.args.epsilon,
        }
        num_clients = len(glb_loss)
        num_epochs = len(glb_loss[0])
        
        # 1. Client batch-level loss columns
        for client_idx in range(num_clients):
            client_loss = glb_loss[client_idx]
            flat_loss = []
            for epoch_losses in client_loss:
                flat_loss.extend([round(x,4) for x in epoch_losses])
            data_dict[f'batch_loss_{client_idx}'] = json.dumps(flat_loss)
        # 2. Client epoch-level average loss
        for client_idx in range(num_clients):
            epoch_averages = [round(np.mean(epoch_loss),4) for epoch_loss in glb_loss[client_idx]]
            data_dict[f'epoch_loss_{client_idx}'] = json.dumps(epoch_averages)
        # 3. Global batch-level metrics
        global_batch_loss = []
        for epoch_idx in range(num_epochs):
            for batch_idx in range(len(glb_loss[0][epoch_idx])):
                client_losses = []
                for client in range(num_clients):
                    if batch_idx < len(glb_loss[client][epoch_idx]):
                        client_losses.append(glb_loss[client][epoch_idx][batch_idx])
                global_batch_loss.append(round(np.mean(client_losses),4))
        data_dict['batch_loss'] = json.dumps(global_batch_loss)
        # 4. Global epoch-level metrics
        global_epoch_loss = []
        for epoch_idx in range(num_epochs):
            epoch_loss = np.mean([np.mean(glb_loss[client][epoch_idx]) for client in range(num_clients)])
            global_epoch_loss.append(round(epoch_loss,4))
        data_dict['epoch_loss'] = json.dumps(global_epoch_loss)
        return data_dict

    
    def save_loss(
        self,
        data_dict: Dict[str, List[float]]
    ) -> None:
        """Save training metrics to CSV with structured columns."""
        if self.rank != 0:
            return
        # DataFrame
        param_order = [
            'dataset', 'methods', 'arch', 'nodes', 'topology', 
            'dirichlet_alpha', 'lr', 'epsilon'
        ]
        metric_cols = [col for col in data_dict.keys() if col not in param_order]
        column_order = param_order + metric_cols
        df = pd.DataFrame([data_dict])[column_order]
        # Save CSV
        csv_var = f"{self.args.dataset}_{self.args.arch}_{self.args.methods}"
        csv_path = f"./outputs/{csv_var}.csv"
        if os.path.exists(csv_path):
            existing_df = pd.read_csv(csv_path)
            df = pd.concat([existing_df, df]).drop_duplicates(
                subset=param_order, keep='last'
            )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        print(csv_path)
        df.to_csv(csv_path, index=False)


    def save_checkpoint(self):
        """save the trained avg model"""
        state_dict = self.model.cpu().state_dict()
        avg_state_dict = {}
        for key, param in state_dict.items():
            param_tensor = param.data.to(self.device)
            all_params = [torch.zeros_like(param_tensor) for _ in range(self.args.n_clients)]
            dist.all_gather(all_params, param_tensor)
            avg_param = torch.mean(torch.stack(all_params), dim=0)
            avg_state_dict[key] = avg_param
        if self.rank == 0:
            check_dir = f"./checkpoints/"
            if not os.path.exists(check_dir):
                os.makedirs(check_dir)
            check_file = check_dir + f'model.pth.tar'
            self.model.load_state_dict(avg_state_dict)
            torch.save(self.model.state_dict(), check_file)


