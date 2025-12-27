import copy
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
from torch.multiprocessing import Process, spawn
import torch.distributed as dist
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from utils.util import accuracy, precision, flatten_tensors, unflatten_tensors

import os
import copy
import numpy as np
from math import ceil
import random
import pickle
from collections import Counter
from collections import defaultdict
import time
import datetime
from typing import Any

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import LBFGS
import torch.distributed as dist
from torch.autograd import Variable
from torch.multiprocessing import Process
from torch.multiprocessing import spawn


class Trainer_Base:
    def __init__(
        self,
        args: dict,
        rank: int,
        graph: Any,
        model: nn.Module,
        trn_bs: int,
        device: str,
        lr: float,
        ct: float,
        epsilon: float
    ):
        self.args = args
        self.rank = rank
        self.graph = graph
        self.size = graph.size
        self.model = model  
        self.trn_bs = trn_bs
        self.epsilon = epsilon
        self.lr = lr
        self.ct = ct
        self.device = device
        self.sigma = (2 * 4.2 * self.ct) / (self.trn_bs * self.epsilon * 8)
        
        self.groups, self.dist_groups, self.p2p_groups = self.graph.get_comm_group()
        self.neighbor_rank, self.neighbor_weight, self.neighbor, self.dsm = self.graph.get_neighbor_info(self.rank)
        self.neighbor_rank_ns = [x for x in self.neighbor_rank if x != self.rank]
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, gamma=0.995, step_size=3)
        self.sample_rate = self.trn_bs * len(self.dsm) / 5e5

    def local_train(
        self, 
        model: nn.Module, 
        data: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model.zero_grad()
        output = model(data)
        loss = self.criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.ct
        )
        return output, loss

    def train(self):
        raise NotImplementedError

    def test(
        self, 
        data: torch.Tensor, 
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            input_var = data.to(self.device)
            target_var = target.reshape(-1).to(self.device)
            output = self.model(input_var)
            loss = self.criterion(output, target_var)
            acc = accuracy(output.data, target_var)
            return loss.float(), acc

    def add_dp(
        self, 
        pos: torch.Tensor, 
        clipping: bool = True
    ) -> torch.Tensor:
        clipper = pos.norm(2)
        clipped_pos = pos * min(1, self.ct / clipper) if clipping else pos
        noise = torch.normal(0, self.sigma, size=pos.size()).to(self.device)
        return clipped_pos + noise

    def add_clip(
        self, 
        pos: torch.Tensor, 
        clipping: bool = True
    ) -> torch.Tensor:
        clipper = pos.norm(p=2)
        return pos * min(1, self.ct / clipper) if clipping else pos

    def cross_model(self) -> Dict[int, List[torch.Tensor]]:
        received_params = {nb_rank: [] for nb_rank in self.neighbor_rank_ns}
        for param in self.model.parameters():
            received_params_buf = {
                idx: param.data.clone().cpu() if idx == self.rank else torch.zeros_like(param.data).cpu()
                for idx in self.neighbor_rank
            }
            for src_rank, group in enumerate(self.groups):
                if self.rank in group:
                    dist.broadcast(
                        tensor=received_params_buf[src_rank],
                        src=src_rank,
                        group=self.dist_groups[src_rank]
                    )
            for src in self.neighbor_rank_ns:
                received_params[src].append(received_params_buf[src].to(self.device))
        received_params_buf.clear()
        torch.cuda.empty_cache()
        return received_params

    def cross_noised_model(self) -> Dict[int, List[torch.Tensor]]:
        received_params = {nb_rank: [] for nb_rank in self.neighbor_rank_ns}
        param_buf = copy.deepcopy(self.model)
        for param in param_buf.parameters():
            param.data = self.add_dp(param.data, False)
        for param in param_buf.parameters():
            received_params_buf = {
                idx: param.data.clone() if idx == self.rank else torch.zeros_like(param.data)
                for idx in self.neighbor_rank
            }
            for src_rank, group in enumerate(self.groups):
                if self.rank in group:
                    dist.broadcast(
                        tensor=received_params_buf[src_rank],
                        src=src_rank,
                        group=self.dist_groups[src_rank]
                    )
            for src in self.neighbor_rank_ns:
                received_params[src].append(received_params_buf[src])
        return received_params

    def combine_model(self, received_params: Dict[int, List[torch.Tensor]]):
        for i, param in enumerate(self.model.parameters()):
            param.data.mul_(self.neighbor[self.rank])
            for src_id in self.neighbor_rank_ns:
                param.data.add_(received_params[src_id][i] * self.neighbor[src_id])

    def flatten_gradients(
        self, 
        model: nn.Module
    ) -> Dict[int, torch.Tensor]:
        grad = {}
        for src_id in self.neighbor_rank_ns:
            gradient_buffer = {
                name: param.grad.data 
                for name, param in model[src_id].named_parameters() 
                if param.requires_grad
            }
            grad_buf = []
            for g in gradient_buffer.values():
                grad_buf.append(g)
            grad[src_id] = flatten_tensors(grad_buf)
        return grad

    def unflatten_gradients(
        self, 
        grad: Dict[int, torch.Tensor], 
        ref_buf: nn.Module
    ) -> List[List[torch.Tensor]]:
        neighbor_grad = [[] for _ in self.neighbor_rank_ns]
        ref_params = list(ref_buf.parameters())
        for src_idx, src_rank in enumerate(self.neighbor_rank_ns):
            unflat_tensor = unflatten_tensors(grad[src_rank], ref_params)
            neighbor_grad[src_idx] = list(unflat_tensor)
        return neighbor_grad

    def _save_model(self, var, epoch=0):
        tmp_path_dir = f'./model/'
        if not os.path.exists(tmp_path_dir): os.makedirs(tmp_path_dir)
        torch.save(var, tmp_path_dir + f"/{epoch}.pth")

    def _save_grad(self, var, epoch=0):
        tmp_path_dir = f'./grad/'
        if not os.path.exists(tmp_path_dir): os.makedirs(tmp_path_dir)
        torch.save(var, tmp_path_dir + f"/{epoch}.pt")

    def _save_data(self, var, epoch=0):
        tmp_path_dir = f'./data/'
        if not os.path.exists(tmp_path_dir): os.makedirs(tmp_path_dir)
        torch.save(var, tmp_path_dir + f"/{epoch}.pt")
        


class Trainer_CG(Trainer_Base):
    def __init__(
        self,
        args: dict,
        rank: int,
        graph: Any,
        model: nn.Module,
        trn_bs: int,
        device: str,
        lr: float,
        ct: float,
        epsilon: float,
        momentum: float
    ):
        super().__init__(args, rank, graph, model, trn_bs, device, lr, ct, epsilon)
        self.weight_decay = 5e-4
        self.momentum = momentum
        self.nesterov = False
        self.qgm = False
        self.momentum_buff = [
            torch.zeros_like(param.data) 
            for param in model.parameters()
        ]

    def train(self):
        raise NotImplementedError

    def cal_cross_gradients(self, received_params, input_var, target_var, epoch, batch_idx=-1):
        neighbor_params = {idx: copy.deepcopy(self.model) for idx in self.neighbor_rank_ns}
        for dst_id in self.neighbor_rank_ns:
            for i, param in enumerate(neighbor_params[dst_id].parameters()):
                param.data = received_params[dst_id][i]
            neighbor_params[dst_id].train()
            neighbor_params[dst_id].zero_grad()
            output_buf = neighbor_params[dst_id](input_var)
            loss_buf = self.criterion(output_buf, target_var)
            # backward
            loss_buf.backward()
        return neighbor_params

    def cross_multi_gradients(
        self, 
        send_gradient: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        out_msg_buffer = []
        placeholder = torch.zeros_like(next(iter(send_gradient.values()))).cpu()
        send_gradient = {k: v.cpu() for k, v in send_gradient.items()}
        for dst_rank in self.neighbor_rank_ns:
            req = dist.broadcast(
                send_gradient[dst_rank], 
                src=self.rank, 
                group=self.p2p_groups[self.rank][dst_rank], 
                async_op=True
            )
            out_msg_buffer.append((req, send_gradient[dst_rank]))
        in_msg_buffer = {}
        for src_rank in self.neighbor_rank_ns:
            dist.broadcast(
                placeholder,
                src=src_rank, 
                group=self.p2p_groups[src_rank][self.rank]
            )
            in_msg_buffer[src_rank] = placeholder.clone().to(self.device)
        return in_msg_buffer

    
    def update_gradients(self):
        for param in self.model.parameters():
            if self.weight_decay != 0:
                param.grad.data.add_(param.data, alpha=self.weight_decay)
        if self.qgm:
            for p, p_prev, buf in zip(self.model.parameters(), self.prev_params, self.momentum_buff):
                buf.mul_(self.momentum).add_(p_prev.data - p.data, alpha=(1 - self.momentum)/self.lr)
                mom_buff = buf.clone()
                mom_buff.mul_(self.momentum).add_(p.grad.data)
                if self.nesterov:
                    p.grad.data.add_(mom_buff, alpha=self.momentum)
                else:
                    p.grad.data.copy_(mom_buff)
            for p, p_prev in zip(self.model.parameters(), self.prev_params):
                p_prev.data.copy_(p.data)
        else:
            for p, buf in zip(self.model.parameters(), self.momentum_buff):
                buf.mul_(self.momentum).add_(p.grad.data)
                if self.nesterov:
                    p.grad.data.add_(buf, alpha=self.momentum)
                else:
                    p.grad.data.copy_(buf)

    def cross_momentum(self) -> Dict[int, List[torch.Tensor]]:
        received_momentum = {nb_rank: [] for nb_rank in self.neighbor_rank_ns}
        for mome in self.momentum_buff:
            received_mome_buf = {
                idx: mome.clone().cpu() if idx == self.rank else torch.zeros_like(mome).cpu()
                for idx in self.neighbor_rank
            }
            for src_rank, group in enumerate(self.groups):
                if self.rank in group:
                    dist.broadcast(
                        tensor=received_mome_buf[src_rank],
                        src=src_rank,
                        group=self.dist_groups[src_rank]
                    )
            for src in self.neighbor_rank_ns:
                received_momentum[src].append(received_mome_buf[src].to(self.device))
        return received_momentum

    def combine_momentum(self, received_momentum: Dict[int, List[torch.Tensor]]):
        for i, mome in enumerate(self.momentum_buff):
            mome.mul_(self.neighbor[self.rank])
            for src_id in self.neighbor_rank_ns:
                mome.add_(received_momentum[src_id][i] * self.neighbor[src_id])


class Trainer_dpdl(Trainer_CG):

    def __init__(self, args, rank, graph, model, trn_bs, device, lr, ct, epsilon, momentum):
        super().__init__(args, rank, graph, model, trn_bs, device, lr, ct, epsilon, momentum)
        self.ref_buf = copy.deepcopy(self.model)
        self.base_sigma = 96 * self.sample_rate / self.epsilon
        self.alpha_j = 1.0
        self.alpha_i = 1.5
        self.coef = ( 1 / np.sqrt(self.neighbor[self.rank]) + self.size * self.alpha_i / self.alpha_j) / \
                    (np.sqrt(sum([ 1/self.neighbor[self.neighbor_rank[ni]] for ni in range(len(self.neighbor_rank))])))
        self.sigma = self.base_sigma * self.coef
        self.neighbor_rank_ts = torch.tensor(self.graph.get_neighbor_info(self.rank)[0], \
                                             dtype=torch.float32).to(self.device)
        self.neighbor_ts = torch.tensor([torch.tensor(value).to(self.device) \
                                    for key, value in self.graph.get_neighbor_info(self.rank)[2].items()])
    
    def cross_momentum_and_model(self):

        received_params = {nb_rank: [] for nb_rank in self.neighbor_rank_ns}
        received_momentum = {nb_rank: [] for nb_rank in self.neighbor_rank_ns}
        for idx, (param, mome) in enumerate(zip(self.model.parameters(), self.momentum_buff)):
            received_buf = {idx: torch.stack([copy.deepcopy(param.data), copy.deepcopy(mome)]).cpu() if idx == self.rank 
                            else torch.stack([torch.zeros_like(param.data), torch.zeros_like(mome)]).cpu() for idx in self.neighbor_rank}
            for src_rank in range(self.graph.size):
                group = self.groups[src_rank]
                dist_group = self.dist_groups[src_rank]
                if self.rank in group:
                    dist.broadcast(tensor=received_buf[src_rank], src=src_rank, group=dist_group)
            for src in self.neighbor_rank_ns:
                received_params[src].append(received_buf[src][0].to(self.device))
                received_momentum[src].append(received_buf[src][1].to(self.device))
        return received_params, received_momentum

    def train_sample(self, model, input_var, target_var):
        grad_cp = []
        pieces = int(self.trn_bs / 4)
        for idx in range(0, input_var.shape[0],pieces):
            output, loss = self.local_train(model, input_var[idx:idx+pieces], target_var[idx:idx+pieces])
            grad_buf = []
            for i, param in enumerate(model.parameters()):
                grad_buf.append(copy.deepcopy(param.grad.data))
            grad_cp.append(flatten_tensors(grad_buf))
        grad_sum = torch.sum(torch.stack(grad_cp, dim=0), dim=0)
        grad_cp = torch.mean(torch.stack(grad_cp, dim=0), dim=0)
        grad_ns = grad_sum * pieces / self.trn_bs + torch.normal(mean=0.0, std=self.sigma * self.ct, size=grad_sum.size()).to(self.device) / self.trn_bs
        return grad_ns, unflatten_tensors(copy.deepcopy(grad_ns), grad_buf), unflatten_tensors(copy.deepcopy(grad_cp), grad_buf)
    
    def cal_cross_gradients_sample(self, received_params, input_var, target_var, epoch=-1, batch_idx=-1):
        neighbor_params = {idx: copy.deepcopy(self.model) for idx in self.neighbor_rank_ns}
        neighbor_grads = {idx: [] for idx in self.neighbor_rank_ns}
        for dst_id in self.neighbor_rank_ns: 
            # selfgrad
            for i, param in enumerate(neighbor_params[dst_id].parameters()):
                param.data = received_params[dst_id][i]
            neighbor_params[dst_id].train()
            neighbor_grads[dst_id], grad_to_save,_ = self.train_sample(neighbor_params[dst_id], input_var, target_var)
        out_msg_buffer = []
        placeholder = torch.zeros_like(next(iter(neighbor_grads.values()))).cpu()
        neighbor_grads = {k: v.cpu() for k, v in neighbor_grads.items()}
        dist.barrier()
        for dst_rank in self.neighbor_rank_ns:
            req = dist.broadcast(tensor=neighbor_grads[dst_rank], src=self.rank, 
                                    group=self.p2p_groups[self.rank][dst_rank], async_op=True)
            out_msg_buffer.append((req, neighbor_grads[dst_rank]))
        in_msg_buffer = {}
        for src_rank in self.neighbor_rank_ns:
            dist.broadcast(tensor=placeholder, src=src_rank, group=self.p2p_groups[src_rank][self.rank])
            in_msg_buffer[src_rank] = copy.deepcopy(placeholder.to(self.device))
        neighbor_grads.clear()
        torch.cuda.empty_cache()
        return self.unflatten_gradients(in_msg_buffer, self.ref_buf)


    def train(self, data, target, epoch, batch_idx=-1):

        self.model.train()
        self.model.zero_grad()
        self.ref_buf = copy.deepcopy(self.model)
        input_var = Variable(data).to(self.device)
        target_var = Variable(target.reshape(-1)).to(self.device)
        received_params = self.cross_model()
        _, self.self_grad_ns, self.self_grad_cp = self.train_sample(self.model, input_var, target_var)
        output = self.model(input_var)
        loss = self.criterion(output, target_var)
        for i, param in enumerate(self.model.parameters()):
            param.grad.data = self.self_grad_cp[i]
        neighbor_gradient = self.cal_cross_gradients_sample(received_params,input_var,target_var, epoch, batch_idx)
        dist.barrier()
        grad = self.calibration_programming(neighbor_gradient)
        for i, param in enumerate(self.model.parameters()):
            param.grad.data = grad[i]
        # update
        self.update_gradients()
        self.optimizer.step()
        dist.barrier()
        received_params, received_momentum = self.cross_momentum_and_model()
        # agg
        self.combine_momentum(received_momentum)
        self.combine_model(received_params)
        # measure loss
        output = output.float()
        loss = loss.float()
        return loss
    
    def compute_cosine_similarity(self, gi, gj):
        """similarity"""
        norm_gi = torch.norm(gi)
        norm_gj = torch.norm(gj)
        dot_product = torch.dot(gj, gi)
        return dot_product / (norm_gj * norm_gi)

    def compute_cosine_similarity_batch(self, gi, gj_batch):
        """
        gi: (1, grad_size)
        gj_batch: (N, grad_size)
        """
        dot_product = torch.matmul(gj_batch, gi.view(-1, 1))  # (N, 1) * (1, grad_size)
        norm_gi = torch.norm(gi, p=2, dim=0, keepdim=True)  # (1, 1)
        norm_gj_batch = torch.norm(gj_batch, p=2, dim=1, keepdim=True)  # (N, 1)
        cosine_similarity = dot_product / (norm_gi * norm_gj_batch)  # (N, 1)
        return cosine_similarity.view(-1)
    

    # def calibration_programming(self, neighbor_gradient):
    #     grad = []
    #     neighbor_weights = torch.cat((self.neighbor_ts[:self.rank], self.neighbor_ts[self.rank+1:], 
    #                             self.neighbor_ts[self.rank:self.rank+1]),dim=0).to(self.device)
    #     self_grad_list = []
    #     self_grad_ns_list = []
    #     grad_shapes = []
    #     for i, param in enumerate(self.model.parameters()):
    #         self_grad = param.grad.data.to(self.device)
    #         grad_shapes.append(self_grad.shape)
    #         self_grad_list.append(self_grad.flatten())
    #         self_grad_ns_list.append(self.self_grad_ns[i].flatten())
    #     gi = torch.cat(self_grad_list, dim=0)
    #     gi_ns = torch.cat(self_grad_ns_list, dim=0)
    #     neighbor_grads = []
    #     for ni in range(len(self.neighbor_rank_ns)):
    #         neighbor_grad_list = []
    #         for i, param in enumerate(self.model.parameters()):
    #             neighbor_grad_list.append(neighbor_gradient[ni][i].flatten())
    #         neighbor_grads.append(torch.cat(neighbor_grad_list, dim=0))
    #     grads = torch.cat((torch.stack(neighbor_grads, dim=0), gi_ns.unsqueeze(0)), dim=0).to(self.device)
    #     cos_sim = torch.sigmoid((-1) * self.compute_cosine_similarity_batch(gi, grads))
    #     # adjusted_grads_j = grads * ((1 / self.size) / torch.sqrt(neighbor_weights.view(-1, 1)))
    #     adjusted_grads_j = grads * torch.sqrt(neighbor_weights.view(-1, 1) / self.size) * self.alpha_j
    #     adjusted_grads_i = gi.expand(len(self.neighbor_rank_ns)+1, -1) * neighbor_weights.view(-1, 1) * cos_sim.view(-1, 1)  * self.alpha_i
    #     adjusted_grad = adjusted_grads_j.sum(dim=0) + adjusted_grads_i.sum(dim=0)
    #     grad = []
    #     start_idx = 0
    #     for shape in grad_shapes:
    #         param_size = torch.prod(torch.tensor(shape))
    #         param_grad = adjusted_grad[start_idx:start_idx + param_size]
    #         grad.append(param_grad.view(*shape))
    #         start_idx += param_size 
    #     return grad

    
    def calibration_programming(self, neighbor_gradient):
        grad = []
        neighbor_weights = torch.cat((self.neighbor_ts[:self.rank], self.neighbor_ts[self.rank+1:], 
                                self.neighbor_ts[self.rank:self.rank+1]),dim=0).to(self.device)
        for i, param in enumerate(self.model.parameters()):
            self_grad = param.grad.data.to(self.device)  # Ensure gradient is on the correct device
            grad_size = self_grad.size()
            gi = copy.deepcopy(self_grad).flatten()  # Flatten the gradient
            gi_ns = copy.deepcopy(self.self_grad_ns)[i].flatten()
            grads = torch.cat((torch.stack([neighbor_gradient[ni][i].flatten() \
                            for ni in range(len(self.neighbor_rank_ns))], dim=0), gi_ns.unsqueeze(0))).to(self.device)
            cos_sim = torch.sigmoid((-1) * self.compute_cosine_similarity_batch(gi, grads))
            # weighted broad
            adjusted_grads_j = grads * ((1 / self.size) / torch.sqrt(neighbor_weights.view(-1, 1))) * self.alpha_j
            # adjusted_grads_j =  grads * torch.sqrt(neighbor_weights.view(-1, 1) / self.size) * self.alpha_j
            adjusted_grads_i = gi.expand(len(self.neighbor_rank_ns)+1, -1) * neighbor_weights.view(-1, 1) * cos_sim.view(-1, 1) * self.alpha_i
            adjusted_grad = adjusted_grads_j.sum(dim=0) + adjusted_grads_i.sum(dim=0)
            grad.append(adjusted_grad.view(*grad_size))

        return grad
