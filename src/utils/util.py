
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy import io
import torch
from torch.utils import data
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from IPython.core.debugger import set_trace
import scipy.io as sio
from itertools import combinations
from scipy.special import gamma
from scipy.special import loggamma
from scipy import stats
from scipy.optimize import minimize
# from sklearn import svm
# from sklearn import mixture
import random
import datetime
import collections
import logging
import math
import sys
import copy

class NCE(object):

    def __init__(self, u, v) -> None:
        self.u = u
        self.v = v
        self.loss = self.NCE_loss(u,v)

    def cosine_sim(self, x, y):
        x = self.normalize(x, axis=-1)
        y = self.normalize(y, axis=-1)
        cosine = torch.mm(x, y.permute(1, 0))
        return cosine

    def pairwise_obj(self, u, v):
        f = lambda x: torch.exp(x / self.tau)
        inter_view = f(self.cosine_sim(u, v))
        intra_view = f(self.cosine_sim(u, u))
        positive_l = inter_view.diag()  # 正例是对应相同点在两个视图的嵌入的cos近似度（u_i和v_i）
        all_l = inter_view.sum(dim=1) + intra_view.sum(dim=1) - intra_view.diag()  # 两次加和算了两次中心部位
        return -torch.log(positive_l / all_l)

    def NCE_loss(self, u, v):
        u = self.act(self.fc(u))
        v = self.act(self.fc(v))
        loss_obj1 = self.pairwise_obj(u, v)
        loss_obj2 = self.pairwise_obj(v, u)
        loss_obj = torch.mean(0.5 * (loss_obj1 + loss_obj2))
        return loss_obj


def flatten_tensors(tensors):
    """
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)

def unflatten(flat, tensor):
  
   offset=0
   numel = tensor.numel()
   output = (flat.narrow(0, offset, numel).view_as(tensor))
   return output


def group_by_dtype(tensors):
    """
    Returns a dict mapping from the tensor dtype to a list containing all
    tensors of that dtype.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
    """
    tensors_by_dtype = collections.defaultdict(list)
    for tensor in tensors:
        tensors_by_dtype[tensor.dtype].append(tensor)
    return tensors_by_dtype


def communicate(tensors, communication_op):
    """
    Communicate a list of tensors.
    Arguments:
        tensors (Iterable[Tensor]): list of tensors.
        communication_op: a method or partial object which takes a tensor as
            input and communicates it. It can be a partial object around
            something like torch.distributed.all_reduce.
    """
    tensors_by_dtype = group_by_dtype(tensors)
    for dtype in tensors_by_dtype:
        flat_tensor = flatten_tensors(tensors_by_dtype[dtype])
        communication_op(tensor=flat_tensor)
        for f, t in zip(unflatten_tensors(flat_tensor, tensors_by_dtype[dtype]),
                        tensors_by_dtype[dtype]):
            t.set_(f)


def is_power_of(N, k):
    """
    Returns True if N is a power of k
    """
    assert isinstance(N, int) and isinstance(k, int)
    assert k >= 0 and N > 0
    if k == 0 and N == 1:
        return True
    if k in (0, 1) and N != 1:
        return False

    return k ** int(round(math.log(N, k))) == N


def create_process_group(ranks):
    """
    Creates and lazy intializes a new process group. Assumes init_process_group
    has already been called.
    Arguments:
        ranks (list<int>): ranks corresponding to the processes which should
            belong the created process group
    Returns:
        new process group
    """
    initializer_tensor = torch.Tensor([1])
    if torch.cuda.is_available():
        initializer_tensor = initializer_tensor.cuda()
    new_group = dist.new_group(ranks)
    dist.all_reduce(initializer_tensor, group=new_group)
    return new_group

def quantize_tensor(out_msg, comp_fn, quantization_level, is_biased = True):
    #print(quantization_level)
    out_msg_comp = copy.deepcopy(out_msg)
    quantized_values = comp_fn.compress(out_msg_comp, None, quantization_level, is_biased)
    #print(out_msg-quantized_values)
    
    return quantized_values

def quantize_layerwise(out_msg, comp_fn, quantization_level, is_biased = True):
    #print(quantization_level)
    quantized_values = []
    
    for param in out_msg:
        # quantize.
        #print(param.size())
        _quantized_values = comp_fn.compress(param, None, quantization_level, is_biased)
        quantized_values.append(_quantized_values)

    return quantized_values

def sparsify_layerwise(out_msg, comp_fn, comp_op, compression_ratio, is_biased=True):
    selected_values  = []
    selected_indices = []
    selected_shapes  = []
    
    for param in out_msg:
        #print(param.shape, param.size(0))
#        if param.size(0)==1:
#            ratio = 0
#        else:
#            ratio=compression_ratio
        #print(ratio)
            
        p = flatten_tensors(param)
        values, indices = comp_fn.compress(p, comp_op, compression_ratio, is_biased)
        selected_values.append(values)
        selected_indices.append(indices)
        selected_shapes.append(len(values)) # should be same for all nodes, length of compressed tensor at each layer
        
    flat_values  = flatten_tensors(selected_values)
    flat_indices = flatten_tensors(selected_indices)
    comp_msg     = torch.cat([flat_values, flat_indices.type(flat_values.dtype)])
    return comp_msg, selected_shapes

def unsparsify_layerwise(msg, shapes, ref_param):
    # ref_msg is the out_msg from sparsify_layerwise.....need it just for the shape
    #sparse_msg = torch.zeros_like(ref_param)
    out_msg    = []
    val_size   = int(len(msg)/2)
    values     = msg[:val_size]
    indices    = msg[val_size:]
    indices    = indices.type(torch.cuda.LongTensor)
    
    pointer = 0
    i       = 0
    for ref in ref_param:
        param = torch.zeros_like(ref)
        p = flatten_tensors(param)
        layer_values  = values[pointer:(pointer+shapes[i])]
        layer_indices = indices[pointer:(pointer+shapes[i])]
        p[layer_indices] = layer_values.type(ref.data.dtype)
        layer_msg        = unflatten(p, ref)
        #print(layer_msg.size(), ref.size())
        out_msg.append(layer_msg)
        pointer  += shapes[i]
        i        += 1
        
    return out_msg
        

def precision(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy(output, target):
    # p = torch.softmax(output, dim=1).argmax(1)
    # res = p.eq(target).sum().item()
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum()
    accuracy = 100.0 * correct / target.size(0)
    return accuracy
