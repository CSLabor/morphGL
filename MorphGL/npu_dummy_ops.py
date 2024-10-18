from MorphGL.utils import get_logger
from .iterators import Blank_iter
from typing import Iterator
mlog = get_logger()

import time
import torch
import torch_npu

def npu_nn_wrapper(model_name, in_size, hidden_size, out_size, num_layers):
    mlog(f"build a MLP as a model placeholder")
    layers = torch.nn.ModuleList()
    layers.append(torch.nn.Linear(in_size, hidden_size))
    for _ in range(num_layers-2):
        layers.append(torch.nn.Linear(hidden_size, hidden_size))
    layers.append(torch.nn.Linear(hidden_size, out_size))
    return layers

def prepare_batching_cpu(train_idx, batch_size, feat_dim):
    if train_idx.shape[0] == 0:
        return Blank_iter()
    return CPU_Dummy_Batcher(train_idx, batch_size, feat_dim)

def prepare_batching_npu(train_idx, batch_size, feat_dim):
    if train_idx.shape[0] == 0:
        return Blank_iter()
    return NPU_Dummy_Batcher(idx, batch_size, feat_dim)
    
class CPU_Dummy_Batcher(Iterator):
    def __init__(self, idx, batch_size, feat_dim, per_batch_time=0.40):
        self.per_batch_time = per_batch_time # time unit: s
        self.last_generation_ts = 0
        self.pos = 0
        self.dummy_batch = torch.rand(200*1024**2/feat_dim//4, feat_dim)
        self._idx = idx
        self.bs = batch_size
        self.length = math.ceil(idx.shape[0] / self.bs)

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, new_idx: torch.Tensor):
        self._idx = new_idx
        self.length = math.ceil(self._idx.shape[0] / self.bs)

    def __iter__(self):
        self.last_generation_ts = time.time()
        self.pos = 0
        return self
    
    def __next__(self):
        if self.pos == self.length:
            raise StopIteration
        self.pos += 1
        cur_ts = time.time()
        if cur_ts - self.last_generation_ts >= self.per_batch_time:
            self.last_generation_ts += self.per_batch_time
        else:
            time.sleep(self.last_generation_ts + self.per_batch_time - cur_ts)
            self.last_generation_ts = time.time()
        return self.dummy_batch

    def try_one(self):
        if self.pos == self.length:
            raise StopIteration
        cur_ts = time.time()
        if cur_ts - self.last_generation_ts >= self.per_batch_time:
            # successful try
            self.last_generation_ts += self.per_batch_time
            self.pos += 1
            return self.dummy_batch
        else:
            return None

    def __len__(self):
        return self.length

class NPU_Dummy_Batcher(Iterator):
    def __init__(self, idx, batch_size, feat_dim):
        self.device = torch.device("npu:0")
        self.idx = idx
        self.bs = batch_size
        self.length = math.ceil(idx.shape[0]/self.bs)
        self.dummy_batch = torch.rand(200*1024**2/feat_dim//4, feat_dim).to(self.device)
        self.pos = 0
        # for placeholder computation
        self.A = torch.rand(8000, 8000).to(self.device)
        self.B = torch.rand(8000, 8000).to(self.device)
        self.iter_num = 10

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, new_idx: torch.Tensor):
        self._idx = new_idx.npu()
        self.length = math.ceil(self._idx.shape[0] / self.bs)

    def _fake_compute(self):
        for _ in range(self.iter_num):
            self.A = self.A @ self.B
            self.A = self.A / self.A.sum()

    def __iter__(self):
        self.pos = 0
        return self
    
    def __next__(self):
        if self.pos == self.length:
            raise StopIteration
        self.pos += 1
        self._fake_compute()
        return self.dummy_batch

    def __len__(self):
        return self.length
