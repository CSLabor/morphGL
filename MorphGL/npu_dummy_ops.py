from MorphGL.utils import get_logger
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

def prepare_batching_cpu(length=100, per_batch_time=0.10):
    return CPU_Dummy_Batcher(length, per_batch_time)

def prepare_batching_npu(length=100):
    return NPU_Dummy_Batcher(length)
    
class CPU_Dummy_Batcher(Iterator):
    def __init__(self, length=100, per_batch_time=0.10):
        # time unit: s
        self.length = length
        self.per_batch_time = per_batch_time
        self.last_generation_ts = 0
        self.pos = 0
        self.dummy_batch = torch.rand(100*1024**2//4)

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
    def __init__(self, length=100):
        self.length = length
        self.pos = 0
        self.device = torch.device("npu:0")
        self.dummy_batch = torch.rand(100).to(self.device)
        self.A = torch.rand(8000, 8000).to(self.device)
        self.B = torch.rand(8000, 8000).to(self.device)
        self.iter_num = 10

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
