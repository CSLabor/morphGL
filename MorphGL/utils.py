import torch
import random
import logging
import numpy as np
from typing import Iterator

def torch_to_device_wrapper(device):
    def func(tensor, non_blocking=False):
        return tensor.to(device, non_blocking=non_blocking)
    return func

def get_logger(file_path=None):
    if file_path:
        logging.basicConfig(
            format='%(asctime)-15s %(message)s',
            level=logging.INFO,
            filename=file_path,
            filemode='w'
        )
        print("Logs are being recorded at: {}".format(file_path))
    else:
        logging.basicConfig(
            format='%(asctime)-15s %(message)s',
            level=logging.CRITICAL
        )
    log = logging.getLogger(__name__).critical
    return log

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    import dgl
    dgl.seed(seed)
    dgl.random.seed(seed)

def truncate(int_, min_, max_):
    if int_ > max_:
        return max_
    if int_ < min_:
        return min_
    return int_

def get_seeds_list(num_batches, bs, train_idx):
    mlog = get_logger()
    seeds_list = []
    for _ in range(num_batches):
        idxs = torch.randint(0, train_idx.shape[0], (bs,))
        cur_seed = train_idx[idxs].to(train_idx.device)
        seeds_list.append(cur_seed)

    size = num_batches * bs * idxs.element_size() / (1024**3)
    mlog(f"get {num_batches} seeds, {size:.2f}GB on {train_idx.device}")

    return seeds_list

class Blank_iter(Iterator):
    def __init__(self):
        self.length = 0

    def __iter__(self):
        return self
    
    def __next__(self):
        raise StopIteration

    def __len__(self):
        return self.length
 
