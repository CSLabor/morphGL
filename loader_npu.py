from MorphGL.utils import get_logger
mlog = get_logger()

import torch
import numpy as np

def load_npu_data():
    # a placeholder which prepares corresponding data for the dummy NPU operators
    feat_dim = 256
    num_classes = 100
    train_idx = torch.arange(1024*1000) # bs=1024, 1000 batches
    return feat_dim, num_classes, train_idx
    
def partition_train_idx(all_train_idx, ratio=0.5):
    """
    return: CPU_train_idx, GPU_train_idx
    """
    temp_train_idx = all_train_idx[torch.randperm(all_train_idx.shape[0])]
    sep = int(all_train_idx.shape[0] * ratio)
    mlog(f"split into two part, salient {sep} : dgl {all_train_idx.shape[0]-sep}")
    return temp_train_idx[:sep], temp_train_idx[sep:]
