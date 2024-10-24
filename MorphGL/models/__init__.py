import torch
from .gsg import SAGE
from .gat import GAT
from .gcn import GCN

def dgl_nn_wrapper(model_name, in_size, hidden_size, out_size, num_layers):
    if model_name == 'sage':
        return SAGE(in_size, hidden_size, out_size, num_layers)
    elif model_name == 'gcn':
        return GCN(in_size, hidden_size, out_size, num_layers)
    elif model_name == 'gat':
        return GAT(in_size, hidden_size, out_size, num_layers)
    else:
        raise ValueError
