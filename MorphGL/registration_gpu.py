import dgl
import torch
from .models import dgl_nn_wrapper
from .utils import torch_to_device_wrapper
from .iterators import prepare_salient, prepare_ducati

table = {
        'batching': { # prepare(...) ==> Generator produces batches with CPU/GPU
            'cpu': prepare_salient,
            'gpu+pcie': prepare_ducati,
        },
        'transferring': { # gather_pinned_tensor_rows(cpu_pinned_tensor, gpu_idx) ==> sliced tensor on GPU
            'pcie': torch_to_device_wrapper(torch.device('cuda:0')),
            'gpu+pcie': dgl.utils.pin_memory.gather_pinned_tensor_rows,
        },
        'training': { # dgl_nn_wrapper(...) ==> GNN model on GPU
            'gpu': dgl_nn_wrapper,
        }
}

