import torch
import torch_npu
from .utils import torch_to_device_wrapper

from .npu_dummy_ops import prepare_batching_cpu, prepare_batching_npu, npu_nn_wrapper
table = {
        'batching': {
            'cpu': prepare_batching_cpu,
            'npu+pcie': prepare_batching_npu,
        },
        'transferring': {
            'pcie': torch_to_device_wrapper(torch.device('npu:0')),
            'npu+pcie': None,
        },
        'training': {
            'npu': npu_nn_wrapper,
        }
}

