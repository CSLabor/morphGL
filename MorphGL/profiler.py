from .utils import get_logger
mlog = get_logger()

import os
import math
import time
import torch
import random
import numpy as np

def measure_model_training_time(num_trials, device, gpu_loader, model, loss_fn, optimizer):
    """
    model training time (ms) on GPU or NPU
    """
    torch_device_api_entry = torch.cuda if device.type == 'cuda' else torch.npu
    avgs = []
    mlog(f"\n=======")
    for r in range(num_trials):
        mlog(f"Profile RUN {r} for Model")
        gpu_iter = iter(gpu_loader)
        num_batches = gpu_iter.length
        start_events = [torch_device_api_entry.Event(enable_timing=True) for _ in range(num_batches)]
        end_events = [torch_device_api_entry.Event(enable_timing=True) for _ in range(num_batches)]
        # first saturate GPU with some ops
        m1 = torch.rand(2000,2000,device=device)
        m2 = torch.rand(2000,2000,device=device)
        for _ in range(30):
            m1 = torch.matmul(m1,m2)
        del m1, m2
        if device.type == 'npu':
            fake_y = torch.zeros(gpu_iter.bs,).long().npu()
        #mlog(f"Profile warmup finish")

        # then time
        for i in range(num_batches):
            if device.type == 'cuda':
                batch_x, batch_y, adjs = next(gpu_iter)
                start_events[i].record()
                batch_pred = model(adjs, batch_x)
                loss = loss_fn(batch_pred, batch_y.reshape(-1))
            else:
                batch = next(gpu_iter)
                start_events[i].record()
                batch_pred = model(batch)[:gpu_iter.bs]
                loss = loss_fn(batch_pred, fake_y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_events[i].record()
        torch_device_api_entry.synchronize()
        elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        #mlog(elapsed_times[:5])
        elapsed_times = elapsed_times[1:]
        mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
        avgs.append(np.mean(elapsed_times))
    return np.mean(avgs)

def measure_npu_batching_time(num_trials, npu_loader):
    device = torch.device(f'npu:0')
    mlog(f"\n=======")
    avgs = []
    for r in range(num_trials):
        mlog(f"Profile RUN {r} for NPU Batching")
        gpu_iter = iter(npu_loader)
        num_batches = gpu_iter.length
        start_events = [torch.npu.Event(enable_timing=True) for _ in range(num_batches)]
        end_events = [torch.npu.Event(enable_timing=True) for _ in range(num_batches)]
        # first saturate GPU with other ops
        m1 = torch.rand(2000,2000,device=device)
        m2 = torch.rand(2000,2000,device=device)
        for _ in range(30):
            m1 = torch.matmul(m1,m2)
        del m1, m2
        #mlog(f"Profile warmup finish")

        # then time
        for i in range(num_batches):
            start_events[i].record()
            ret = next(gpu_iter)
            end_events[i].record()
        torch.npu.synchronize()
        elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        #mlog(elapsed_times[:5])
        mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
        avgs.append(np.mean(elapsed_times))
    return np.mean(avgs[1:]) if len(avgs) > 1 else avgs[0]

def measure_gpu_batching_time(num_trials, gpu_loader):
    """
    return average gpu batching time in ms
    """
    device = torch.device(f'cuda:0')
    mlog(f"\n=======")
    avgs = []
    for r in range(num_trials):
        mlog(f"Profile RUN {r} for GPU Batching")
        gpu_iter = iter(gpu_loader)
        num_batches = gpu_iter.length
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        # first saturate GPU with other ops
        m1 = torch.rand(2000,2000,device=device)
        m2 = torch.rand(2000,2000,device=device)
        for _ in range(30):
            m1 = torch.matmul(m1,m2)
        del m1, m2

        # then time
        for i in range(num_batches):
            start_events[i].record()
            ret = next(gpu_iter)
            end_events[i].record()
        torch.cuda.synchronize()
        elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        #mlog(elapsed_times[:5])
        mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
        avgs.append(np.mean(elapsed_times))
    return np.mean(avgs[1:]) if len(avgs) > 1 else avgs[0]

def measure_cpu_batching_time(num_trials, cpu_loader):
    """
    return average cpu batching time in ms
    cpu warmup is slow, so the first run is discarded from stats
    """
    from time import perf_counter
    avgs = []
    mlog(f"\n=======")
    for r in range(1+num_trials):
        if r:
            mlog(f"Profile RUN {r} for CPU Batching ")
        else:
            mlog(f"Warmup for CPU Batching")
        tmp_cnt = 0
        cpu_iter = iter(cpu_loader)
        durs = []
        st = perf_counter()
        while True:
            try:
                ret = cpu_iter.try_one()
                if ret is None:
                    continue
                ed = perf_counter()
                durs.append(1000*(ed-st))
                st = perf_counter()
                tmp_cnt += 1
                if tmp_cnt > 100 and r == 0:
                    # first warmup we run for 100 batches only
                    del cpu_iter
                    break
            except StopIteration:
                break
        if r:
            #mlog(durs[:5])
            mlog(f"{np.mean(durs):.2f} ± {np.std(durs):.2f} ms/batch")
            avgs.append(np.mean(durs))
        else:
            mlog(f"Warmup finish for CPU Batching")
    return np.mean(avgs)

def measure_dma_transfering_to_npu_time(cpu_loader):
    """
    return DMA transferring time in ms
    """
    device = torch.device(f'npu:0')
    mlog(f"\n=======")
    mlog("measuring DMA transferring time")
    num_batches = 50
    saved_batches = []
    while len(saved_batches) < num_batches:
        cpu_iter = iter(cpu_loader)
        for ret in cpu_iter:
            saved_batches.append(ret)
            if len(saved_batches) == num_batches:
                break
    # DMA running time is very stable (if the input is stable), so we only give one run
    start_events = [torch.npu.Event(enable_timing=True) for _ in range(num_batches)]
    end_events = [torch.npu.Event(enable_timing=True) for _ in range(num_batches)]

    # then time
    for i, batch in enumerate(saved_batches):
        start_events[i].record()
        batch.to(device, non_blocking=True)
        end_events[i].record()
    torch.npu.synchronize()

    elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    #mlog(elapsed_times[:5])
    mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
    return np.mean(elapsed_times)


def measure_dma_transfering_to_gpu_time(cpu_loader):
    """
    return DMA transferring time in ms
    """
    device = torch.device(f'cuda:0')
    mlog(f"\n=======")
    mlog("measuring DMA transferring time")
    num_batches = 50
    saved_batches = []
    while len(saved_batches) < num_batches:
        cpu_iter = iter(cpu_loader)
        for ret in cpu_iter:
            saved_batches.append(ret)
            if len(saved_batches) == num_batches:
                break
    # DMA running time is very stable (if the input is stable), so we only give one run
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
    # first saturate GPU and PCIe
    for batch in saved_batches[-10:]:
        batch.to(device, non_blocking=True)

    # then time
    for i, batch in enumerate(saved_batches):
        start_events[i].record()
        batch.to(device, non_blocking=True)
        end_events[i].record()
    torch.cuda.synchronize()
    elapsed_times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    #mlog(elapsed_times[-5:])
    mlog(f"{np.mean(elapsed_times):.2f} ± {np.std(elapsed_times):.2f} ms/batch")
    return np.mean(elapsed_times)


def measure_gpu_batching_model_on_gpu_time(num_trials, gpu_loader, model, loss_fn, optimizer):
    """
    return:
    * GPU batching time
    * Model training time 
    in ms
    """
    device = torch.device(f'cuda:0')
    mlog(f"\n=======")
    avgs = []
    for r in range(num_trials):
        mlog(f"Profile RUN {r} for GPU batching and model")
        gpu_iter = iter(gpu_loader)
        num_batches = gpu_iter.length
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        batching_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_batches)]
        # first saturate GPU with some ops
        m1 = torch.rand(2000,2000,device=device)
        m2 = torch.rand(2000,2000,device=device)
        for _ in range(30):
            m1 = torch.matmul(m1,m2)
        del m1, m2

        # then time
        for i in range(num_batches):
            start_events[i].record()
            batch = next(gpu_iter)
            batching_events[i].record()
            batch_x, batch_y, adjs = batch
            batch_pred = model(adjs, batch_x)
            loss = loss_fn(batch_pred, batch_y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_events[i].record()
        torch.cuda.synchronize()

        cur_ret = []
        batching_times = [s.elapsed_time(e) for s, e in zip(start_events, batching_events)][1:]
        cur_ret.append(np.mean(batching_times))
        model_times = [s.elapsed_time(e) for s, e in zip(batching_events, end_events)][1:]
        cur_ret.append(np.mean(model_times))

        mlog(f"gpu_batching: {np.mean(batching_times):.2f} ± {np.std(batching_times):.2f} ms/batch")
        mlog(f"model training: {np.mean(model_times):.2f} ± {np.std(model_times):.2f} ms/batch")

        avgs.append(cur_ret)
    return list(np.mean(avgs, axis=0))


def measure_batch_storage_size(gpu_loader):
    """
    ATTENTION: here adjs are in int64
    RETURN: avg size of one batch in MB
    """
    mlog(f"\n=======")
    mlog("RUN average batch size in MB")
    size_in_bytes = []
    gpu_iter = iter(gpu_loader)
    for _ in range(50):
        batch_x, batch_y, batch_adjs = next(gpu_iter)
        x_size = batch_x.element_size() * batch_x.numel()
        y_size = batch_y.element_size() * batch_y.numel()
        """
        adj size calculation:
            * coo edge list: 2 * num_edges * 8 Bytes
            * eid list: 1 * num_edges * 8 Bytes
            * src&dst node list: (num_src_nodes + num_dst_nodes) * 8 Bytes
        """
        adj_size = 8 * sum([adj.num_src_nodes() + adj.num_dst_nodes() + 3 * adj.num_edges() for adj in batch_adjs])
        size_in_bytes.append((x_size + y_size + adj_size)/1024**2)

    mlog(size_in_bytes[:5])
    mlog(f"{np.mean(size_in_bytes):.2f} ± {np.std(size_in_bytes):.2f} MB")
    return np.mean(size_in_bytes)

if __name__ == '__main__':
    pass
