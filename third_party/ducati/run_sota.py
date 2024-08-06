import dgl
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from mylog import get_logger
mlog = get_logger()

import DUCATI
from model import SAGE
from load_graph import load_dc_raw_with_counts
from common import set_random_seeds, get_seeds_list

def entry(args, graph, all_data, seeds_list, counts):
    fanouts = [int(x) for x in args.fanouts.split(",")]
    sampler = dgl.dataloading.NeighborSampler(fanouts)
    gpu_flag, gpu_map, all_cache, _ = DUCATI.CacheConstructor.form_nfeat_cache(args, all_data, counts)

    # prepare a buffer
    input_nodes, _, _ = sampler.sample(graph, seeds_list[0])
    estimate_max_batch = int(1.2*input_nodes.shape[0])
    nfeat_buf = torch.zeros((estimate_max_batch, args.fake_dim), dtype=torch.float).cuda()
    label_buf = torch.zeros((args.bs, ), dtype=torch.long).cuda()
    mlog(f"buffer size: {(estimate_max_batch*args.fake_dim*4+args.bs*8)/(1024**3):.3f} GB")


    # prepare model
    model = SAGE(args.fake_dim, args.num_hidden, args.n_classes, len(fanouts), F.relu, args.dropout)
    model = model.cuda()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def retrieve_data(cpu_data, gpu_data, idx, out_buf):
        nonlocal gpu_map, gpu_flag
        gpu_mask = gpu_flag[idx]
        gpu_nids = idx[gpu_mask]
        gpu_local_nids = gpu_map[gpu_nids].long()
        cpu_nids = idx[~gpu_mask]

        cur_res = out_buf[:idx.shape[0]]
        cur_res[gpu_mask] = gpu_data[gpu_local_nids]
        cur_res[~gpu_mask] = cpu_data[cpu_nids]
        return cur_res

    def run_one_list(target_list):
        nonlocal gpu_flag, gpu_map, all_cache, all_data, sampler
        for seeds in target_list:
            # adj
            input_nodes, output_nodes, blocks = sampler.sample(graph, seeds)
            # nfeat
            cur_nfeat = retrieve_data(all_data[0], all_cache[0], input_nodes, nfeat_buf) # fetch nfeat
            cur_label = retrieve_data(all_data[1], all_cache[1], input_nodes[:args.bs], label_buf) # fetch label
            # train
            batch_pred = model(blocks, cur_nfeat)
            loss = loss_fcn(batch_pred, cur_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # add the first run as warmup
    avgs = []
    for _ in range(args.runs+1):
        torch.cuda.synchronize()
        tic = time.time()
        run_one_list(seeds_list)
        torch.cuda.synchronize()
        avg_duration = 1000*(time.time() - tic)/len(seeds_list)
        avgs.append(avg_duration)
    avgs = avgs[1:]
    mlog(f"sota: {args.nfeat_budget:.3f}GB nfeat cache time: {np.mean(avgs):.2f} ± {np.std(avgs):.2f}ms\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # dataset params
    parser.add_argument("--dataset", type=str, choices=['ogbn-papers100M', 'uk', 'uk-union', 'twitter'],
                        default='ogbn-papers100M')
    parser.add_argument("--pre-epochs", type=int, default=2) # PreSC params

    # running params
    parser.add_argument("--nfeat-budget", type=float, default=1) # in GB
    parser.add_argument("--bs", type=int, default=8000)
    parser.add_argument("--fanouts", type=str, default='15,15,15')
    parser.add_argument("--batches", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--fake-dim", type=int, default=100)
    parser.add_argument("--pre-batches", type=int, default=100)

    # gnn model params
    parser.add_argument('--num-hidden', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.003)

    args = parser.parse_args()
    mlog(args)
    set_random_seeds(0)

    graph, n_classes = load_dc_raw_with_counts(args)
    args.n_classes = n_classes
    graph, all_data, train_idx, counts = DUCATI.CacheConstructor.separate_features_idx(args, graph)
    train_idx = train_idx.cuda()
    graph.pin_memory_()
    mlog(graph)

    # get seeds candidate
    seeds_list = get_seeds_list(args, train_idx)
    del train_idx

    entry(args, graph, all_data, seeds_list, counts)