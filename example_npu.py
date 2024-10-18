import os
import gc
import time
import torch
import psutil
import numpy as np
import torch.nn as nn
import torch.optim as optim

import MorphGL
from parser import make_parser
from loaders import load_shared_data, construct_graph_from_arrays

if __name__ == "__main__":
    MorphGL.utils.set_seeds(0)
    mlog = MorphGL.utils.get_logger()
    args = make_parser().parse_args()
    cur_process = psutil.Process(os.getpid())
    mlog(f'CMDLINE: {" ".join(cur_process.cmdline())}')
    mlog(args)
    assert args.device == 'npu'

    # load data
    feat_dim, num_classes, train_idx = load_npu_data()
    if args.baseline in ["npu_salient"]:
        cpu_train_idx = train_idx
        npu_train_idx = torch.tensor([]).npu()
    elif args.baseline in ["npu_ducati"]:
        cpu_train_idx = torch.tensor([])
        npu_train_idx = train_idx.npu()
    else:
        assert args.baseline == ''
        cpu_train_idx = train_idx
        npu_train_idx = train_idx.npu()

    # instantiate (1) batching operators (2) model with registration table
    from MorphGL.registration_npu import table as ntable
    cpu_loader = ntable['batching']['cpu'](cpu_train_idx, args.train_batch_size, feat_dim)
    npu_loader = ntable['batching']['npu+pcie'](npu_train_idx, args.train_batch_size, feat_dim)
    model = gtable['training']['npu'](args.model, feat_dim, args.hidden_features, num_classes, len(args.train_fanouts)).npu()

    mlog(model)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # prepare input dict
    input_dict = {}
    input_dict["device"] = torch.device('npu:0')
    input_dict["CPU_loader"] = cpu_loader
    input_dict["GPU_loader"] = npu_loader
    input_dict["model"] = (model, loss_fcn, optimizer)
    input_dict["batch_info"] = (args.train_batch_size, train_idx)

    if args.baseline.strip():
        #################################
        # run baseline 
        #################################
        if args.baseline.strip() in ['salient']:
            sched_plan = ('naive_pipe', 0, 0)
        else:
            assert args.baseline.strip() in ['ducati']
            sched_plan = ('no_pipe', 0, 0)
    else:
        #################################
        # run MorphGL
        #################################
        if args.profs == '':
            # rerun profiling
            MorphGL.Profiler(args.trials, input_dict)
        else:
            # use provided profiling infos
            t_cache, t_gpu, t_cpu, t_dma, t_model, total_batches = [float(x.strip()) for x in args.profs.split(",")]
            MorphGL.prof_infos = t_cache, t_gpu, t_cpu, t_dma, t_model, int(total_batches)

        # decide the partition and schedule plan
        partition_plan, feedback = None, None
        while True:
            partition_plan = MorphGL.Partitioner(partition_plan, feedback)
            feedback, sched_plan, converge = MorphGL.Scheduler(partition_plan, args.buffer_size)
            if converge:
                break

    mlog(sched_plan)
    gc.collect()
    torch.cuda.empty_cache()
    trainer = MorphGL.Executor(input_dict, sched_plan)
    #################################
    # train epochs
    #################################
    durs = []
    for r in range(args.epochs):
        mlog(f'\n\n==============')
        mlog(f'RUN {r}')
        torch.cuda.synchronize()
        tic = time.time()
        trainer.train_one_epoch()
        torch.cuda.synchronize()
        dur = time.time() - tic
        mlog(dur)
        durs.append(dur)
        torch.cuda.empty_cache()
    mlog(durs)
    mlog(f"averaged epoch time: {np.mean(durs[1:]):.2f} ± {np.std(durs[1:]):.2f}")
