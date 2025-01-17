import math
from .profiler import *
from .partitioner import *
from .scheduler import *
from .executor import MorphScheduledTrainer
from .utils import *
from loader_npu import partition_train_idx
import MorphGL

mlog = MorphGL.utils.get_logger()
prof_infos = None

def Profiler(num_trials, input_dict):
    set_seeds(0)
    device = input_dict["device"]
    cpu_loader = input_dict["CPU_loader"]
    gpu_loader = input_dict["GPU_loader"]
    model, loss_fcn, optimizer = input_dict["model"]
    batch_size, train_idx = input_dict["batch_info"]
    global prof_infos

    #measure_batch_storage_size(gpu_loader)
    if device.type == 'npu':
        avg_npu_batching_time = measure_npu_batching_time(num_trials, gpu_loader)
        avg_model_time = measure_model_training_time(num_trials, device, gpu_loader, model, loss_fcn, optimizer)
        avg_dma_time = measure_dma_transfering_to_npu_time(cpu_loader)
        avg_cpu_batching_time = measure_cpu_batching_time(num_trials, cpu_loader)
        total_num_batches = math.ceil(train_idx.shape[0] / batch_size)
        assert prof_infos is None
        prof_infos = avg_npu_batching_time, avg_cpu_batching_time, avg_dma_time, avg_model_time, total_num_batches
        mlog(f"profs: {avg_npu_batching_time:.2f},{avg_cpu_batching_time:.2f},{avg_dma_time:.2f},{avg_model_time:.2f},{total_num_batches}")
    else:
        # GPU
        avg_gpu_batching_time, avg_model_time = measure_gpu_batching_model_on_gpu_time(num_trials, gpu_loader, model, loss_fcn, optimizer)
        avg_dma_time = measure_dma_transfering_to_gpu_time(cpu_loader)
        avg_cpu_batching_time = measure_cpu_batching_time(num_trials, cpu_loader)
        total_num_batches = math.ceil(train_idx.shape[0] / batch_size)
        assert prof_infos is None
        prof_infos = avg_gpu_batching_time, avg_cpu_batching_time, avg_dma_time, avg_model_time, total_num_batches
        mlog(f"profs: {avg_gpu_batching_time:.2f},{avg_cpu_batching_time:.2f},{avg_dma_time:.2f},{avg_model_time:.2f},{total_num_batches}")

def Partitioner(oldplan, feedback):
    """
    return either tuple (n_cpu, n_gpu) or None, which means naive pipeline
    """
    global prof_infos
    assert prof_infos is not None
    t_gpu, t_cpu, t_dma, t_model, n_total = prof_infos

    # extreme cases, naive pipeline
    if (t_model > t_cpu and t_model > t_dma) or (t_dma > t_model and t_dma > t_cpu):
        return None

    # initial guess
    if feedback is None and oldplan is None:
        return initial_guess(n_total, t_cpu, t_dma, t_model, t_gpu)

    # finetune with feedback
    return tune_with_feedback(feedback, oldplan)

def Scheduler(oldplan, gpu_buffer_size):
    """
    return:
    * feedback: 1 for too much cpu workload and 0 for too much gpu workload
    * sched_plan: (sched_type, cpu_buffer_size, gpu_buffer_size)
        * sched type: naive_pipe, ada_pipe
    * converge: True or False
    """
    if oldplan is None:
        return None, ("naive_pipe", None, None), True

    global prof_infos
    assert prof_infos is not None
    t_gpu, t_cpu, t_dma, t_model, n_total = prof_infos

    if t_model > t_dma:
        optim_n_cpu = int(n_total * (t_gpu + t_model) / (t_gpu + t_cpu))
        dma_buffer_size = round(gpu_buffer_size*(optim_n_cpu)/(n_total-optim_n_cpu))
        return None, ("ada_pipe", dma_buffer_size, gpu_buffer_size), True

    old_cpu, old_gpu = oldplan
    cur_dma_buffer_size = round(gpu_buffer_size*old_cpu/old_gpu)
    # make sure larger buffer is smaller than 10
    if cur_dma_buffer_size > gpu_buffer_size:
        factor = cur_dma_buffer_size / gpu_buffer_size
        cur_dma_buffer_size = gpu_buffer_size
        gpu_buffer_size = round(gpu_buffer_size/factor)
    feedback, sched_plan, converge = simulate(cur_dma_buffer_size, gpu_buffer_size, t_cpu, t_dma, t_gpu, t_model)
    return feedback, ("ada_pipe", *sched_plan), converge

def Executor(input_dict, sched_plan):
    device = input_dict["device"]
    cpu_loader = input_dict["CPU_loader"]
    gpu_loader = input_dict["GPU_loader"]
    model, loss_fcn, optimizer = input_dict["model"]
    _, train_idx = input_dict["batch_info"]

    pipe_type, cpu_buffer_size, gpu_buffer_size = sched_plan

    if pipe_type == "naive_pipe":
        # extreme case of naive pipeline
        cpu_loader.idx = train_idx
        trainer = MorphScheduledTrainer(device, cpu_loader, MorphGL.iterators.Blank_iter(), 
                model, optimizer, loss_fcn, gpu_buffer_size, cpu_buffer_size)
    elif pipe_type == 'ada_pipe':
        # adaptive buffers overlapping
        cpu_loader.idx, gpu_loader.idx = partition_train_idx(train_idx, 
                cpu_buffer_size/(cpu_buffer_size+gpu_buffer_size))
        trainer = MorphScheduledTrainer(device, cpu_loader, gpu_loader, 
                model, optimizer, loss_fcn, gpu_buffer_size, cpu_buffer_size)
    else:
        # pure GPU-based training
        assert pipe_type == 'no_pipe'
        gpu_loader.idx = train_idx.cuda()
        trainer = MorphScheduledTrainer(device, MorphGL.iterators.Blank_iter(), gpu_loader, 
                model, optimizer, loss_fcn, gpu_buffer_size, cpu_buffer_size)

    return trainer


