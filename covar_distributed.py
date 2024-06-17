import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch import distributed as dist
import os
from covar_sgd import CovarTrainer
import math

TMP_STATE_DICT_FILE = 'tmp_state_dict.pt'

def ddp_setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("gloo",rank=rank,world_size=world_size) #TODO: solve compatability issues with NCCL and cuFINUFFT
    torch.cuda.set_device(rank)


def ddp_train(rank,world_size,covar_model,dataset,batch_size_per_proc,savepath = None,kwargs = {}):
    ddp_setup(rank,world_size)
    device = torch.device(f'cuda:{rank}')
    
    
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size_per_proc,shuffle = False,sampler = DistributedSampler(dataset))
    covar_model = covar_model.to(device)
    covar_model = DDP(covar_model,device_ids=[rank])
    trainer = CovarTrainer(covar_model,dataloader,device,savepath)

    trainer.train(**kwargs)

    if(rank == 0):
        torch.save(covar_model.module.state_dict(),TMP_STATE_DICT_FILE)

    dist.destroy_process_group()


def trainParallel(covar_model,dataset,num_gpus = 'max',batch_size=1,savepath = None,**kwargs):
    if(num_gpus == 'max'):
        num_gpus = torch.cuda.device_count()

    if(batch_size % num_gpus != 0):
        batch_size = (math.ceil(batch_size/num_gpus) * num_gpus)
        print(f'Batch size is not a multiple of number of GPUs used, increasing batch size to {batch_size}')
    batch_size_per_gpu = int(batch_size / num_gpus)

    mp.spawn(ddp_train,args=(num_gpus,covar_model,dataset,batch_size_per_gpu,savepath,kwargs),nprocs = num_gpus)

    covar_model.load_state_dict(torch.load(TMP_STATE_DICT_FILE))
    os.remove(TMP_STATE_DICT_FILE)

