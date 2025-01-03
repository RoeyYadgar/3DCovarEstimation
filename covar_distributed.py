import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch import distributed as dist
import os
from covar_sgd import CovarTrainer,Covar
from iterative_covar_sgd import IterativeCovarTrainer,IterativeCovar,IterativeCovarVer2,IterativeCovarTrainerVer2
import math

TMP_STATE_DICT_FILE = 'tmp_state_dict.pt'

def ddp_setup(rank,world_size,backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(backend,rank=rank,world_size=world_size) #TODO: solve compatability issues with NCCL and cuFINUFFT
    torch.cuda.set_device(rank)


def ddp_train(rank,world_size,covar_model,dataset,batch_size_per_proc,savepath = None,kwargs = {}):
    backend = 'nccl' if kwargs.get('nufft_disc') is not None else 'gloo' #For some reason cuFINUFFT breaks when using NCCL backend
    ddp_setup(rank,world_size,backend)
    device = torch.device(f'cuda:{rank}')
    
    num_workers = min(4,(os.cpu_count()-1)//world_size)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size_per_proc,shuffle = False,sampler = DistributedSampler(dataset),
                                             num_workers=num_workers,prefetch_factor=10,persistent_workers=True,pin_memory=True,pin_memory_device=f'cuda:{rank}')
    covar_model = covar_model.to(device)
    covar_model = DDP(covar_model,device_ids=[rank])
    if(type(covar_model.module) == Covar):
        trainer = CovarTrainer(covar_model,dataloader,device,savepath)
    elif(type(covar_model.module) == IterativeCovar):
        trainer = IterativeCovarTrainer(covar_model,dataloader,device,savepath)
    elif(type(covar_model.module) == IterativeCovarVer2):
        trainer = IterativeCovarTrainerVer2(covar_model,dataloader,device,savepath)
    trainer.process_ind = (rank,world_size)

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

