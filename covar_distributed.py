import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch import distributed as dist
import os
from covar_sgd import CovarTrainer,Covar,compute_updated_fourier_reg
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
    use_halfsets = kwargs.pop('use_halfsets',False)
    num_reg_update_iters = kwargs.pop('num_reg_update_iters',None)
    
    num_workers = min(4,(os.cpu_count()-1)//world_size)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size_per_proc,shuffle = False,sampler = DistributedSampler(dataset))
                                             #num_workers=num_workers,prefetch_factor=10,persistent_workers=True,pin_memory=True,pin_memory_device=f'cuda:{rank}') #TODO: check if this is needed
    
    covar_model = covar_model.to(device)
    if(use_halfsets):
        group1 = dist.new_group(ranks=list(range(world_size//2)))
        group2 = dist.new_group(ranks=list(range(world_size//2,world_size)))
        is_group1 = rank in range(world_size//2)
        covar_model = DDP(covar_model,device_ids=[rank],process_group=group1 if is_group1 else group2)
    else:
        covar_model = DDP(covar_model,device_ids=[rank])
    if(type(covar_model.module) == Covar):
        trainer = CovarTrainer(covar_model,dataloader,device,savepath)
    elif(type(covar_model.module) == IterativeCovar):
        trainer = IterativeCovarTrainer(covar_model,dataloader,device,savepath)
    elif(type(covar_model.module) == IterativeCovarVer2):
        trainer = IterativeCovarTrainerVer2(covar_model,dataloader,device,savepath)
    trainer.process_ind = (rank,world_size)


    if(not use_halfsets):
        trainer.train(**kwargs)
    else:
        num_epochs = kwargs.pop('max_epochs')
        trainer.setup_training(**kwargs)
        trainer.fourier_reg *= 0
        for _ in range(num_reg_update_iters):
            trainer.train_epochs(num_epochs,restart_optimizer=True)
            eigenvecs = trainer.covar.eigenvecs
            eigenvecs = eigenvecs[0] * eigenvecs[1].reshape(-1,1,1,1)
            eigenvecs_list = [torch.zeros_like(eigenvecs) for _ in range(world_size)]
            dist.all_gather(tensor_list = eigenvecs_list,tensor = eigenvecs)
            #eigenvecs_list will have the same eigenvecs in each distributed group (i.e. [eigenvecs1,...,eigenvecs1,eigenvesc2,...,eigenvecs2])
            eigenvecs1 = eigenvecs_list[0]
            eigenvecs2 = eigenvecs_list[-1]
            new_fourier_reg_term = compute_updated_fourier_reg(eigenvecs1,eigenvecs2,trainer.filter_gain/2,trainer.fourier_reg,covar_model.module.rank,covar_model.module.resolution)
            trainer.fourier_reg = new_fourier_reg_term

        #Train a single model on the whole dataset
        for param in covar_model.parameters(): #Update state of group2
            dist.broadcast(param,src=0)
        trainer._covar = DDP(covar_model.module,device_ids=[rank]) #Reset DDP model to include all ranks
        trainer.train_epochs(num_epochs,restart_optimizer=True)
        trainer.complete_training()

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

