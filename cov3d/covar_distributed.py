import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch import distributed as dist
import os
from cov3d.covar_sgd import CovarTrainer,CovarPoseTrainer,compute_updated_fourier_reg
from cov3d.covar import Covar
from cov3d.utils import get_cpu_count
import math

TMP_STATE_DICT_FILE = 'tmp_state_dict.pt'

def ddp_setup(rank,world_size,backend):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group(backend,rank=rank,world_size=world_size) #TODO: solve compatability issues with NCCL and cuFINUFFT
    torch.cuda.set_device(rank)


def ddp_train(rank,world_size,covar_model,dataset,batch_size_per_proc,optimize_pose=False,mean_model = None,pose=None,savepath = None,gt_data=None,kwargs = {}):
    backend = 'nccl' if kwargs.get('nufft_disc') is not None else 'gloo' #For some reason cuFINUFFT breaks when using NCCL backend
    ddp_setup(rank,world_size,backend)
    device = torch.device(f'cuda:{rank}')
    use_halfsets = kwargs.pop('use_halfsets',False)
    num_reg_update_iters = kwargs.pop('num_reg_update_iters',None)
    
    num_workers = min(4,(get_cpu_count()-1)//world_size)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size_per_proc,shuffle = False,sampler = DistributedSampler(dataset),
                                             num_workers=num_workers,prefetch_factor=10,persistent_workers=True,pin_memory=True,pin_memory_device=f'cuda:{rank}')
    
    covar_model = covar_model.to(device)
    if(use_halfsets):
        ranks1 = list(range(world_size//2))
        ranks2 = list(range(world_size//2,world_size))
        group1 = dist.new_group(ranks=ranks1)
        group2 = dist.new_group(ranks=ranks2)
        is_group1 = rank in range(world_size//2)
        if not is_group1:
            torch.manual_seed(1) #Reinitalize vectors in group2
            covar_model.set_vectors(covar_model.init_random_vectors(covar_model.rank))
        covar_model = DDP(covar_model,device_ids=[rank],process_group=group1 if is_group1 else group2)
        #Attach the ranks to the model so that we can use them in the trainer 
        covar_model.ranks_in_group = ranks1 if is_group1 else ranks2
        print(f'Rank {rank} is in group {covar_model.ranks_in_group}')
    else:
        covar_model = DDP(covar_model,device_ids=[rank])
        covar_model.ranks_in_group = list(range(world_size))
    if(not optimize_pose):
        trainer = CovarTrainer(covar_model,dataloader,device,savepath,gt_data=gt_data)
    else:
        mean_model = mean_model.to(device)
        pose = pose.to(device)
        mean_model = DDP(mean_model,device_ids=[rank])
        #pose is not wrapped with DDP, since it has sparse gradients - we instead sync it manually
        trainer = CovarPoseTrainer(covar_model,dataloader,device,mean_model,pose,savepath,gt_data=gt_data)
    
    trainer.process_ind = (rank,world_size)


    if(not use_halfsets):
        num_epochs = kwargs.pop('max_epochs')
        trainer.setup_training(**kwargs)
        for _ in range(num_reg_update_iters):
            trainer.train_epochs(num_epochs,restart_optimizer=True)
            eigenvecs = trainer.covar.eigenvecs
            eigenvecs = eigenvecs[0] * (eigenvecs[1]**0.5).reshape(-1,1,1,1)
            trainer.compute_fourier_reg_term(eigenvecs)
            trainer.covar.orthogonal_projection()
        trainer.train_epochs(num_epochs,restart_optimizer=True)
        trainer.complete_training()
    else:
        num_epochs = kwargs.pop('max_epochs')
        trainer.setup_training(**kwargs)
        for _ in range(num_reg_update_iters):
            trainer.train_epochs(num_epochs,restart_optimizer=True)
            eigenvecs = trainer.covar.eigenvecs
            eigenvecs = eigenvecs[0] * (eigenvecs[1]**0.5).reshape(-1,1,1,1)
            eigenvecs_list = [torch.zeros_like(eigenvecs) for _ in range(world_size)]
            dist.all_gather(tensor_list = eigenvecs_list,tensor = eigenvecs)
            #eigenvecs_list will have the same eigenvecs in each distributed group (i.e. [eigenvecs1,...,eigenvecs1,eigenvesc2,...,eigenvecs2])
            eigenvecs1 = eigenvecs_list[0]
            eigenvecs2 = eigenvecs_list[-1]
            new_fourier_reg_term,covariance_fsc = compute_updated_fourier_reg(eigenvecs1,eigenvecs2,trainer.filter_gain/2,trainer.fourier_reg,covar_model.module.resolution,trainer.optimize_in_fourier_domain,trainer.dataset.mask)
            trainer.update_fourier_reg_halfsets(new_fourier_reg_term)
            if(trainer.logTraining):
                trainer.training_log['covariance_fsc_halfset'] = covariance_fsc

        #Train a single model on the whole dataset
        with torch.no_grad():
            for param in covar_model.parameters(): #Update state of group2
                dist.broadcast(param,src=0)
        #Reset DDP model to include all ranks
        #Get vectors from covar_model with no grid_correction
        covar_model.module.grid_correction = None
        vectors = covar_model.module.get_vectors_spatial_domain().clone().detach()
        covar_model = covar_model.module.__class__(covar_model.module.resolution,
                    covar_model.module.rank,
                    pixel_var_estimate=covar_model.module.pixel_var_estimate,
                    fourier_domain = not covar_model.module._in_spatial_domain,
                    upsampling_factor=covar_model.module.upsampling_factor,
                    vectors=vectors).to(device)
        covar_model = DDP(covar_model,device_ids=[rank])
        covar_model.module.init_grid_correction(kwargs.get('nufft_disc',None))
        covar_model.ranks_in_group = list(range(world_size))
        #Set the optimizer to the new model and retrain
        trainer._covar = covar_model
        trainer.restart_optimizer()
        trainer.train_epochs(num_epochs,restart_optimizer=True)
        trainer.complete_training()

    if(rank == 0):
        torch.save(covar_model.module.state_dict(),TMP_STATE_DICT_FILE)
        if(optimize_pose):
            torch.save(mean_model.module.state_dict(),f'mean_{TMP_STATE_DICT_FILE}')
            torch.save(pose.state_dict(),f'pose_{TMP_STATE_DICT_FILE}')

    dist.destroy_process_group()


def trainParallel(covar_model,dataset,batch_size,num_gpus = 'max',optimize_pose=False,mean_model = None,pose=None,savepath = None,gt_data=None,**kwargs):
    if(num_gpus == 'max'):
        num_gpus = torch.cuda.device_count()

    if(batch_size % num_gpus != 0):
        batch_size = (math.ceil(batch_size/num_gpus) * num_gpus)
        print(f'Batch size is not a multiple of number of GPUs used, increasing batch size to {batch_size}')
    batch_size_per_gpu = int(batch_size / num_gpus)

    mp.spawn(ddp_train,args=(num_gpus,covar_model,dataset,batch_size_per_gpu,optimize_pose,mean_model,pose,savepath,gt_data,kwargs),nprocs = num_gpus)


    def update_module_from_state_dict(module,state_dict_file):
        module.load_state_dict(torch.load(state_dict_file))
        os.remove(state_dict_file)

    update_module_from_state_dict(covar_model,TMP_STATE_DICT_FILE)
    if(optimize_pose):
        update_module_from_state_dict(mean_model,f'mean_{TMP_STATE_DICT_FILE}')
        update_module_from_state_dict(pose,f'pose_{TMP_STATE_DICT_FILE}')

