from covar_estimation import *
from utils import principalAngles , cosineSimilarity , sim2imgsrc , nonNormalizedGS

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import pickle
from aspire.volume import Volume
from aspire.utils import Rotation
from aspire.volume import rotated_grids
from nufft_plan import NufftPlan
from projection_funcs import vol_forward

def ptsrot2nufftplan(pts_rot,img_size,**kwargs):
    num_plans = pts_rot.shape[0] #pts_rot has a shape [num_images,3,L^2]
    plans = []
    for i in range(num_plans):
        #plan = NufftPlan((img_size,)*3,gpu_method=1,gpu_sort=0,**kwargs)
        plan = NufftPlan((img_size,)*3,**kwargs)
        plan.setpts(pts_rot[i]) 
        plans.append(plan)

    return plans

class CovarDataset(Dataset):
    def __init__(self,src,rank):
        images = src.images[:]
        self.rank = rank
        self.resolution = src.L
        self.im_norm_factor = np.mean(np.linalg.norm(images[:],axis=(1,2))) / self.resolution 
        self.images = torch.tensor(images.asnumpy()/self.im_norm_factor)
        self.pts_rot = torch.tensor(rotated_grids(self.resolution,src.rotations).copy()).reshape((3,self.images.shape[0],self.resolution**2))
        self.pts_rot = self.pts_rot.transpose(0,1) 
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        return self.images[idx] , self.pts_rot[idx]

def dataset_collate(batch):
    images,plans = zip(*batch)
    return torch.stack(images),plans

class CovarTrainer():
    def __init__(self,covar,train_data,device,training_log_freq = 50,vectorsGD = None):
        self.device = device
        self.train_data = train_data
        self.covar = covar.to(device)
        self.training_log_freq = training_log_freq
        self.vectorsGD = vectorsGD

        self.log_epoch_ind = []
        if(vectorsGD != None):
            self.log_cosine_sim = []


        batch_size = train_data.batch_size
        self.isDDP = type(self.covar) == torch.nn.parallel.distributed.DistributedDataParallel
        vectors = self.covar_vectors()
        rank = vectors.shape[0]
        dtype = vectors.dtype
        vol_shape = vectors.shape[1:] 
        #TODO : check gpu_sort effect
        self.nufft_plans = [NufftPlan(vol_shape,batch_size=rank,dtype = dtype,gpu_device_id = self.device.index,gpu_method = 1,gpu_sort = 0) for i in range(batch_size)]

    def covar_vectors(self):
        return self.covar.module.vectors if self.isDDP else self.covar.vectors

    def run_batch(self,images,nufft_plans):
        self.optimizer.zero_grad()
        cost_val,vectors = self.covar.forward(images,nufft_plans,self.reg)
        cost_val.backward()
        #torch.nn.utils.clip_grad_value_(self.covar.parameters(), 1000) #TODO : check for effect of gradient clipping
        self.optimizer.step()

        return cost_val,vectors

    def run_epoch(self,epoch):
        #self.train_data.sampler.set_epoch(epoch)
        pbar = tqdm(total=len(self.train_data), desc=f'Epoch {epoch} , ',position=0,leave=True)
        for batch_ind,data in enumerate(self.train_data):
            images,pts_rot = data
            num_ims = images.shape[0]
            pts_rot = pts_rot.to(self.device)
            images = images.to(self.device)
            for i in range(num_ims):
                self.nufft_plans[i].setpts(pts_rot[i])
            
            cost_val,vectors = self.run_batch(images,self.nufft_plans[:num_ims])

            if(batch_ind % self.training_log_freq == 0):
                self.log_training(vectors)
                cosine_sim_val = np.mean(np.sqrt(np.sum(self.log_cosine_sim[-1] ** 2,axis = 0)))
                pbar_description =  "cost value : {:.2e}".format(cost_val) +",  cosine sim : {:.2f}".format(cosine_sim_val)
                pbar_description = f'gpu : {self.device.index} ,' + pbar_description
                pbar.set_description(pbar_description)

            pbar.update(1)

    def train(self,max_epochs,lr = 5e-5,momentum = 0.9,optim_type = 'SGD',reg = 0,gamma_lr = 1,gamma_reg = 1):
        #TODO : add orthogonal projection option
        lr *= self.train_data.batch_size

        if(optim_type == 'SGD'):
            self.optimizer = torch.optim.SGD(self.covar.parameters(),lr = lr,momentum = momentum)
        elif(optim_type == 'Adam'):
            self.optimizer = torch.optim.Adam(self.covar.parameters(),lr = lr)
        
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size = 1, gamma = gamma_lr)
        self.reg = reg
        for epoch in range(max_epochs):
            self.run_epoch(epoch)

            self.reg *= gamma_reg
            scheduler.step()


    def log_training(self,vectors):
        if(len(self.log_epoch_ind) != 0):
            self.log_epoch_ind.append(self.log_epoch_ind[-1] + self.training_log_freq / len(self.train_data))
        else:
            self.log_epoch_ind.append(self.training_log_freq / len(self.train_data))
        

        if(self.vectorsGD != None):
            self.log_cosine_sim.append(cosineSimilarity(vectors.cpu().detach().numpy(),self.vectorsGD.numpy()))

class Covar(torch.nn.Module):
    def __init__(self,resolution,rank,dtype = torch.float32,norm_factor = 1,vectors = None):
        super(Covar,self).__init__()
        self.resolution = resolution
        self.rank = rank
        self.dtype = dtype
        if(vectors == None):
            self.vectors = (torch.randn((rank,resolution,resolution,resolution),dtype=self.dtype))/(self.resolution ** 2) 
        else:
            self.vectors = torch.clone(vectors)
        self.vectors /= norm_factor
        self.vectors.requires_grad = True 
        self.vectors = torch.nn.Parameter(self.vectors,requires_grad=True)


    def cost(self,images,nufft_plans,reg = 0):
        return cost(self.vectors,images,nufft_plans,reg)


    def forward(self,images,nufft_plans,reg):
        return self.cost(images,nufft_plans,reg),self.vectors
    


def cost(vols,images,nufft_plans,reg = 0):
    batch_size = images.shape[0]
    rank = vols.shape[0]
    projected_vols = vol_forward(vols,nufft_plans)

    images = images.reshape((batch_size,1,-1))
    projected_vols = projected_vols.reshape((batch_size,rank,-1))

    norm_images_term = torch.pow(torch.norm(images,dim=(1,2)),4)
    images_projvols_term = torch.matmul(projected_vols,images.transpose(1,2))
    projvols_prod_term = torch.matmul(projected_vols,projected_vols.transpose(1,2))
    
    cost_val = (norm_images_term - 2 * torch.sum(torch.pow(images_projvols_term,2),dim=(1,2))
                + torch.sum(torch.pow(projvols_prod_term,2),dim=(1,2)))
    
    cost_val = torch.mean(cost_val,dim=0)
    if(reg != 0):
        vols = vols.reshape((rank,-1))
        vols_prod = torch.matmul(vols,vols.transpose(0,1))
        reg_cost = torch.sum(torch.pow(vols_prod,2))
        cost_val += reg * reg_cost

    return cost_val

