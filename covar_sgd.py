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
from pytorch_finufft import functional as torchnufft
from nufft_plan import NufftPlan
from projection_funcs import vol_forward

class CovarDataset(Dataset):
    def __init__(self,src,rank):
        images = src.images[:]
        self.rank = rank
        self.resolution = src.L
        self.im_norm_factor = np.mean(np.linalg.norm(images[:],axis=(1,2))) / self.resolution 
        self.images = torch.tensor(images.asnumpy()/self.im_norm_factor)
        self.pts_rot = torch.tensor(rotated_grids(self.resolution,src.rotations).copy()).reshape((3,self.images.shape[0],self.resolution**2))
        
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        plans = []
        if(type(idx) == int):
            plan = NufftPlan((self.resolution,)*3,batch_size = self.rank,dtype = self.images.dtype,eps = 1.2e-6)
            plan.setpts(self.pts_rot[:,idx].to('cuda:0')) #TODO : load to the same gpu of the image
            plans = plan
        elif(type(idx) == slice):
            start = idx.start if idx.start != None else 0
            stop = idx.stop if idx.stop != None else self.images.shape[0]
            step = idx.step if idx.step != None else 1
            for i in range(start,stop,step):
                plan = NufftPlan((self.resolution,)*3,batch_size = self.rank,dtype = self.images.dtype,eps = 1.2e-6)
                plan.setpts(self.pts_rot[:,i].to('cuda:0')) #TODO : load to the same gpu of the image
                plans.append(plan)
        return self.images[idx] , plans#self.pts_rot[:,idx]

def dataset_collate(batch):
    images,plans = zip(*batch)
    return torch.stack(images),plans

class CovarTrainer():
    def __init__(self,covar,train_data,device,training_log_freq = 20,vectorsGD = None):
        self.device = device
        self.train_data = train_data
        self.covar = covar.to(device)
        #self.covar = DDP(covar,device_ids=[device])
        self.training_log_freq = training_log_freq
        self.vectorsGD = vectorsGD

        self.log_epoch_ind = []
        if(vectorsGD != None):
            self.log_cosine_sim = []
    def run_batch(self,images,nufft_plans):
        self.optimizer.zero_grad()
        cost_val = self.covar.cost(images,nufft_plans)
        cost_val.backward()
        #torch.nn.utils.clip_grad_value_(self.covar.parameters(), 1000) #TODO : check for effect of gradient clipping
        self.optimizer.step()

        return cost_val

    def run_epoch(self,epoch):
        #batch_size = len(next(iter(self.train_data)))
        #self.train_data.sampler.set_epoch(epoch)
        pbar = tqdm(total=len(self.train_data), desc=f'Epoch {epoch} , ',position=0,leave=True)
        for batch_ind,data in enumerate(self.train_data):
            images,nufft_plans = data
            images = images.to(self.device)
            cost_val = self.run_batch(images,nufft_plans)

            if(batch_ind % self.training_log_freq == 0):
                self.log_training()
                cosine_sim_val = np.mean(np.sqrt(np.sum(self.log_cosine_sim[-1] ** 2,axis = 0)))
                pbar_description =  "cost value : {:.2e}".format(cost_val) +",  cosine sim : {:.2f}".format(cosine_sim_val)
                pbar.set_description(pbar_description)

            pbar.update(1)
            #print(torch.norm(self.covar.vectors.detach()))

    def train(self,max_epochs,lr = 5e-5,momentum = 0.9,optim_type = 'SGD',reg = 0,gamma_lr = 1,gamma_reg = 1,orthogonal_projection = False):
        if(optim_type == 'SGD'):
            self.optimizer = torch.optim.SGD(self.covar.parameters(),lr = lr,momentum = momentum)
        elif(optim_type == 'Adam'):
            self.optimizer = torch.optim.Adam(self.covar.parameters(),lr = lr)
        
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size = 1, gamma = gamma_lr)
        for epoch in range(max_epochs):
            self.run_epoch(epoch)

            reg *= gamma_reg
            scheduler.step()


    def log_training(self):
        if(len(self.log_epoch_ind) != 0):
            self.log_epoch_ind.append(self.log_epoch_ind[-1] + self.training_log_freq / len(self.train_data))
        else:
            self.log_epoch_ind.append(self.training_log_freq / len(self.train_data))
        

        if(self.vectorsGD != None):
            self.log_cosine_sim.append(cosineSimilarity(self.covar.vectors.cpu().detach().numpy(),self.vectorsGD.numpy()))

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


    def cost(self,images,nufft_plans):
        batch_size = images.shape[0]
        rank = self.vectors.shape[0]
        projected_vols = vol_forward(self.vectors,nufft_plans)

        images = images.reshape((batch_size,1,-1))
        projected_vols = projected_vols.reshape((batch_size,rank,-1))

        norm_images_term = torch.pow(torch.norm(images,dim=(1,2)),4)
        images_projvols_term = torch.matmul(projected_vols,images.transpose(1,2))
        projvols_prod_term = torch.matmul(projected_vols,projected_vols.transpose(1,2))
        
        cost_val = (norm_images_term - 2 * torch.sum(torch.pow(images_projvols_term,2),dim=(1,2))
                    + torch.sum(torch.pow(projvols_prod_term,2),dim=(1,2)))
        
        cost_val = torch.mean(cost_val,dim=0)
        #TODO : add regulairzation

        return cost_val

'''
class Covar():
    def __init__(self,resolution,rank,src,vectors = None,vectorsGD = None):
        
        self.rank = rank
        self.resolution = resolution 
        self.src = src
        
        if(type(self.src) == aspire.source.simulation.Simulation):
            self.src = sim2imgsrc(src)
        
        self.im_norm_factor = np.mean(np.linalg.norm(self.src.images[:],axis=(1,2))) / self.resolution 
        
        self.src._cached_im /= self.im_norm_factor
        
        
        if vectors is None:
            self.vectors = (np.float32(np.random.randn(rank,resolution,resolution,resolution)))/np.sqrt(self.resolution**3) 
        else:
            if(type(vectors) == np.ndarray):
                self.vectors = vectors.reshape((rank,resolution,resolution,resolution))
            else:
                self.vectors = vectors.asnumpy()
            
        
        self.vectors = torch.tensor(self.vectors/self.im_norm_factor,dtype= np2torchDtype(self.vectors.dtype),requires_grad = True)
        
        #self.device = torch.device('cuda')
        #self.to(self.device)
        
        self.verbose_freq = 50
        self.log_freq = 10
        self.epoch_ind_log = []
        self.cost_log = []
        
        self.vectorsGD = vectorsGD
        if(vectorsGD is not None):
            self.vectorsGD /= self.im_norm_factor
            self.cosine_sim_log = []
            self.principal_angles_log = []
        
    
    def toVol(self):
        
        return Volume(self.vectors.detach().numpy())
        
    def cost(self,image_ind,images,reg = 0):
        
        return CovarCost.apply(self.vectors,self.src,image_ind,images,reg)
        
        
    def train(self,batch_size,epoch_num,lr = 5,momentum = 0.9,optim_type = 'SGD',reg = 0,gamma_lr = 1 ,gamma_reg = 1, orthogonal_projection = False):
        
        if(optim_type == 'SGD'):
            optimizer = torch.optim.SGD([self.vectors],lr = lr,momentum = momentum)
        elif(optim_type == 'Adam'):
            optimizer = torch.optim.Adam([self.vectors],lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 1, gamma = gamma_lr)
        
        for i in range(1,epoch_num+1):
            self.train_epoch(batch_size,optimizer,reg,orthogonal_projection)
            
            reg *= gamma_reg
            scheduler.step()
    
    def train_epoch(self,batch_size,optimizer ,reg = 0 , orthogonal_projection = False):
        
        #self.train() #Todo check if needed
        #projections = projections.to(self.device)
        #rotations = rotations.to(self.device)
        
        dataset_len = self.src.n
        pbar = tqdm(total=int(np.ceil(dataset_len/batch_size)), desc="SGD Progress",position=0,leave=True)
        
        
        for batch_idx in range(int(np.ceil(dataset_len/batch_size))):
            
            batch_image_ind = (batch_idx)*batch_size
            images = Image(self.src.images[batch_image_ind + np.arange(0,batch_size)])
            
            optimizer.zero_grad()
            cost = self.cost(batch_image_ind,images,reg)
            cost.backward()
                    
            optimizer.step()
            
            if(orthogonal_projection):
                with torch.no_grad():
                    self.vectors.data = nonNormalizedGS(self.vectors)             
                    #vectors_svd = torch.linalg.svd(self.vectors.reshape((self.rank,-1)), full_matrices = False)
                    #orthogonal_vecs = vectors_svd[1].reshape((self.rank,-1)) * vectors_svd[2]
                    #self.vectors.data = orthogonal_vecs.reshape((self.rank,self.resolution,self.resolution,self.resolution))
                    

                    
            
            pbar.update(1)
            
            
            
            if(batch_idx % self.log_freq == 0):
                self.log_training(cost.detach().numpy(), batch_size*self.log_freq / dataset_len)
                if(torch.isnan(cost)):
                    raise Exception('Cost value is nan')
            
            if(batch_idx % self.verbose_freq == 0):
                if(self.vectorsGD is not None):
                    cosine_sim_val = np.mean(np.sqrt(np.sum(self.cosine_sim_log[-1] ** 2,axis = 0)))
                    pbar_description  = "cost value : {:.2e}".format(cost) +   ",  cosine sim : {:.2f}".format(cosine_sim_val)
                else:
                    pbar_description  = f'cost value : {cost}'
                pbar.set_description(pbar_description)
                
            

    def log_training(self,cost_val,epoch_ratio):
        if(len(self.epoch_ind_log) != 0):
            self.epoch_ind_log.append(self.epoch_ind_log[-1] + epoch_ratio)
        else:
            self.epoch_ind_log.append(epoch_ratio)
        self.cost_log.append(cost_val)
        
        if(self.vectorsGD is not None):
            self.cosine_sim_log.append(cosineSimilarity(self.vectors.detach().numpy(),self.vectorsGD))
            self.principal_angles_log.append(principalAngles(self.vectors.detach().numpy(),self.vectorsGD))
        

    def save(self,filename):        
        with open(filename,'wb') as file:
            pickle.dump(self,file)
        

    @staticmethod
    def load(filename):
        with open(filename,'rb') as file:
            return pickle.load(file)
'''