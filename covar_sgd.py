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


class Covar():
    def __init__(self,resolution,rank,mean_vol,src,vectors = None,vectorsGD = None):
        

        
        self.rank = rank
        self.resolution = resolution 
        self.mean = mean_vol     
        self.src = src
        
        if(type(self.src) == aspire.source.simulation.Simulation):
            self.src = sim2imgsrc(src)
        
        self.im_norm_factor = np.mean(np.linalg.norm(self.src.images[:],axis=(1,2))) / self.resolution 
        
        self.src._cached_im /= self.im_norm_factor
        
        
        if vectors is None:
            self.vectors = (np.float32(np.random.randn(rank,resolution,resolution,resolution)))/self.resolution**2
        else:
            if(type(vectors) == np.ndarray):
                self.vectors = vectors.reshape((rank,resolution,resolution,resolution))
            else:
                self.vectors = vectors.asnumpy()
            
        self.vectors /= self.im_norm_factor
        self.vectors = torch.tensor(self.vectors,dtype= np2torchDtype(self.vectors.dtype),requires_grad = True)
        
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
        