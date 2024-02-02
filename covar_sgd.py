from covar_estimation import *

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from aspire.volume import Volume
from aspire.utils import Rotation

'''
class ProjDataset(Dataset):
    def __init__(self,projections,rotations):
        
        self.projections= projections
        self.rotations = rotations
        
    def __len__(self):
        return len(self.rotations)
    
    def __getitem__(self,index):
        projection = self.projections[index]
        rotation = self.rotations[index]
        
        return projection,rotation
'''

class Covar(nn.Module):
    def __init__(self,resolution,rank,mean_vol,vectors = None):
        
        super(Covar,self).__init__()
        
        self.rank = rank
        self.resolution = resolution 
        self.mean = mean_vol        
        
        if vectors is None:
            self.vectors = (np.float32(np.random.randn(rank,resolution,resolution,resolution)))
        else:
            if(type(vectors) == np.ndarray):
                self.vectors = vectors.reshape((rank,resolution,resolution,resolution))
            else:
                self.vectors = vectors.asnumpy()
            
        
        self.vectors = torch.tensor(self.vectors,dtype=torch.float32,requires_grad = True)
        
        #self.device = torch.device('cuda')
        #self.to(self.device)
        
        self.verbose_freq = 50
    
    def toVol(self):
        
        return Volume(self.vectors.detach().numpy())
        
    def cost(self,projections,rotations):
        
        return CovarCost.apply(self.vectors,projections,rotations)
        
        
    def train(self,projections,rotations,batch_size,epoch_num,lr = 0.05,momentum = 0.9):
        
        #proj_loader = DataLoader(dataset= ProjDataset(projections,rotations),
        #                         batch_size=batch_size,shuffle = True,collate_fn= lambda x : x)
        
        optimizer = torch.optim.SGD([self.vectors],lr = lr,momentum = momentum)
        
        
        for i in range(1,epoch_num+1):
            self.train_epoch(projections,rotations,batch_size,optimizer)
    
    def train_epoch(self,projections,rotations,batch_size,optimizer):
        
        #self.train() #Todo check if needed
        #projections = projections.to(self.device)
        #rotations = rotations.to(self.device)
        dataset_len = len(rotations)
        for batch_idx in range(int(np.ceil(dataset_len/batch_size))):
            
            batch_samples = np.random.randint(0,dataset_len,batch_size)
            projs = projections[batch_samples]
            rots = Rotation.from_matrix(rotations[batch_samples])
            
            optimizer.zero_grad()
            cost = self.cost(projs,rots)
            cost.backward()
            
            optimizer.step()
            
            
            if(batch_idx % self.verbose_freq == 0):
                print("{:e}".format(cost))
                print(batch_idx)
        
        
        
        
    
        


    
    