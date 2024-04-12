from covar_estimation import *
from utils import principalAngles , cosineSimilarity , sim2imgsrc , nonNormalizedGS
import os
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
    def __init__(self,src,vectorsGD = None):
        images = src.images[:]
        self.resolution = src.L
        self.im_norm_factor = np.mean(np.linalg.norm(images[:],axis=(1,2))) / self.resolution #Normalize so the norm is 1 with respect to the inner prod vec1^T * vec2 / L**2 
        self.images = torch.tensor(images.asnumpy()/self.im_norm_factor)
        self.pts_rot = torch.tensor(rotated_grids(self.resolution,src.rotations).copy()).reshape((3,self.images.shape[0],self.resolution**2))
        self.pts_rot = self.pts_rot.transpose(0,1) 
        
        if(type(vectorsGD) == torch.Tensor or type(vectorsGD) == np.ndarray):
            if(type(vectorsGD) != torch.Tensor):
                vectorsGD = torch.tensor(vectorsGD)
        self.vectorsGD = vectorsGD / self.im_norm_factor
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        return self.images[idx] , self.pts_rot[idx]

def dataset_collate(batch):
    images,plans = zip(*batch)
    return torch.stack(images),plans

class CovarTrainer():
    def __init__(self,covar,train_data,device,training_log_freq = 50):
        self.device = device
        self.train_data = train_data
        self.covar = covar.to(device)
        
        batch_size = train_data.batch_size
        self.isDDP = type(self.covar) == torch.nn.parallel.distributed.DistributedDataParallel
        vectors = self.covar_vectors()
        rank = vectors.shape[0]
        dtype = vectors.dtype
        vol_shape = vectors.shape[1:] 
        #TODO : check gpu_sort effect
        self.nufft_plans = [NufftPlan(vol_shape,batch_size=rank,dtype = dtype,gpu_device_id = self.device.index,gpu_method = 1,gpu_sort = 0) for i in range(batch_size)]

        self.logTraining = self.device.index == 0 #Only log training on the first gpu
        self.training_log_freq = training_log_freq
        if(self.logTraining):
            self.vectorsGD = train_data.dataset.vectorsGD
            self.log_epoch_ind = []
            if(self.vectorsGD != None):
                self.vectorsGD = self.vectorsGD.to(self.device)
                self.log_cosine_sim = []
                self.log_fro_err = []
        self.counter = 0
    def covar_vectors(self):
        return self.covar.module.vectors if self.isDDP else self.covar.vectors

    def run_batch(self,images,nufft_plans):
        self.optimizer.zero_grad()
        cost_val,vectors = self.covar.forward(images,nufft_plans,self.reg,self.counter)
        with torch.no_grad():
            #print((self.counter,cost_val,torch.norm(images)))
            self.counter+=1
        cost_val.backward()
        #torch.nn.utils.clip_grad_value_(self.covar.parameters(), 10) #TODO : check for effect of gradient clipping
        #print(vectors,vectors.grad)
        self.optimizer.step()

        return cost_val,vectors

    def run_epoch(self,epoch):
        if(self.isDDP):
            self.train_data.sampler.set_epoch(epoch)
        if(self.logTraining):
            pbar = tqdm(total=len(self.train_data), desc=f'Epoch {epoch} , ',position=0,leave=True)

        for batch_ind,data in enumerate(self.train_data):
            images,pts_rot = data
            num_ims = images.shape[0]
            pts_rot = pts_rot.to(self.device)
            images = images.to(self.device)
            #nufft_plans = [NufftPlan((101,101,101),batch_size=4,dtype = torch.float32,gpu_device_id = self.device.index) for i in range(num_ims)]
            #for i in range(num_ims):
            #    nufft_plans[i].setpts(pts_rot[i])
            #cost_val,vectors = self.run_batch(images,nufft_plans[:num_ims])
            for i in range(num_ims):
                self.nufft_plans[i].setpts(pts_rot[i])
            cost_val,vectors = self.run_batch(images,self.nufft_plans[:num_ims])

            if(self.logTraining):
                if((batch_ind % self.training_log_freq == 0)):
                    self.log_training(vectors)
                    cosine_sim_val = np.mean(np.sqrt(np.sum(self.log_cosine_sim[-1] ** 2,axis = 0)))
                    fro_err_val = self.log_fro_err[-1]
                    pbar_description = f"Epoch {epoch} , "
                    pbar_description =  pbar_description + "cost value : {:.2e}".format(cost_val) +",  cosine sim : {:.2f}".format(cosine_sim_val) + ", frobenium norm error : {:.2e}".format(fro_err_val)
                    pbar.set_description(pbar_description)

                pbar.update(1)
    
    def train(self,max_epochs,lr = 5e-5,momentum = 0.9,optim_type = 'SGD',reg = 0,gamma_lr = 1,gamma_reg = 1,orthogonal_projection = False):
        #TODO : add orthogonal projection option
        if(orthogonal_projection):
            raise Exception("Not implemented yet")
        lr *= self.train_data.batch_size
        lr *= (self.train_data.dataset.resolution**4)

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
            with torch.no_grad():
                self.log_cosine_sim.append(cosineSimilarity(vectors.cpu().detach().numpy(),self.vectorsGD.cpu().numpy())) #TODO : implement cosine sim on torch
                vectorsGD = self.vectorsGD.reshape((self.vectorsGD.shape[0],-1))
                vectors = self.covar_vectors().reshape((vectorsGD.shape))
                self.log_fro_err.append((frobeniusNormDiff(vectorsGD,vectors)/frobeniusNorm(vectorsGD)).cpu().numpy())


    def results_dict(self):
        covar = self.covar.module if self.isDDP else self.covar
        ckp = covar.state_dict()
        ckp['vectorsGD'] = self.vectorsGD
        ckp['log_epoch_ind'] = self.log_epoch_ind
        ckp['log_cosine_sim'] = self.log_cosine_sim
        ckp['log_fro_err'] = self.log_fro_err

        return ckp

    def save_result(self,savepath):
        savedir = os.path.split(savepath)[0]
        os.makedirs(savedir,exist_ok=True)
        ckp = self.results_dict()
        torch.save(ckp,savepath)
                
class Covar(torch.nn.Module):
    def __init__(self,resolution,rank,dtype = torch.float32,vectors = None):
        super(Covar,self).__init__()
        self.resolution = resolution
        self.rank = rank
        self.dtype = dtype
        if(vectors == None):
            self.vectors = (torch.randn((rank,resolution,resolution,resolution),dtype=self.dtype))/(self.resolution ** 1) 
        else:
            self.vectors = torch.clone(vectors)
        self.vectors.requires_grad = True 
        self.vectors = torch.nn.Parameter(self.vectors,requires_grad=True)


    def cost(self,images,nufft_plans,reg = 0,ind = None):
        return cost(self.vectors,images,nufft_plans,reg,ind)


    def forward(self,images,nufft_plans,reg,ind):
        return self.cost(images,nufft_plans,reg,ind),self.vectors
    


def cost(vols,images,nufft_plans,reg = 0,ind=None):
    batch_size = images.shape[0]
    rank = vols.shape[0]
    L = vols.shape[-1]
    projected_vols = vol_forward(vols,nufft_plans)

    images = images.reshape((batch_size,1,-1))/L
    projected_vols = projected_vols.reshape((batch_size,rank,-1))/L

    norm_images_term = torch.pow(torch.norm(images,dim=(1,2)),4)
    images_projvols_term = torch.matmul(projected_vols,images.transpose(1,2))
    projvols_prod_term = torch.matmul(projected_vols,projected_vols.transpose(1,2))
    
    cost_val = (norm_images_term - 2 * torch.sum(torch.pow(images_projvols_term,2),dim=(1,2))
                + torch.sum(torch.pow(projvols_prod_term,2),dim=(1,2)))
    
    cost_val = torch.mean(cost_val,dim=0)
    with torch.no_grad():
        import scipy 
        import pickle
        from nufft_plan import nufft_forward
        #if(torch.abs(torch.norm(projected_vols)/ torch.norm(vol_forward(vols,nufft_plans)/L) - 1) > 1e-2 and cost_val > 1e3):
        if(cost_val > 1e3):
            print((ind,cost_val,torch.norm(images),torch.norm(vols),torch.norm(projected_vols),torch.norm(vol_forward(vols,nufft_plans)/L),torch.norm(nufft_forward(vols,nufft_plans[0])/(L**3))))
            data = {'pts' : nufft_plans[0].pts.cpu().numpy() , 'vols' : vols.cpu().numpy(), 'projs' : projected_vols.cpu().numpy()}
            print(nufft_plans[0].pts)
            scipy.io.savemat('data/pts.mat',data)
            pickle.dump(data,open('data/pts.bin','wb'))
            ghgh
        else:
            #scipy.io.savemat('data/pts2.mat',{'pts' : nufft_plans[0].pts.cpu().numpy() , 'vols' : vols.cpu().numpy(), 'projs' : projected_vols.cpu().numpy()})
            #pickle.dump(data,open('data/pts2.bin','wb'))
            pass
            
    if(reg != 0):
        vols = vols.reshape((rank,-1))/(L ** 1.5)
        vols_prod = torch.matmul(vols,vols.transpose(0,1))
        reg_cost = torch.sum(torch.pow(vols_prod,2))
        cost_val += reg * reg_cost

    return cost_val

def frobeniusNorm(vecs):
    #Returns the frobenius norm of a symmetric matrix given by its eigenvectors (multiplied by the corresponding sqrt(eigenval)) (assuming row vectors as input)
    vecs_inn_prod = torch.matmul(vecs,vecs.transpose(0,1))
    return torch.sqrt(torch.sum(torch.pow(vecs_inn_prod,2)))

def frobeniusNormDiff(vec1,vec2):
    #returns the frobenius norm of the diffrence of two symmetric matrices given by their eigenvectors (multiplied by the corresponding sqrt(eigenval)) (assuming row vectors as input)
    
    normdiff_squared = torch.pow(frobeniusNorm(vec1),2) + torch.pow(frobeniusNorm(vec2),2)  - 2*torch.sum(torch.pow(torch.matmul(vec1,vec2.transpose(0,1)),2))
    
    return torch.sqrt(normdiff_squared)


def trainCovar(covar_model,dataset,batch_size,savepath = None,**kwargs):
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size)
    device = torch.device('cuda:0')
    covar_model = covar_model.to(device)
    trainer = CovarTrainer(covar_model,dataloader,device)
    trainer.train(**kwargs) 
    if(savepath != None):
        trainer.save_result(savepath)

    return
