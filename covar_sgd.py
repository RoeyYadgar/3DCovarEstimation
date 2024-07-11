from utils import principalAngles , cosineSimilarity , sim2imgsrc , nonNormalizedGS
import os
import torch
from torch.utils.data import Dataset
from aspire.utils import support_mask
import numpy as np
from tqdm import tqdm
import copy
from aspire.volume import Volume
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
    def __init__(self,src,noise_var,vectorsGD = None,mean_volume = None):
        images = src.images[:]
        if(mean_volume is not None): #Substracted projected mean from images
            projected_mean = src.vol_forward(mean_volume,0,src.n)
            projected_mean = projected_mean.asnumpy().astype(images.dtype)
            images -= projected_mean 
        images = images.shift(-src.offsets)
        images = images/src.amplitudes[:,np.newaxis,np.newaxis]
        self.resolution = src.L
        self.images = torch.tensor(images)
        self.pts_rot = torch.tensor(rotated_grids(self.resolution,src.rotations).copy()).reshape((3,self.images.shape[0],self.resolution**2))
        self.pts_rot = self.pts_rot.transpose(0,1) 
        self.noise_var = noise_var

        if(type(vectorsGD) == torch.Tensor or type(vectorsGD) == np.ndarray):
            if(type(vectorsGD) != torch.Tensor):
                vectorsGD = torch.tensor(vectorsGD)
        self.vectorsGD = vectorsGD

        self.filter_indices = torch.tensor(src.filter_indices)
        num_filters = len(src.unique_filters)
        self.unique_filters = torch.zeros((num_filters,src.L,src.L))
        for i in range(num_filters):
            self.unique_filters[i] = torch.tensor(src.unique_filters[i].evaluate_grid(src.L))
   
        self.filters_gain = self.estimate_filters_gain()
        self.signal_var = self.estimate_signal_var()
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        return self.images[idx] , self.pts_rot[idx] , self.filter_indices[idx]
    

    def get_subset(self,idx):
        subset = self.copy()
        subset.images = subset.images[idx]
        subset.pts_rot = subset.pts_rot[idx]
        subset.filter_indices = subset.filter_indices[idx]

        return subset

    def remove_vol_from_images(self,vol,coeffs = None,copy_dataset = False):
        device = vol.device
        num_vols = vol.shape[0]
        if(coeffs is None):
            coeffs = torch.ones(num_vols,len(self))
        dataset = self.copy() if copy_dataset else self

        
        nufft_plan = NufftPlan((self.resolution,)*3,batch_size=num_vols,dtype=vol.dtype,gpu_device_id = device.index,gpu_method=1,gpu_sort = 0)

        for i in range(len(dataset)):
            _,pts_rot,filter_ind = dataset[i]
            pts_rot = pts_rot.to(device)
            filt = dataset.unique_filters[filter_ind].to(device)
            nufft_plan.setpts(pts_rot)

            vol_proj = vol_forward(vol,nufft_plan,filt).reshape(num_vols,dataset.resolution,dataset.resolution)

            dataset.images[i] -= torch.sum(vol_proj * coeffs[i].reshape((-1,1,1)),dim=0).cpu()

        return dataset

    def copy(self):
        return copy.deepcopy(self)
    

    def estimate_signal_var(self,support_radius = None,batch_size=512):
        #Estimates the signal variance per pixel
        #TODO : use different estimation in low SNR?
        L = self.images.shape[-1]
        mask = torch.tensor(support_mask(L,support_radius))
        mask_size = torch.sum(mask)
        
        estimated_var = 0
        for i in range(0,len(self.images),batch_size):
            images_masked = self.images[i:i+batch_size][:,mask]
            estimated_var += torch.sum(images_masked**2)/(len(self.images) * mask_size)

        estimated_var -= self.noise_var 
        estimated_var /= self.filters_gain
        

        return estimated_var
    
    def estimate_filters_gain(self):
        #TODO : weighted average based on number of usages of each unique_filter
        #TODO : weighted sum with spectrum of genertic particle
        L = self.images.shape[-1]
        num_filters = len(self.unique_filters)
        if(num_filters == 0): #When there are no filters the gain is 1
            return 1
        
        estimated_filters_gain = 0
        for i in range(num_filters):
            estimated_filters_gain += (torch.norm(self.unique_filters[i]) ** 2)/(num_filters * L**2)

        return estimated_filters_gain
        

def dataset_collate(batch):
    images,plans = zip(*batch)
    return torch.stack(images),plans

class CovarTrainer():
    def __init__(self,covar,train_data,device,save_path = None,training_log_freq = 50):
        self.device = device
        self.train_data = train_data
        self._covar = covar.to(device)
        
        self.batch_size = train_data.batch_size
        self.isDDP = type(self._covar) == torch.nn.parallel.distributed.DistributedDataParallel
        vectors = self.covar_vectors()
        rank = vectors.shape[0]
        dtype = vectors.dtype
        vol_shape = vectors.shape[1:] 
        #TODO : check gpu_sort effect
        self.nufft_plans = [NufftPlan(vol_shape,batch_size=rank,dtype = dtype,gpu_device_id = self.device.index,gpu_method = 1,gpu_sort = 0) for i in range(self.batch_size)]
        self.filters = train_data.dataset.unique_filters.to(self.device)
        self.noise_var = train_data.dataset.noise_var

        self.logTraining = self.device.index == 0 #Only log training on the first gpu
        self.training_log_freq = training_log_freq
        if(self.logTraining):
            self.vectorsGD = train_data.dataset.vectorsGD
            self.log_epoch_ind = []
            self.log_cosine_sim = []
            self.log_fro_err = []
            if(self.vectorsGD != None):
                self.vectorsGD = self.vectorsGD.to(self.device)    

        self.save_path = save_path

    @property
    def covar(self):
        return self._covar.module if self.isDDP else self._covar
    
    def covar_vectors(self):
        return self.covar.vectors

    def run_batch(self,images,nufft_plans,filters):
        self.optimizer.zero_grad()
        cost_val,vectors = self._covar.forward(images,nufft_plans,filters,self.noise_var,self.reg)
        cost_val.backward()
        #torch.nn.utils.clip_grad_value_(self.covar.parameters(), 10) #TODO : check for effect of gradient clipping
        self.optimizer.step()

        if(self.use_orthogonal_projection):
            self.covar.orthogonal_projection()

        return cost_val,vectors

    def run_epoch(self,epoch):
        if(self.isDDP):
            self.train_data.sampler.set_epoch(epoch)
        if(self.logTraining):
            pbar = tqdm(total=len(self.train_data), desc=f'Epoch {epoch} , ',position=0,leave=True)

        for batch_ind,data in enumerate(self.train_data):
            images,pts_rot,filter_indices = data
            num_ims = images.shape[0]
            pts_rot = pts_rot.to(self.device)
            images = images.to(self.device)
            filters = self.filters[filter_indices] if len(self.filters) > 0 else None
            for i in range(num_ims):
                self.nufft_plans[i].setpts(pts_rot[i])
            cost_val,vectors = self.run_batch(images,self.nufft_plans[:num_ims],filters)

            if(self.logTraining):
                if((batch_ind % self.training_log_freq == 0)):
                    self.log_training(vectors)
                    pbar_description = f"Epoch {epoch} , " + "cost value : {:.2e}".format(cost_val)
                    if(self.vectorsGD is not None):
                        cosine_sim_val = np.mean(np.sqrt(np.sum(self.log_cosine_sim[-1] ** 2,axis = 0)))
                        fro_err_val = self.log_fro_err[-1]
                        pbar_description =  pbar_description +",  cosine sim : {:.2f}".format(cosine_sim_val) + ", frobenium norm error : {:.2e}".format(fro_err_val)
                        pbar_description =  pbar_description +f",  cosine sim : {self.log_cosine_sim[-1]}"
                    pbar.set_description(pbar_description)

                pbar.update(1)
    
    def train(self,max_epochs,lr = 5e-5,momentum = 0.9,optim_type = 'SGD',reg = 0,gamma_lr = 1,gamma_reg = 1,orthogonal_projection = False):
        self.use_orthogonal_projection = orthogonal_projection

        lr *= self.train_data.batch_size #Scale learning rate with batch size
        #lr *= (self.train_data.dataset.resolution**4) # Cost function scales with L^(-4)
        
        if(optim_type == 'SGD'):
            lr /= self.train_data.dataset.signal_var #gradient of cost function scales with amplitude ^ 3 and so learning rate must scale with amplitude ^ 2 (since we want GD steps to scale linearly with amplitude). signal_var is an estimate for amplitude^2
            lr /= self.train_data.dataset.filters_gain ** 2 #gradient of cost function scales with filter_amplitude ^ 4 and so learning rate must scale with filter_amplitude ^ 4 (since we want GD steps to not scale at all). filters_gain is an estimate for filter_amplitude^2
            self.optimizer = torch.optim.SGD(self.covar.parameters(),lr = lr,momentum = momentum)
        elif(optim_type == 'Adam'):
            self.optimizer = torch.optim.Adam(self.covar.parameters(),lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size = 1, gamma = gamma_lr)
        reg /= self.train_data.dataset.filters_gain ** 2 #regularization constant should scale the same as cost function 
        self.reg = reg
        for epoch in range(max_epochs):
            self.run_epoch(epoch)

            self.reg *= gamma_reg
            scheduler.step()

            if(self.logTraining and self.save_path is not None):
                self.save_result()


    def log_training(self,vectors):
        if(len(self.log_epoch_ind) != 0):
            self.log_epoch_ind.append(self.log_epoch_ind[-1] + self.training_log_freq / len(self.train_data))
        else:
            self.log_epoch_ind.append(self.training_log_freq / len(self.train_data))
        

        if(self.vectorsGD != None):
            with torch.no_grad():
                #self.log_cosine_sim.append(cosineSimilarity(vectors.cpu().detach().numpy(),self.vectorsGD.cpu().numpy()))
                self.log_cosine_sim.append(cosineSimilarity(vectors.detach(),self.vectorsGD))
                vectorsGD = self.vectorsGD.reshape((self.vectorsGD.shape[0],-1))
                vectors = self.covar_vectors().reshape((vectors.shape[0],-1))
                self.log_fro_err.append((frobeniusNormDiff(vectorsGD,vectors)/frobeniusNorm(vectorsGD)).cpu().numpy())


    def results_dict(self):
        ckp = self.covar.state_dict()
        ckp['vectorsGD'] = self.vectorsGD
        ckp['log_epoch_ind'] = self.log_epoch_ind
        ckp['log_cosine_sim'] = self.log_cosine_sim
        ckp['log_fro_err'] = self.log_fro_err

        return ckp

    def save_result(self):
        savedir = os.path.split(self.save_path)[0]
        os.makedirs(savedir,exist_ok=True)
        ckp = self.results_dict()
        torch.save(ckp,self.save_path)
                
class Covar(torch.nn.Module):
    def __init__(self,resolution,rank,dtype = torch.float32,vectors = None):
        super().__init__()
        self.resolution = resolution
        self.rank = rank
        self.dtype = dtype
        if(vectors is None):
            self.vectors = (torch.randn((rank,resolution,resolution,resolution),dtype=self.dtype))/(self.resolution ** 1) 
        else:
            self.vectors = torch.clone(vectors)
        self.vectors.requires_grad = True 
        self.vectors = torch.nn.Parameter(self.vectors,requires_grad=True)


    def cost(self,images,nufft_plans,filters,noise_var,reg = 0):
        return cost(self.vectors,images,nufft_plans,filters,noise_var,reg)


    def forward(self,images,nufft_plans,filters,noise_var,reg):
        return self.cost(images,nufft_plans,filters,noise_var,reg),self.vectors
    
    @property
    def eigenvecs(self):
        with torch.no_grad():
            vectors = self.vectors.clone().reshape(self.rank,-1)
            _,eigenvals,eigenvecs = torch.linalg.svd(vectors,full_matrices = False)
            eigenvecs = eigenvecs.reshape(self.vectors.shape)
            eigenvals = eigenvals ** 2
            return eigenvecs,eigenvals
        

    def orthogonal_projection(self):
        with torch.no_grad():
            vectors = self.vectors.reshape(self.rank,-1)
            _,S,V = torch.linalg.svd(vectors,full_matrices = False)
            orthogonal_vectors  = S.reshape(-1,1) * V
            self.vectors.data.copy_(orthogonal_vectors.view_as(self.vectors))



def cost(vols,images,nufft_plans,filters,noise_var,reg = 0):
    batch_size = images.shape[0]
    rank = vols.shape[0]
    L = vols.shape[-1]
    projected_vols = vol_forward(vols,nufft_plans,filters)

    images = images.reshape((batch_size,1,-1))/L
    projected_vols = projected_vols.reshape((batch_size,rank,-1))/L
    noise_var /= L**2

    norm_squared_images = torch.pow(torch.norm(images,dim=(1,2)),2)
    images_projvols_term = torch.matmul(projected_vols,images.transpose(1,2))
    projvols_prod_term = torch.matmul(projected_vols,projected_vols.transpose(1,2))
    
    cost_val = (torch.pow(norm_squared_images,2) - 2 * torch.sum(torch.pow(images_projvols_term,2),dim=(1,2))
                + torch.sum(torch.pow(projvols_prod_term,2),dim=(1,2)))
    
    #Add noise cost terms
    norm_squared_projvols = torch.diagonal(projvols_prod_term,dim1=1,dim2=2)
    cost_val += 2 * noise_var * (torch.sum(norm_squared_projvols,dim=1)-norm_squared_images) + (noise_var * L) ** 2
    
    cost_val = torch.mean(cost_val,dim=0)
            
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


def evalCovarEigs(dataset,eigs,batch_size = 8,reg=0):
    device = eigs.device
    filters = dataset.unique_filters.to(device)
    num_eigs = eigs.shape[0]
    L = eigs.shape[1]
    nufft_plans = [NufftPlan((L,)*3,batch_size=num_eigs,dtype = eigs.dtype,gpu_device_id = device.index,gpu_method = 1,gpu_sort = 0) for i in range(batch_size)]

    cost_val = 0
    for i in range(0,len(dataset),batch_size):
        images,pts_rot,filter_indices = dataset[i:i+batch_size]
        num_ims = images.shape[0]
        pts_rot = pts_rot.to(device)
        images = images.to(device)
        batch_filters = filters[filter_indices] if len(filters) > 0 else None
        for j in range(num_ims):
            nufft_plans[j].setpts(pts_rot[j])
        cost_term = cost(eigs,images,nufft_plans,batch_filters,dataset.noise_var,reg=reg) * batch_size
        cost_val += cost_term
   
    return cost_val/len(dataset)


def trainCovar(covar_model,dataset,batch_size,savepath = None,**kwargs):
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = True)
    device = torch.device('cuda:0')
    covar_model = covar_model.to(device)
    trainer = CovarTrainer(covar_model,dataloader,device,savepath)
    trainer.train(**kwargs) 
    return
