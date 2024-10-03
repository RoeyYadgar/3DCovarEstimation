from utils import principalAngles , cosineSimilarity , sim2imgsrc , nonNormalizedGS
import os
import torch
from torch.utils.data import Dataset
from aspire.utils import support_mask,fuzzy_mask
import numpy as np
from tqdm import tqdm
import copy
from aspire.volume import Volume
from aspire.volume import rotated_grids
from nufft_plan import NufftPlan
from projection_funcs import vol_forward,centered_fft2,centered_fft3
from fsc_utils import rpsd,average_fourier_shell,sum_over_shell,expand_fourier_shell

class CovarDataset(Dataset):
    def __init__(self,src,noise_var,vectorsGD = None,mean_volume = None):
        images = src.images[:]
        if(mean_volume is not None): #Substracted projected mean from images
            batch_size = 512
            for i in range(0,src.n,batch_size): #TODO : do this with own wrapper of nufft to improve run time
                projected_mean = src.vol_forward(mean_volume,i,batch_size)
                projected_mean = projected_mean.asnumpy().astype(images.dtype)
                images[i:min(i+batch_size,src.n)] -= projected_mean 
        images = images.shift(-src.offsets)
        images = images/(src.amplitudes[:,np.newaxis,np.newaxis].astype(images.dtype))
        self.resolution = src.L
        self.images = torch.tensor(images)
        self.pts_rot = torch.tensor(rotated_grids(self.resolution,src.rotations).copy()).reshape((3,self.images.shape[0],self.resolution**2))
        self.pts_rot = self.pts_rot.transpose(0,1) 
        self.noise_var = noise_var

        self.set_vectorsGD(vectorsGD)

        self.filter_indices = torch.tensor(src.filter_indices)
        num_filters = len(src.unique_filters)
        self.unique_filters = torch.zeros((num_filters,src.L,src.L))
        for i in range(num_filters):
            self.unique_filters[i] = torch.tensor(src.unique_filters[i].evaluate_grid(src.L))
   
        self.estimate_filters_gain()
        self.estimate_signal_var()
        self.mask_images()
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        return self.images[idx] , self.pts_rot[idx] , self.filter_indices[idx]
    

    def set_vectorsGD(self,vectorsGD):
        if(type(vectorsGD) == torch.Tensor or type(vectorsGD) == np.ndarray):
            if(type(vectorsGD) != torch.Tensor):
                vectorsGD = torch.tensor(vectorsGD)
        self.vectorsGD = vectorsGD

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

        
        nufft_plan = NufftPlan((self.resolution,)*3,batch_size=num_vols,dtype=vol.dtype,device = device)

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
        mask = torch.tensor(support_mask(self.resolution,support_radius))
        mask_size = torch.sum(mask)
        
        signal_psd = torch.zeros((self.resolution,self.resolution))
        for i in range(0,len(self.images),batch_size):
            images_masked = self.images[i:i+batch_size][:,mask]
            images_masked = self.images[i:i+batch_size] * mask
            signal_psd += torch.sum(torch.abs(centered_fft2(images_masked))**2,axis=0)
        signal_psd /= len(self.images) * (self.resolution ** 2) * mask_size
        signal_rpsd = average_fourier_shell(signal_psd)

        noise_psd = torch.ones((self.resolution,self.resolution)) * self.noise_var / (self.resolution**2) 
        noise_rpsd = average_fourier_shell(noise_psd)

        self.signal_rpsd = (signal_rpsd - noise_rpsd)/(self.radial_filters_gain)
        self.signal_rpsd[self.signal_rpsd < 0] = 0 #in low snr setting the estimatoin for high radial resolution might not be accurate enough
        self.signal_var = sum_over_shell(self.signal_rpsd,self.resolution,2).item()
            
    def estimate_filters_gain(self):
        average_filters_gain_spectrum = torch.mean(self.unique_filters ** 2,axis=0) 
        radial_filters_gain = average_fourier_shell(average_filters_gain_spectrum)
        estimated_filters_gain = sum_over_shell(radial_filters_gain,self.resolution,2).item() / (self.resolution**2)

        self.filters_gain = estimated_filters_gain
        self.radial_filters_gain = radial_filters_gain


    def mask_images(self):
        mask = torch.tensor(fuzzy_mask((self.resolution,)*2,dtype=np.float32),dtype=self.images.dtype)
        self.images *= mask
        
        

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
        self.nufft_plans = NufftPlan(vol_shape,batch_size = rank, dtype=dtype,device=device)
        self.filters = train_data.dataset.unique_filters
        if(len(self.filters) < 10000): #TODO : set the threhsold based on available memory of a single GPU
            self.filters = self.filters.to(self.device)
        self.noise_var = train_data.dataset.noise_var
        self.save_path = save_path
        self.logTraining = self.device.index == 0 or self.device == torch.device('cpu') #Only log training on the first gpu
        self.training_log_freq = training_log_freq
        if(self.logTraining):
            self.vectorsGD = train_data.dataset.vectorsGD
            self.log_epoch_ind = []
            self.log_cosine_sim = []
            self.log_fro_err = []
            if(self.vectorsGD != None):
                self.vectorsGD = self.vectorsGD.to(self.device)    

        L = vol_shape[-1]
        vectorsGD_rpsd = rpsd(*train_data.dataset.vectorsGD.reshape((-1,L,L,L)))
        self.fourier_reg = (self.noise_var) / (torch.mean(expand_fourier_shell(vectorsGD_rpsd,L,3),dim=0)) #TODO : validate this regularization term
        self.reg_scale = 1/(len(self.train_data.dataset)) #The sgd is performed on cost/batch_size + reg_term while its supposed to be sum(cost) + reg_term. This ensures the regularization term scales in the appropirate manner

    @property
    def covar(self):
        return self._covar.module if self.isDDP else self._covar
    
    def covar_vectors(self):
        return self.covar.vectors

    def run_batch(self,images,nufft_plans,filters):
        self.optimizer.zero_grad()
        cost_val,vectors = self._covar.forward(images,nufft_plans,filters,self.noise_var,self.reg_scale,self.fourier_reg)
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

        self.cost_in_epoch = torch.tensor(0,device=self.device,dtype=torch.float32)
        for batch_ind,data in enumerate(self.train_data):
            images,pts_rot,filter_indices = data
            num_ims = images.shape[0]
            pts_rot = pts_rot.to(self.device)
            images = images.to(self.device)
            filters = self.filters[filter_indices].to(self.device) if len(self.filters) > 0 else None
            self.nufft_plans.setpts(pts_rot.transpose(0,1).reshape((3,-1)))
            cost_val,vectors = self.run_batch(images,self.nufft_plans,filters)
            with torch.no_grad():
                self.cost_in_epoch += cost_val * self.batch_size

            if(self.logTraining):
                if((batch_ind % self.training_log_freq == 0)):
                    self.log_training(vectors)
                    pbar_description = f"Epoch {epoch} , " + "cost value : {:.2e}".format(cost_val)
                    if(self.vectorsGD is not None):
                        #TODO : update log metrics, use principal angles
                        cosine_sim_val = np.mean(np.sqrt(np.sum(self.log_cosine_sim[-1] ** 2,axis = 0)))
                        fro_err_val = self.log_fro_err[-1]
                        pbar_description =  pbar_description +",  cosine sim : {:.2f}".format(cosine_sim_val) + ", frobenium norm error : {:.2e}".format(fro_err_val)
                        pbar_description += f" , vecs norm : {torch.norm(vectors)}"
                        #pbar_description =  pbar_description +f",  cosine sim : {self.log_cosine_sim[-1]}"
                    pbar.set_description(pbar_description)

                pbar.update(1)

        if(self.isDDP):
            torch.distributed.all_reduce(self.cost_in_epoch,op=torch.distributed.ReduceOp.SUM)

        if(self.logTraining):
            print("Total cost value in epoch : {:.2e}".format(self.cost_in_epoch.item()))
            
    
    def train(self,max_epochs,lr = None,momentum = 0.9,optim_type = 'Adam',reg = 0,gamma_lr = 1,gamma_reg = 1,orthogonal_projection = False,scale_params = True):

        self.use_orthogonal_projection = orthogonal_projection

        if(lr is None):
            lr = 1e-2 if optim_type == 'Adam' else 1e-2 #Default learning rate for Adam/SGD optimizer

        if(scale_params):
            lr *= self.train_data.batch_size #Scale learning rate with batch size
            #reg *= self.train_data.dataset.filters_gain ** 2 #regularization constant should scale the same as cost function 
            #reg /= self.covar.resolution ** 2 #gradient of cost function scales linearly with L while regulriaztion scales with L^3
        
        
        if(optim_type == 'SGD'):
            if(scale_params):
                lr /= self.train_data.dataset.signal_var #gradient of cost function scales with amplitude ^ 3 and so learning rate must scale with amplitude ^ 2 (since we want GD steps to scale linearly with amplitude). signal_var is an estimate for amplitude^2
                lr /= self.train_data.dataset.filters_gain ** 2 #gradient of cost function scales with filter_amplitude ^ 4 and so learning rate must scale with filter_amplitude ^ 4 (since we want GD steps to not scale at all). filters_gain is an estimate for filter_amplitude^2
                #TODO : expirmantation suggests that normalizng lr by resolution is not needed. why?
                #lr /= self.covar.resolution #gradient of cost function scales linearly with L
            self.optimizer = torch.optim.SGD(self.covar.parameters(),lr = lr,momentum = momentum)
        elif(optim_type == 'Adam'):
            self.optimizer = torch.optim.Adam(self.covar.parameters(),lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=1)

        print(f'Actual learning rate {lr}')
        self.reg_scale*=reg
        for epoch in range(max_epochs):
            self.run_epoch(epoch)

            self.reg_scale *= gamma_reg
            scheduler.step(self.cost_in_epoch)
            print(f'New learning rate set to {scheduler.get_last_lr()}')

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
    def __init__(self,resolution,rank,dtype = torch.float32,pixel_var_estimate = 1,vectors = None):
        super().__init__()
        self.resolution = resolution
        self.rank = rank
        self.dtype = dtype
        self.pixel_var_estimate = pixel_var_estimate
        if(vectors is None):
            self.vectors = self.init_random_vectors(rank)
        else:
            self.vectors = torch.clone(vectors)
        self.vectors.requires_grad = True 
        self.vectors = torch.nn.Parameter(self.vectors,requires_grad=True)

    @property
    def device(self):
        return self.vectors.device

    def init_random_vectors(self,num_vectors):
        return (torch.randn((num_vectors,) + (self.resolution,) * 3,dtype=self.dtype)) * (self.pixel_var_estimate ** 0.5)

    def cost(self,images,nufft_plans,filters,noise_var,reg_scale = 0,fourier_reg = None):
        return cost(self.vectors,images,nufft_plans,filters,noise_var,reg_scale,fourier_reg)


    def forward(self,images,nufft_plans,filters,noise_var,reg_scale,fourier_reg):
        return self.cost(images,nufft_plans,filters,noise_var,reg_scale,fourier_reg),self.vectors
    
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



def cost(vols,images,nufft_plans,filters,noise_var,reg_scale = 0,fourier_reg = None):
    batch_size = images.shape[0]
    rank = vols.shape[0]
    L = vols.shape[-1]
    projected_vols = vol_forward(vols,nufft_plans,filters)

    images = images.reshape((batch_size,1,-1))
    projected_vols = projected_vols.reshape((batch_size,rank,-1))

    norm_squared_images = torch.pow(torch.norm(images,dim=(1,2)),2)
    images_projvols_term = torch.matmul(projected_vols,images.transpose(1,2))
    projvols_prod_term = torch.matmul(projected_vols,projected_vols.transpose(1,2))
    
    cost_val = (torch.pow(norm_squared_images,2) - 2 * torch.sum(torch.pow(images_projvols_term,2),dim=(1,2))
                + torch.sum(torch.pow(projvols_prod_term,2),dim=(1,2)))
    
    #Add noise cost terms
    norm_squared_projvols = torch.diagonal(projvols_prod_term,dim1=1,dim2=2)
    cost_val += 2 * noise_var * (torch.sum(norm_squared_projvols,dim=1)-norm_squared_images) + (noise_var * L) ** 2
    
    cost_val = torch.mean(cost_val,dim=0)
            
    if(fourier_reg is not None and reg_scale != 0):
        vols_fourier = centered_fft3(vols)
        vols_fourier*= torch.sqrt(fourier_reg)
        vols_fourier = vols_fourier.reshape((rank,-1))
        reg_cost = torch.sum(torch.norm(vols_fourier,dim=1)**2)
        cost_val += reg_scale * reg_cost

    return cost_val

def frobeniusNorm(vecs):
    #Returns the frobenius norm of a symmetric matrix given by its eigenvectors (multiplied by the corresponding sqrt(eigenval)) (assuming row vectors as input)
    vecs_inn_prod = torch.matmul(vecs,vecs.transpose(0,1))
    return torch.sqrt(torch.sum(torch.pow(vecs_inn_prod,2)))

def frobeniusNormDiff(vec1,vec2):
    #returns the frobenius norm of the diffrence of two symmetric matrices given by their eigenvectors (multiplied by the corresponding sqrt(eigenval)) (assuming row vectors as input)
    
    normdiff_squared = torch.pow(frobeniusNorm(vec1),2) + torch.pow(frobeniusNorm(vec2),2)  - 2*torch.sum(torch.pow(torch.matmul(vec1,vec2.transpose(0,1)),2))
    
    return torch.sqrt(normdiff_squared)


def evalCovarEigs(dataset,eigs,batch_size = 8,reg_scale = 0,fourier_reg = None):
    device = eigs.device
    filters = dataset.unique_filters.to(device)
    num_eigs = eigs.shape[0]
    L = eigs.shape[1]
    nufft_plans = NufftPlan(batch_size,(L,)*3,batch_size=num_eigs,dtype = eigs.dtype,device = device)
    cost_val = 0
    for i in range(0,len(dataset),batch_size):
        images,pts_rot,filter_indices = dataset[i:i+batch_size]
        pts_rot = pts_rot.to(device)
        images = images.to(device)
        batch_filters = filters[filter_indices] if len(filters) > 0 else None
        nufft_plans.setpts(pts_rot.transpose(0,1).reshape((3,-1)))
        cost_term = cost(eigs,images,nufft_plans,batch_filters,dataset.noise_var,reg_scale=reg_scale,fourier_reg=fourier_reg) * batch_size
        cost_val += cost_term
   
    return cost_val/len(dataset)


def trainCovar(covar_model,dataset,batch_size,savepath = None,**kwargs):
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = True)
    trainer = CovarTrainer(covar_model,dataloader,covar_model.device,savepath)
    trainer.train(**kwargs) 
    return
