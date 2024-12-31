from utils import cosineSimilarity,soft_edged_kernel,get_torch_device,get_complex_real_dtype
import os
import torch
import time
from torch.utils.data import Dataset
from aspire.utils import support_mask,fuzzy_mask
import numpy as np
from tqdm import tqdm
import copy
from aspire.volume import Volume
from aspire.volume import rotated_grids
from nufft_plan import NufftPlan,NufftPlanDiscretized
from projection_funcs import vol_forward,centered_fft2,centered_ifft2,centered_fft3,centered_ifft3,pad_tensor
from fsc_utils import rpsd,average_fourier_shell,sum_over_shell,expand_fourier_shell

class CovarDataset(Dataset):
    def __init__(self,src,noise_var,vectorsGD = None,mean_volume = None,mask='fuzzy'):
        self.resolution = src.L
        self.pts_rot = torch.tensor(rotated_grids(self.resolution,src.rotations).copy()).reshape((3,src.n,self.resolution**2)) #TODO : replace this with torch affine_grid with size (N,1,L,L,1)
        self.pts_rot = self.pts_rot.transpose(0,1) 
        self.pts_rot = (torch.remainder(self.pts_rot + torch.pi , 2 * torch.pi) - torch.pi) #After rotating the grids some of the points can be outside the [-pi , pi]^3 cube
        self.noise_var = noise_var

        self.set_vectorsGD(vectorsGD)

        self.filter_indices = torch.tensor(src.filter_indices.astype(int)) #For some reason ASPIRE store filter_indices as string for some star files
        num_filters = len(src.unique_filters)
        self.unique_filters = torch.zeros((num_filters,src.L,src.L))
        for i in range(num_filters):
            self.unique_filters[i] = torch.tensor(src.unique_filters[i].evaluate_grid(src.L))
   
        self.images = torch.tensor(self.preprocess_images(src,mean_volume))
        self.estimate_filters_gain()
        self.estimate_signal_var()
        self.mask_images(mask)

        self.dtype = self.images.dtype
        self._in_spatial_domain = True
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        return self.images[idx] , self.pts_rot[idx] , self.filter_indices[idx]
    
    '''
    def prepare_batch(self,idx,nufft_plan):
        device = nufft_plan.device
        images,pts_rot,filter_indices = self.__getitem__(idx)
        images = images.to(device)
        pts_rot = pts_rot.to(device)
        filters = self.unique_filters[filter_indices].to(device) if len(self.unique_filters) > 0 else None

        nufft_plan.setpts(pts_rot.transpose(0,1).reshape((3,-1)))

        return images,nufft_plan,filters
    '''

    def preprocess_images(self,src,mean_volume,batch_size=512):
        device = get_torch_device()
        mean_volume = torch.tensor(mean_volume.asnumpy(),device=device)
        nufft_plan = NufftPlan((self.resolution,)*3,batch_size = 1, dtype=mean_volume.dtype,device=device)

        images = src.images[:]
        images = images.shift(-src.offsets)
        images = images/(src.amplitudes[:,np.newaxis,np.newaxis].astype(images.dtype))
        if(mean_volume is not None): #Substracted projected mean from images. Using own implemenation of volume projection since Aspire implemention is too slow
            for i in range(0,src.n,batch_size): #TODO : do this with own wrapper of nufft to improve run time
                pts_rot = self.pts_rot[i:(i+batch_size)]
                filter_indices = self.filter_indices[i:(i+batch_size)]
                filters = self.unique_filters[filter_indices].to(device) if len(self.unique_filters) > 0 else None
                pts_rot = pts_rot.to(device)
                nufft_plan.setpts(pts_rot.transpose(0,1).reshape((3,-1)))
                projected_mean = vol_forward(mean_volume,nufft_plan,filters).squeeze(1)


                images[i:min(i+batch_size,src.n)] -= projected_mean.cpu().numpy().astype(images.dtype)


        return images
    
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
    

    def to_fourier_domain(self):
        if(self._in_spatial_domain):
            self.images = centered_fft2(self.images)
            #TODO : transform points into grid_sample format here instead of in discretization function?
            self.noise_var *= self.resolution**2 #2-d Fourier transform scales everything by a factor of L (and the variance scaled by L**2)
            self.dtype = get_complex_real_dtype(self.dtype)
            self._in_spatial_domain = False

    def to_spatial_domain(self):
        if(not self._in_spatial_domain):
            self.images = centered_ifft2(self.images)
            self.noise_var /= self.resolution**2
            self.dtype = get_complex_real_dtype(self.dtype)
            self._in_spatial_domain = True

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


    def mask_images(self,mask,batch_size=512):
        if(mask == 'fuzzy'):
            mask = torch.tensor(fuzzy_mask((self.resolution,)*2,dtype=np.float32),dtype=self.images.dtype)
            self.images *= mask
        elif(isinstance(mask,Volume) or isinstance(mask,str)):
            if(isinstance(mask,str)):
                if(os.path.isfile(mask)):
                    mask = Volume.load(mask)
                else:
                    raise Exception(f'Mask input {mask} is not a valid file')

            if(mask.resolution > self.resolution):
                mask = mask.downsample(self.resolution)
            
            min_mask_val = mask.asnumpy().min()
            max_mask_val = mask.asnumpy().max()
            if(np.abs(min_mask_val) > 1e-3 or np.abs(max_mask_val - 1) > 1e-3):
                print(f'Warning : mask volume range is [{min_mask_val},{max_mask_val}]. Normalzing mask')
                mask = (mask - min_mask_val) / (max_mask_val - min_mask_val)

            device = get_torch_device()

            mask = torch.tensor(mask.asnumpy(),device=device)

            softening_kernel = soft_edged_kernel(radius=5,L=self.resolution,dim=2)
            softening_kernel = torch.tensor(softening_kernel,device=device)
            softening_kernel_fourier = centered_fft2(softening_kernel)

            nufft_plan = NufftPlan((self.resolution,)*3,batch_size = 1, dtype=mask.dtype,device=device)

            for i in range(0,len(self.images),batch_size):
                _,pts_rot,_ = self[i:(i+batch_size)]
                pts_rot = pts_rot.to(device)
                nufft_plan.setpts(pts_rot.transpose(0,1).reshape((3,-1)))
                projected_mask = vol_forward(mask,nufft_plan).squeeze(1)

                if(i == 0): #Use first batch to determine threshold
                    vals = projected_mask.reshape(-1).cpu().numpy()
                    threshold = np.percentile(vals[vals > 10 ** (-1.5)],10) #filter values which aren't too close to 0 and take a threhosld that captures 90% of the projected mask
                
                mask_binary = projected_mask > threshold
                mask_binary_fourier = centered_fft2(mask_binary)

                soft_mask_binary = centered_ifft2(mask_binary_fourier * softening_kernel_fourier).real

                self.images[i:min(i+batch_size,len(self.images))] *= soft_mask_binary.cpu()
        
        
        
class CovarTrainer():
    def __init__(self,covar,train_data,device,save_path = None,training_log_freq = 50):
        self.device = device
        self.train_data = train_data
        self._covar = covar.to(device)
        
        self.batch_size = train_data.data_iterable.batch_size if (not isinstance(train_data,torch.utils.data.DataLoader)) else train_data.batch_size
        self.isDDP = type(self._covar) == torch.nn.parallel.distributed.DistributedDataParallel
        vectors = self.covar_vectors()
        vol_shape = vectors.shape[1:] 
        self.filters = self.dataset.unique_filters
        if(len(self.filters) < 10000): #TODO : set the threhsold based on available memory of a single GPU
            self.filters = self.filters.to(self.device)
        self.save_path = save_path
        self.logTraining = self.device.index == 0 or self.device == torch.device('cpu') #Only log training on the first gpu
        self.training_log_freq = training_log_freq
        if(self.logTraining):
            self.vectorsGD = self.dataset.vectorsGD
            self.log_epoch_ind = []
            self.log_cosine_sim = []
            self.log_fro_err = []
            if(self.vectorsGD != None):
                self.vectorsGD = self.vectorsGD.to(self.device)    

        L = self.dataset.resolution
        vectors_gd_padded = pad_tensor(self.dataset.vectorsGD.reshape((-1,L,L,L)),vol_shape,dims=[-1,-2,-3])
        vectorsGD_rpsd = rpsd(*vectors_gd_padded)
        self.fourier_reg = (self.noise_var) / (torch.mean(expand_fourier_shell(vectorsGD_rpsd,vol_shape[-1],3),dim=0)) #TODO : validate this regularization term        
        self.reg_scale = 1/(len(self.dataset)) #The sgd is performed on cost/batch_size + reg_term while its supposed to be sum(cost) + reg_term. This ensures the regularization term scales in the appropirate manner

    @property
    def dataset(self):
        return self.train_data.data_iterable.dataset if (not isinstance(self.train_data,torch.utils.data.DataLoader)) else self.train_data.dataset
    
    @property
    def dataloader_len(self):
        return len(self.train_data.data_iterable) if (not isinstance(self.train_data,torch.utils.data.DataLoader)) else len(self.train_data)

    @property
    def covar(self):
        return self._covar.module if self.isDDP else self._covar
    
    @property
    def noise_var(self):
        return self.dataset.noise_var
    
    def covar_vectors(self):
        return self.covar.get_vectors()

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
            pbar = tqdm(total=self.dataloader_len, desc=f'Epoch {epoch} , ',position=0,leave=True)

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
                    vectors = self.covar_vectors()
                    self.log_training()
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
            
    
    def train(self,max_epochs,lr = None,momentum = 0.9,optim_type = 'Adam',reg = 0,gamma_lr = 1,gamma_reg = 1,nufft_disc = None,orthogonal_projection = False,scale_params = True):

        self.use_orthogonal_projection = orthogonal_projection

        if(lr is None):
            lr = 1e-2 if optim_type == 'Adam' else 1e-2 #Default learning rate for Adam/SGD optimizer

        if(scale_params):
            lr *= self.batch_size #Scale learning rate with batch size
            #reg *= self.dataset.filters_gain ** 2 #regularization constant should scale the same as cost function 
            #reg /= self.covar.resolution ** 2 #gradient of cost function scales linearly with L while regulriaztion scales with L^3
        
        
        if(optim_type == 'SGD'):
            if(scale_params):
                lr /= self.dataset.signal_var #gradient of cost function scales with amplitude ^ 3 and so learning rate must scale with amplitude ^ 2 (since we want GD steps to scale linearly with amplitude). signal_var is an estimate for amplitude^2
                lr /= self.dataset.filters_gain ** 2 #gradient of cost function scales with filter_amplitude ^ 4 and so learning rate must scale with filter_amplitude ^ 4 (since we want GD steps to not scale at all). filters_gain is an estimate for filter_amplitude^2
                #TODO : expirmantation suggests that normalizng lr by resolution is not needed. why?
                #lr /= self.covar.resolution #gradient of cost function scales linearly with L
                #TODO : should lr scale by L^2.5 when performing optimization in fourier domain? since gradient scales with L^4 and volume scales with L^1.5
            self.optimizer = torch.optim.SGD(self.covar.parameters(),lr = lr,momentum = momentum)
        elif(optim_type == 'Adam'):
            self.optimizer = torch.optim.Adam(self.covar.parameters(),lr = lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=1)

        vectors = self.covar_vectors()
        rank = vectors.shape[0]
        dtype = vectors.dtype
        vol_shape = (self.covar.resolution,)*3
        self.optimize_in_fourier_domain = nufft_disc is not None #When disciraztion of NUFFT is used we optimize the objective function if fourier domain since the discritzation receives as input the volume in its fourirer domain.
        if(self.optimize_in_fourier_domain):
            self.nufft_plans = NufftPlanDiscretized(vol_shape,upsample_factor=self.covar.upsampling_factor,mode=nufft_disc)
            self.dataset.to_fourier_domain()
        else:
            self.nufft_plans = NufftPlan(vol_shape,batch_size = rank, dtype=dtype,device=self.device)

        print(f'Actual learning rate {lr}')
        self.reg_scale*=reg
        for epoch in range(max_epochs):
            self.run_epoch(epoch)

            self.reg_scale *= gamma_reg
            scheduler.step(self.cost_in_epoch)
            print(f'New learning rate set to {scheduler.get_last_lr()}')

            if(self.logTraining and self.save_path is not None):
                self.save_result()

        if(self.optimize_in_fourier_domain):#Transform back to spatial domain            
            self.dataset.to_spatial_domain()


    def log_training(self):
        if(len(self.log_epoch_ind) != 0):
            self.log_epoch_ind.append(self.log_epoch_ind[-1] + self.training_log_freq / self.dataloader_len)
        else:
            self.log_epoch_ind.append(self.training_log_freq / self.dataloader_len)
        

        if(self.vectorsGD != None):
            with torch.no_grad():
                #self.log_cosine_sim.append(cosineSimilarity(vectors.cpu().detach().numpy(),self.vectorsGD.cpu().numpy()))
                vectors = self.covar.get_vectors_spatial_domain()
                vectors = vectors.reshape((vectors.shape[0],-1))
                self.log_cosine_sim.append(cosineSimilarity(vectors.detach(),self.vectorsGD))
                vectorsGD = self.vectorsGD.reshape((self.vectorsGD.shape[0],-1))
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
    def __init__(self,resolution,rank,dtype = torch.float32,pixel_var_estimate = 1,fourier_domain = False,upsampling_factor=2,vectors = None):
        super().__init__()
        self.resolution = resolution
        self.rank = rank
        self.pixel_var_estimate = pixel_var_estimate
        self.dtype = dtype
        self.upsampling_factor = upsampling_factor

        #We split tensor to real and imagniry part
        # In the case of spatial domain imag part would be zeros and won't be used.
        # In the case of fourier domain these tensors store the real and imag part seperately. The reason we don't use a complex tensor is that pytorch DDP doesn't support complex parameters (might be supported in the future?)
        vectors = self.init_random_vectors(rank) if vectors is None else torch.clone(vectors)

        if(fourier_domain):
            #fourier_vectors = centered_fft3(vectors,padding_size=(self.resolution*self.upsampling_factor,)*3)
            #self._vectors_real = torch.nn.Parameter(fourier_vectors.real,requires_grad=True)
            #self._vectors_imag = torch.nn.Parameter(fourier_vectors.imag,requires_grad=True)
            self.cost_func = cost_fourier_domain
            self._in_spatial_domain = False
        else:
            self.cost_func = cost
            self._in_spatial_domain = True
        self._vectors_real = torch.nn.Parameter(vectors,requires_grad=True)
        

    @property
    def device(self):
        return self._vectors_real.device
    
    @property
    def vectors(self):
        #return self._vectors_real if self._in_spatial_domain else (self._vectors_real,self._vectors_imag)
        return self._vectors_real

    def get_vectors(self): #This method is different in `vectors`` property when `self._in_spatial_domain == False` : in that case this method returns a complex tensor instead of (real,imag) tuple
        return self.get_vectors_spatial_domain() if self._in_spatial_domain else self.get_vectors_fourier_domain()
    
    def init_random_vectors(self,num_vectors):
        return (torch.randn((num_vectors,) + (self.resolution,) * 3,dtype=self.dtype)) * (self.pixel_var_estimate ** 0.5)

    def cost(self,images,nufft_plans,filters,noise_var,reg_scale = 0,fourier_reg = None):
        return self.cost_func(self.get_vectors(),images,nufft_plans,filters,noise_var,reg_scale,fourier_reg)


    def forward(self,images,nufft_plans,filters,noise_var,reg_scale,fourier_reg):
        return self.cost(images,nufft_plans,filters,noise_var,reg_scale,fourier_reg),self.vectors
    
    @property
    def eigenvecs(self):
        with torch.no_grad():
            vectors = self.get_vectors_spatial_domain().clone().reshape(self.rank,-1)
            _,eigenvals,eigenvecs = torch.linalg.svd(vectors,full_matrices = False)
            eigenvecs = eigenvecs.reshape((self.rank,self.resolution,self.resolution,self.resolution))
            eigenvals = eigenvals ** 2
            return eigenvecs,eigenvals
        
    def get_vectors_fourier_domain(self):
        return centered_fft3(self.vectors,padding_size=(self.resolution*self.upsampling_factor,)*3)

    def get_vectors_spatial_domain(self):
        return self.vectors

    def orthogonal_projection(self):
        with torch.no_grad():
            vectors = self.get_vectors_spatial_domain().reshape(self.rank,-1)
            _,S,V = torch.linalg.svd(vectors,full_matrices = False)
            orthogonal_vectors  = (S.reshape(-1,1) * V).view_as(self._vectors_real)
            self._vectors_real.data.copy_(orthogonal_vectors)



    def state_dict(self,*args,**kwargs):
        state_dict = super().state_dict(*args,**kwargs)
        state_dict.update({'_in_spatial_domain' : self._in_spatial_domain, 'cost_func' : self.cost_func})
        return state_dict
    
    def load_state_dict(self,state_dict,*args,**kwargs):
        self._in_spatial_domain = state_dict.pop('_in_spatial_domain')
        self.cost = state_dict.pop('cost_func')
        super().load_state_dict(state_dict, *args, **kwargs)
        return
            



def cost(vols,images,nufft_plans,filters,noise_var,reg_scale = 0,fourier_reg = None):
    batch_size = images.shape[0]
    rank = vols.shape[0]
    L = vols.shape[-1]
    projected_vols = vol_forward(vols,nufft_plans,filters)

    images = images.reshape((batch_size,1,-1))
    projected_vols = projected_vols.reshape((batch_size,rank,-1))

    #norm_squared_images = torch.pow(torch.norm(images,dim=(1,2)),2) #This term is constant with respect to volumes
    images_projvols_term = torch.matmul(projected_vols,images.transpose(1,2))
    projvols_prod_term = torch.matmul(projected_vols,projected_vols.transpose(1,2))
    
    cost_val = (- 2 * torch.sum(torch.pow(images_projvols_term,2),dim=(1,2))
                + torch.sum(torch.pow(projvols_prod_term,2),dim=(1,2))) # +torch.pow(norm_squared_images,2) term is constant
    
    #Add noise cost terms
    norm_squared_projvols = torch.diagonal(projvols_prod_term,dim1=1,dim2=2)
    cost_val += 2 * noise_var * (torch.sum(norm_squared_projvols,dim=1)) #-2*noise_var*norm_squared_images+(noise_var * L) ** 2 #Term is constant
    
    cost_val = torch.mean(cost_val,dim=0)
            
    if(fourier_reg is not None and reg_scale != 0):
        vols_fourier = centered_fft3(vols)
        vols_fourier*= torch.sqrt(fourier_reg)
        vols_fourier = vols_fourier.reshape((rank,-1))
        vols_fourier_inner_prod = vols_fourier @ vols_fourier.conj().T
        #reg_cost = torch.sum(torch.norm(vols_fourier,dim=1)**2)
        reg_cost = torch.sum(torch.pow(vols_fourier_inner_prod.abs(),2))
        cost_val += reg_scale * reg_cost /(L**4) #L^4 needed here because objective function scales with L^2 when moving into fourier space? #TODO : not sure this is needed (should get canceled with noise?)

    return cost_val

#TODO : merge this into a single function in cost
def cost_fourier_domain(vols,images,nufft_plans,filters,noise_var,reg_scale = 0,fourier_reg = None):
    batch_size = images.shape[0]
    #vols can either be a 2-tuple of (real,imag) tensors or a complex tensor
    rank = vols[0].shape[0] if isinstance(vols,tuple) else vols.shape[0]
    L = images.shape[-1]
    projected_vols = vol_forward(vols,nufft_plans,filters,fourier_domain=True)


    images = images.reshape((batch_size,1,-1))
    projected_vols = projected_vols.reshape((batch_size,rank,-1))

    #norm_squared_images = torch.pow(torch.norm(images,dim=(1,2)),2) #This term is constant with respect to volumes
    images_projvols_term = torch.matmul(projected_vols,images.transpose(1,2).conj())
    projvols_prod_term = torch.matmul(projected_vols,projected_vols.transpose(1,2).conj())
    
    cost_val = (- 2 * torch.sum(torch.pow(images_projvols_term.abs(),2),dim=(1,2))
                + torch.sum(torch.pow(projvols_prod_term.abs(),2),dim=(1,2))) #+torch.pow(norm_squared_images,2) term is constant
    
    #Add noise cost terms
    norm_squared_projvols = torch.diagonal(projvols_prod_term,dim1=1,dim2=2).real #This should be real already but this ensures the dtype gets actually converted
    cost_val += 2 * noise_var * (torch.sum(norm_squared_projvols,dim=1)) #-2*noise_var*norm_squared_images+(noise_var * L) ** 2 #Term is constant
    cost_val = torch.mean(cost_val,dim=0)

    if(fourier_reg is not None and reg_scale != 0):        
        vols_fourier = torch.complex(*vols) * torch.sqrt(fourier_reg)
        vols_fourier = vols_fourier.reshape((rank,-1))
        vols_fourier_inner_prod = vols_fourier @ vols_fourier.conj().T
        #reg_cost = torch.sum(torch.norm(vols_fourier,dim=1)**2)
        reg_cost = torch.sum(torch.pow(vols_fourier_inner_prod.abs(),2))
        cost_val += reg_scale * reg_cost

    return cost_val / (L ** 4) #Cost value in fourier domain scales with L^4 compared to spatial domain

def frobeniusNorm(vecs):
    #Returns the frobenius norm of a symmetric matrix given by its eigenvectors (multiplied by the corresponding sqrt(eigenval)) (assuming row vectors as input)
    vecs_inn_prod = torch.matmul(vecs,vecs.transpose(0,1).conj())
    return torch.sqrt(torch.sum(torch.pow(vecs_inn_prod,2)))

def frobeniusNormDiff(vec1,vec2):
    #returns the frobenius norm of the diffrence of two symmetric matrices given by their eigenvectors (multiplied by the corresponding sqrt(eigenval)) (assuming row vectors as input)
    
    normdiff_squared = torch.pow(frobeniusNorm(vec1),2) + torch.pow(frobeniusNorm(vec2),2)  - 2*torch.sum(torch.pow(torch.matmul(vec1,vec2.transpose(0,1).conj()),2))
    
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
    num_workers = max(1,os.cpu_count()-1)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = True,
                                             num_workers=num_workers,prefetch_factor=10,persistent_workers=True,pin_memory=True,pin_memory_device='cuda:0')
    #from torchtnt.utils.data.data_prefetcher import CudaDataPrefetcher
    #dataloader = CudaDataPrefetcher(dataloader,device=covar_model.device,num_prefetch_batches=4) #TODO : should this be used here? doesn't seem to improve perforamnce
    trainer = CovarTrainer(covar_model,dataloader,covar_model.device,savepath)
    trainer.train(**kwargs) 
    return
