import os
import torch
import time
from aspire.utils import support_mask,fuzzy_mask
import numpy as np
from tqdm import tqdm
import copy
from cov3d.utils import cosineSimilarity,get_cpu_count
from cov3d.nufft_plan import NufftPlan,NufftPlanDiscretized
from cov3d.projection_funcs import vol_forward,centered_fft3
from cov3d.fsc_utils import rpsd,average_fourier_shell,expand_fourier_shell,upsample_and_expand_fourier_shell,covar_fsc

class CovarTrainer():
    def __init__(self,covar,train_data,device,save_path = None,gt_data=None,training_log_freq = 50):
        self.device = device
        self.train_data = train_data
        self._covar = covar.to(device)
        
        self.batch_size = train_data.data_iterable.batch_size if (not isinstance(train_data,torch.utils.data.DataLoader)) else train_data.batch_size
        self.isDDP = type(self._covar) == torch.nn.parallel.distributed.DistributedDataParallel
        self.filters = self.dataset.unique_filters
        if(len(self.filters) < 10000): #TODO : set the threhsold based on available memory of a single GPU
            self.filters = self.filters.to(self.device)
        self.save_path = save_path
        self.logTraining = self.device.index == 0 or self.device == torch.device('cpu') #Only log training on the first gpu
        self.training_log_freq = training_log_freq
        self.gt_data = gt_data
        if(self.logTraining):
            self.log_epoch_ind = []
            self.log_cosine_sim = []
            self.log_fro_err = []
            self.covar_fsc_mean = []
            self.lr_history = []
            self.log_cost_val = []
            self.epoch_run_time = []

        self.filter_gain = self.dataset.get_total_gain().to(self.device)
        self.num_reduced_lr_before_stop = 4
        self.fourier_reg = None
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
    

    @property
    def vectorsGT(self):
        if(self.gt_data is not None):
            if(self.gt_data.eigenvecs is not None):
                return self.gt_data.eigenvecs
            else:
                return None
        else:
            return None
    
    def run_batch(self,images,nufft_plans,filters):
        self.optimizer.zero_grad()
        cost_val = self.cost_func(self._covar(dummy_var=None),images,nufft_plans,filters,self.noise_var,self.reg_scale,self.fourier_reg) #Dummy_var is passed since for some reaosn DDP requires forward method to have an argument
        cost_val.backward()
        #torch.nn.utils.clip_grad_value_(self.covar.parameters(), 1e-3 * self.covar.grad_scale_factor) #TODO : check for effect of gradient clipping
        self.optimizer.step()

        if(self.use_orthogonal_projection):
            self.covar.orthogonal_projection()

        return cost_val

    def run_epoch(self,epoch):
        #if(self.isDDP): #TODO: this shuffles the data in DDP which is not wanted when using halfsets regularization scheme
            #self.train_data.sampler.set_epoch(epoch) 
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
            cost_val = self.run_batch(images,self.nufft_plans,filters)
            with torch.no_grad():
                self.cost_in_epoch += cost_val * self.batch_size

            if(self.logTraining):
                if((batch_ind % self.training_log_freq == 0)):
                    self.log_training(epoch,batch_ind,cost_val)
                    pbar_description = f"Epoch {epoch} , " + "cost value : {:.2e}".format(cost_val)
                    pbar_description += f" , vecs norm : {torch.norm(self.covar.get_vectors())}"
                    if(self.vectorsGT is not None):
                        #TODO : update log metrics, use principal angles
                        cosine_sim_val = np.mean(np.sqrt(np.sum(self.log_cosine_sim[-1] ** 2,axis = 0)))
                        fro_err_val = self.log_fro_err[-1]
                        pbar_description =  pbar_description +",  cosine sim : {:.2f}".format(cosine_sim_val) + ", frobenium norm error : {:.2e}".format(fro_err_val) + ", covar fsc mean : {:.2e}".format(self.covar_fsc_mean[-1])
                    pbar.set_description(pbar_description)

                pbar.update(1)

        if(self.isDDP):
            torch.distributed.all_reduce(self.cost_in_epoch,op=torch.distributed.ReduceOp.SUM)

        if(self.logTraining):
            print("Total cost value in epoch : {:.2e}".format(self.cost_in_epoch.item()))
            
    
    def train(self,max_epochs,**training_kwargs):
        self.setup_training(**training_kwargs)
        self.train_epochs(max_epochs)
        self.complete_training()

    def setup_training(self,lr = None,momentum = 0.9,optim_type = 'Adam',reg = 1,nufft_disc = 'bilinear',orthogonal_projection = False,scale_params = True,objective_func='ml'):
        self.use_orthogonal_projection = orthogonal_projection

        if(lr is None):
            lr = 1e-1 if optim_type == 'Adam' else 1e-2 #Default learning rate for Adam/SGD optimizer
            
        lr *= self.batch_size
        self.lr = lr
        self.optim_type = optim_type
        self.scale_params = scale_params
        self.momentum = momentum
        self.restart_optimizer()
        

        rank = self.covar.rank
        dtype = self.covar.dtype
        vol_shape = (self.covar.resolution,)*3
        self.optimize_in_fourier_domain = nufft_disc != 'nufft' #When disciraztion of NUFFT is used we optimize the objective function if fourier domain since the discritzation receives as input the volume in its fourirer domain.
        self.objective_func = objective_func
        if(self.optimize_in_fourier_domain):
            self.nufft_plans = NufftPlanDiscretized(vol_shape,upsample_factor=self.covar.upsampling_factor,mode=nufft_disc)
            self.dataset.to_fourier_domain()
            self.cost_func = cost_fourier_domain if objective_func == 'ls' else cost_maximum_liklihood_fourier_domain
        else:
            self.nufft_plans = NufftPlan(vol_shape,batch_size = rank, dtype=dtype,device=self.device)
            self.cost_func = cost if objective_func == 'ls' else cost_maximum_liklihood
        self.covar.init_grid_correction(nufft_disc)
        print(f'Actual learning rate {lr}')
        self.reg_scale*=reg
        self.epoch_index = 0

    def restart_optimizer(self):
        params_lr = self.covar.grad_lr_factor()
        for i in range(len(params_lr)):
            params_lr[i]['lr'] *= self.lr

        if(self.optim_type == 'SGD'):
            self.optimizer = torch.optim.SGD(params_lr,momentum = self.momentum)
        elif(self.optim_type == 'Adam'):
            self.optimizer = torch.optim.Adam(params_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,patience=0)

    def train_epochs(self,max_epochs,restart_optimizer = False):
        if(restart_optimizer):
            self.restart_optimizer()

        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            self.run_epoch(self.epoch_index)
            epoch_end_time = time.time()
            print(f'Epoch runtime: {epoch_end_time - epoch_start_time:.2f} seconds')

            self.scheduler.step(self.cost_in_epoch)
            print(f'New learning rate set to {self.scheduler.get_last_lr()}')

            if(self.logTraining and self.save_path is not None):
                self.epoch_run_time.append(epoch_end_time - epoch_start_time)
                self.save_result()

            self.epoch_index += 1
            if self.scheduler.get_last_lr()[0] <= self.lr * (self.scheduler.factor ** self.num_reduced_lr_before_stop):
                print(f"Learning rate has been reduced {self.num_reduced_lr_before_stop} times. Stopping training.")
                break

    def complete_training(self):
        if(self.optimize_in_fourier_domain):#Transform back to spatial domain            
            self.dataset.to_spatial_domain()

    def compute_fourier_reg_term(self,eigenvecs):
        eigen_rpsd = rpsd(*eigenvecs)
        self.fourier_reg = (self.noise_var) / upsample_and_expand_fourier_shell(eigen_rpsd,self.covar.resolution * self.covar.upsampling_factor,3)

    def update_fourier_reg_halfsets(self,fourier_reg):
        fourier_reg = fourier_reg.to(self.device)
        if(self.objective_func == 'ls'):
            fourier_reg = fourier_reg / self.covar.resolution ** 1.5 #TODO: figure out where this factor comes from

        if(self.optimize_in_fourier_domain):
            #This ensures that the fourier_reg term is in the same as the upsampled size of covar
            fourier_reg_radial = average_fourier_shell(fourier_reg) / (self.covar.upsampling_factor ** 3)
            fourier_reg = upsample_and_expand_fourier_shell(fourier_reg_radial,self.covar.resolution * self.covar.upsampling_factor,3)

        self.fourier_reg = fourier_reg

    def log_training(self,num_epoch,batch_ind,cost_val):
        self.log_epoch_ind.append(num_epoch + batch_ind / self.dataloader_len)
        self.lr_history.append(self.scheduler.get_last_lr()[0])
        self.log_cost_val.append(cost_val.detach().cpu().numpy())

        if(self.vectorsGT is not None):
            with torch.no_grad():
                L = self.covar.resolution
                vectors = self.covar.get_vectors_spatial_domain()
                vectorsGT = self.vectorsGT.to(self.device)
                self.covar_fsc_mean.append((covar_fsc(vectorsGT.reshape((vectorsGT.shape[0],L,L,L)),vectors))[:L//2,:L//2].mean().cpu().numpy())
                vectors = vectors.reshape((vectors.shape[0],-1))
                vectorsGT = vectorsGT.reshape((vectorsGT.shape[0],-1))
                self.log_cosine_sim.append(cosineSimilarity(vectors.detach(),vectorsGT))
                self.log_fro_err.append((frobeniusNormDiff(vectorsGT,vectors)/frobeniusNorm(vectorsGT)).cpu().numpy())
                


    def results_dict(self):
        ckp = self.covar.state_dict()
        ckp['vectorsGT'] = self.vectorsGT
        ckp['fourier_reg'] = self.fourier_reg
        ckp['log_epoch_ind'] = self.log_epoch_ind
        ckp['log_cosine_sim'] = self.log_cosine_sim
        ckp['log_fro_err'] = self.log_fro_err
        ckp['covar_fsc_mean'] = self.covar_fsc_mean
        ckp['lr_history'] = self.lr_history
        ckp['log_cost_val'] = self.log_cost_val
        ckp['epoch_run_time'] = self.epoch_run_time

        return ckp

    def save_result(self):
        savedir = os.path.split(self.save_path)[0]
        os.makedirs(savedir,exist_ok=True)
        ckp = self.results_dict()
        torch.save(ckp,self.save_path)


def update_fourier_reg(trainer1,trainer2):
    rank = trainer1.covar.rank
    L = trainer1.covar.resolution
    filter_gain = (trainer1.filter_gain + trainer2.filter_gain)/2
    current_fourier_reg = trainer1.fourier_reg
    #Get the covariance eigenvectors from each trainer
    eigenvecs1 = trainer1.covar.eigenvecs
    eigenvecs1 = eigenvecs1[0] * (eigenvecs1[1]**0.5).reshape(-1,1,1,1)
    
    eigenvecs2 = trainer2.covar.eigenvecs
    eigenvecs2 = eigenvecs2[0] * (eigenvecs2[1]**0.5).reshape(-1,1,1,1)

    new_fourier_reg_tensor = compute_updated_fourier_reg(eigenvecs1,eigenvecs2,filter_gain,current_fourier_reg,L,trainer1.optimize_in_fourier_domain,mask=trainer1.dataset.mask)

    trainer1.update_fourier_reg_halfsets(new_fourier_reg_tensor)
    trainer2.update_fourier_reg_halfsets(new_fourier_reg_tensor)

def compute_updated_fourier_reg(eigenvecs1,eigenvecs2,filter_gain,current_fourier_reg,L,optimize_in_fourier_domain,mask=None):

    if(current_fourier_reg is None):
        current_fourier_reg = torch.zeros((L,)*3,dtype=filter_gain.dtype,device=filter_gain.device)

    averaged_filter_gain = average_fourier_shell(1/filter_gain).reshape(-1,1)
    averaged_filter_gain = (averaged_filter_gain @ averaged_filter_gain.T)
    averaged_filter_gain[:,L//2+1:] = averaged_filter_gain[L//2,L//2]
    averaged_filter_gain[L//2+1:,:] = averaged_filter_gain[L//2,L//2]
    #averaged_filter_gain += 1e-12
    filter_gain_shell_correction = averaged_filter_gain

    #Find a unitary transformation that transforms one set to the other (since these eigenvecs might not be 'aligned')
    if(mask is not None):
        mask = mask.clone().to(eigenvecs1.device)
        eigenvecs1 = eigenvecs1 * mask
        eigenvecs2 = eigenvecs2 * mask

    eigenvecs_fsc = covar_fsc(eigenvecs1,eigenvecs2)
    fsc_epsilon = 1e-6
    eigenvecs_fsc[eigenvecs_fsc < fsc_epsilon] = fsc_epsilon
    eigenvecs_fsc[eigenvecs_fsc > 1-fsc_epsilon] = 1-fsc_epsilon
    
    new_fourier_reg = 1/((eigenvecs_fsc / (1 - eigenvecs_fsc))*filter_gain_shell_correction)
    new_fourier_reg[new_fourier_reg < 0] = 0
    new_fourier_reg /= L ** 4 # This term comes from the normalization by L in projection operator which scales to the power of 4 through the filter gain

    #This is a heuristic approach to get a rank 1 approx of the 'regulariztaion matrix' which allows much faster computation of the regularizaiton term
    new_fourier_reg = expand_fourier_shell(new_fourier_reg.diag().sqrt().unsqueeze(0),L,3)

    if(not optimize_in_fourier_domain):
        #When optimizing in spatial domain regularization needs to be scaled by L^2
        new_fourier_reg /= L ** 2

    return new_fourier_reg

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
        reg_cost = torch.sum(torch.pow(vols_fourier_inner_prod.abs(),2))
        cost_val += reg_scale * reg_cost

    return cost_val

def cost_maximum_liklihood(vols,images,nufft_plans,filters,noise_var,reg_scale=0,fourier_reg=None):
    batch_size = images.shape[0]
    rank = vols.shape[0]

    projected_eigenvecs = vol_forward(vols,nufft_plans,filters)

    images = images.reshape((batch_size,-1,1))
    projected_eigenvecs = projected_eigenvecs.reshape((batch_size,rank,-1))

    projcted_images = torch.matmul(projected_eigenvecs,images) #size (batch, rank,1)

    m = torch.eye(rank,device=vols.device,dtype=vols.dtype).unsqueeze(0) + projected_eigenvecs @ projected_eigenvecs.transpose(1,2) / noise_var
    mean_m = (m.diagonal(dim1=-2,dim2=-1).abs().sum(dim=1)/m.shape[-1]) 
    projcted_images_transformed = torch.linalg.solve(m/mean_m.reshape(-1,1,1),projcted_images) / mean_m.reshape(-1,1,1) #size (batch, rank,1)
    ml_exp_term = - 1/(noise_var**2) * torch.matmul(projcted_images.transpose(1,2).conj(),projcted_images_transformed).squeeze()  #+1/noise_var * torch.norm(images,dim=(1,2)) ** 2  term which is constant
    ml_noise_term = torch.logdet(m) #+ (L**2) * torch.log(noise_var) # term which is constant

    cost_val = 0.5*torch.mean(ml_exp_term + ml_noise_term)

    if(fourier_reg is not None and reg_scale != 0):
        vols_fourier = centered_fft3(vols)
        vols_fourier*= torch.sqrt(fourier_reg)
        reg_cost = torch.sum(torch.norm(vols_fourier.reshape((rank,-1)),dim=1)**2) / (2*noise_var)
        cost_val += reg_scale * reg_cost

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
        #TODO: vols should get Covar's grid_correction reversed here
        vols_fourier = vols * torch.sqrt(fourier_reg)
        vols_fourier = vols_fourier.reshape((rank,-1))
        vols_fourier_inner_prod = vols_fourier @ vols_fourier.conj().T
        reg_cost = torch.sum(torch.pow(vols_fourier_inner_prod.abs(),2))
        cost_val += reg_scale * reg_cost
        

    return cost_val / (L ** 4) #Cost value in fourier domain scales with L^4 compared to spatial domain

def cost_maximum_liklihood_fourier_domain(vols,images,nufft_plans,filters,noise_var,reg_scale=0,fourier_reg=None):
    batch_size = images.shape[0]
    rank = vols.shape[0]
    L = images.shape[-1]

    projected_eigenvecs = vol_forward(vols,nufft_plans,filters,fourier_domain=True)

    images = images.reshape((batch_size,-1,1))
    projected_eigenvecs = projected_eigenvecs.reshape((batch_size,rank,-1))

    projcted_images = torch.matmul(projected_eigenvecs.conj(),images) #size (batch, rank,1)

    m = torch.eye(rank,device=vols.device,dtype=vols.dtype).unsqueeze(0) + projected_eigenvecs.conj() @ projected_eigenvecs.transpose(1,2) / noise_var
    mean_m = (m.diagonal(dim1=-2,dim2=-1).abs().sum(dim=1)/m.shape[-1]) 
    projcted_images_transformed = torch.linalg.solve(m/mean_m.reshape(-1,1,1),projcted_images) / mean_m.reshape(-1,1,1) #size (batch, rank,1)
    ml_exp_term = - 1/(noise_var**2) * torch.matmul(projcted_images.transpose(1,2).conj(),projcted_images_transformed).squeeze()  #+1/noise_var * torch.norm(images,dim=(1,2)) ** 2  term which is constant

    ml_noise_term = torch.logdet(m)  #+(L**2) * torch.log(torch.tensor(noise_var)) term which is constant
    cost_val = 0.5*torch.mean(ml_exp_term + ml_noise_term).real

    if(fourier_reg is not None and reg_scale != 0):
        vols_fourier = vols * torch.sqrt(fourier_reg)
        reg_cost = torch.sum(torch.norm(vols_fourier.reshape((rank,-1)),dim=1)**2) / (2*noise_var)
        cost_val += reg_scale * reg_cost

    return cost_val


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
    nufft_plans = NufftPlan((L,)*3,batch_size=num_eigs,dtype = eigs.dtype,device = device)
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


def trainCovar(covar_model,dataset,batch_size,savepath = None,gt_data=None,**kwargs):
    num_workers = min(4,get_cpu_count()-1)
    use_halfsets = kwargs.pop('use_halfsets')
    num_reg_update_iters = kwargs.pop('num_reg_update_iters',None)
    if(not use_halfsets):
        dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = True,
                                                num_workers=num_workers,prefetch_factor=10,persistent_workers=True,pin_memory=True,pin_memory_device=str(covar_model.device))
        #from torchtnt.utils.data.data_prefetcher import CudaDataPrefetcher
        #dataloader = CudaDataPrefetcher(dataloader,device=covar_model.device,num_prefetch_batches=4) #TODO : should this be used here? doesn't seem to improve perforamnce
        trainer = CovarTrainer(covar_model,dataloader,covar_model.device,savepath,gt_data=gt_data)

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
        covar_model_copy = copy.deepcopy(covar_model)
        with torch.no_grad(): #Reinitalize the copied model since having the same initalization will produce unwanted correlation even after training
            covar_model_copy.set_vectors(covar_model_copy.init_random_vectors(covar_model.rank))
        half1,half2 = dataset.half_split()

        num_epochs = kwargs.pop('max_epochs')

        #TODO: Use sampler like in DDP to not have to split dataset?
        dataloader1 = torch.utils.data.DataLoader(half1,batch_size = batch_size,shuffle = True,
                                                num_workers=num_workers,prefetch_factor=10,persistent_workers=True,pin_memory=True,pin_memory_device=str(covar_model.device))
        trainer1 = CovarTrainer(covar_model,dataloader1,covar_model.device,savepath,gt_data=gt_data)
        trainer1.setup_training(**kwargs) 

        dataloader2 = torch.utils.data.DataLoader(half2,batch_size = batch_size,shuffle = True,
                                                num_workers=num_workers,prefetch_factor=10,persistent_workers=True,pin_memory=True,pin_memory_device=str(covar_model.device))
        trainer2 = CovarTrainer(covar_model_copy,dataloader2,covar_model_copy.device,save_path=None,gt_data=gt_data)
        trainer2.setup_training(**kwargs)

        for i in range(0,num_reg_update_iters):
            trainer1.train_epochs(num_epochs,restart_optimizer=True)
            trainer2.train_epochs(num_epochs,restart_optimizer=True)
            update_fourier_reg(trainer1,trainer2)

        trainer1.complete_training()
        trainer2.complete_training()

        #Train on full dataset #TODO: reuse trainer1 to avoid having to set up the training again, this still requries to transform the dataset to fourier domain if needed
        full_dataloader = torch.utils.data.DataLoader(dataset,batch_size = batch_size,shuffle = True,
                                                num_workers=num_workers,prefetch_factor=10,persistent_workers=True,pin_memory=True,pin_memory_device=str(covar_model.device))
        trainer_final = CovarTrainer(covar_model,full_dataloader,covar_model.device,savepath,gt_data=gt_data)
        trainer_final.fourier_reg = trainer1.fourier_reg
        trainer_final.train(max_epochs=num_epochs,**kwargs)
        

    return
