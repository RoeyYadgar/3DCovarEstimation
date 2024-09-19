import torch
from torch import distributed as dist
from covar_sgd import Covar,CovarDataset,CovarTrainer,cost
from nufft_plan import NufftPlan
from wiener_coords import wiener_coords

class IterativeCovarTrainer(CovarTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._orig_dataset = self.train_data.dataset

    def train(self,*args,**kwargs):
        for i in range(self.covar.rank):
            super().train(*args,**kwargs)

            self.covar.fix_vector()
            eigenvecs,eigenvals = self.covar.eigenvecs
            if(not self.isDDP):
                coords,eigen_forward = wiener_coords(self._orig_dataset,eigenvecs,eigenvals,return_eigen_forward=True)
                coords = coords.to('cpu')
                eigen_span_im = torch.sum(coords[:,:,None,None] * eigen_forward,dim=1)
                self.train_data.dataset.images = self._orig_dataset.images - eigen_span_im 
            else:
                data_len = len(self.train_data.dataset)
                rank = self.process_ind[0]
                world_size = self.process_ind[1]
                samples_per_process = data_len // world_size
                start_ind = rank * samples_per_process
                end_ind = start_ind + samples_per_process if rank != world_size - 1 else data_len
                coords,eigen_forward = wiener_coords(self._orig_dataset,eigenvecs,eigenvals,start_ind = start_ind,end_ind = end_ind,return_eigen_forward=True)
                coords = coords.to('cpu')
                eigen_span_im = torch.sum(coords[:,:,None,None] * eigen_forward,dim=1)

                updated_parts = [torch.zeros_like(eigen_span_im) for _ in range(world_size)]
                dist.all_gather(updated_parts, eigen_span_im)
                complete_eigen_span_im = torch.cat(updated_parts)
                self.train_data.dataset.images = self._orig_dataset.images - complete_eigen_span_im
            

class IterativeCovar(Covar):
    def __init__(self,resolution,rank,dtype= torch.float32,pixel_var_estimate = 1):
        super().__init__(resolution,1,dtype,pixel_var_estimate)
        self.rank = rank
        
        self.current_estimated_rank = 0
    
        self.fixed_vectors = torch.zeros((rank,resolution,resolution,resolution),dtype=self.dtype)
        self.fixed_vectors = torch.nn.Parameter(self.fixed_vectors,requires_grad = False)
        self.fixed_vectors_ampl = torch.nn.Parameter(torch.zeros(rank),requires_grad = False)


    @property
    def eigenvecs(self):
        fixed_vectors = self.fixed_vectors[:self.current_estimated_rank].clone().reshape((self.current_estimated_rank,-1)) * self.fixed_vectors_ampl[:self.current_estimated_rank].reshape((self.current_estimated_rank,-1))
        _,eigenvals,eigenvecs = torch.linalg.svd(fixed_vectors,full_matrices = False)
        eigenvecs = eigenvecs.reshape((self.current_estimated_rank,self.resolution,self.resolution,self.resolution))
        eigenvals = eigenvals ** 2
        return eigenvecs,eigenvals

    def orthogonal_projection(self):
        if(self.current_estimated_rank == 0):
            return
        with torch.no_grad():
            vectors = self.vectors.reshape(1,-1)
            fixed_vectors = self.fixed_vectors[:self.current_estimated_rank].reshape((self.current_estimated_rank,-1))
            coeffs = vectors @ fixed_vectors.T
            self.vectors.data.copy_((vectors - coeffs @ fixed_vectors).view_as(self.vectors))

    def fix_vector(self):
        with torch.no_grad():
            vector = self.vectors.detach().clone()
            vector_ampl = torch.norm(vector).reshape(1)
            normalized_vector = vector/vector_ampl
            
            self.fixed_vectors.data[self.current_estimated_rank] = normalized_vector
            self.fixed_vectors_ampl.data[self.current_estimated_rank] = vector_ampl 
            self.vectors.data.copy_(self.init_random_vectors(1))
            self.current_estimated_rank += 1

    def state_dict(self,*args,**kwargs):
        state_dict = super().state_dict(*args,**kwargs)
        state_dict.update({'current_estimated_rank' : self.current_estimated_rank})
        return state_dict
    
    def load_state_dict(self,state_dict,*args,**kwargs):
        self.current_estimated_rank = state_dict.pop('current_estimated_rank')
        super().load_state_dict(state_dict, *args, **kwargs)
        return

class IterativeCovarTrainerVer2(CovarTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def train(self,*args,**kwargs):
        for i in range(self.covar.rank):
            if(i != 0):
                self.nufft_plans = NufftPlan(self.covar.vectors.shape[1:],batch_size=i+1,dtype = self.covar.vectors.dtype,device=self.device)
            super().train(*args,**kwargs)
            self.covar.fix_vector()
            


class IterativeCovarVer2(IterativeCovar): #TODO : give a better name
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)


    def cost(self,images,nufft_plans,filters,noise_var,reg = 0):
        vectors = torch.cat((self.fixed_vectors[:self.current_estimated_rank],self.vectors),dim=0)
        return cost(vectors,images,nufft_plans,filters,noise_var,reg)
    
