import torch
from covar_sgd import Covar,CovarDataset,CovarTrainer
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
            print(f' Device : {self.device} , dataset length : {len(self.train_data.dataset)}')
            coords = wiener_coords(self.train_data.dataset,eigenvecs,eigenvals)
            self.train_data = torch.utils.data.DataLoader(self._orig_dataset.remove_vol_from_images(eigenvecs,coords,copy_dataset = True),
                                                          batch_size = self.batch_size)

class IterativeCovar(Covar):
    def __init__(self,resolution,rank,dtype= torch.float32):
        super().__init__(resolution,1,dtype)
        self.rank = rank
        
        self.current_estimated_rank = 0
    
        self.fixed_vectors = torch.zeros((0,resolution,resolution,resolution),dtype=self.dtype)
        self.fixed_vectors = torch.nn.Parameter(self.fixed_vectors,requires_grad = False)
        self.fixed_vectors_ampl = torch.nn.Parameter(torch.zeros(0),requires_grad = False)


    @property
    def eigenvecs(self):
        fixed_vectors = self.fixed_vectors.clone().reshape((self.current_estimated_rank,-1)) * self.fixed_vectors_ampl.reshape((self.current_estimated_rank,-1))
        _,eigenvals,eigenvecs = torch.linalg.svd(fixed_vectors,full_matrices = False)
        eigenvecs = eigenvecs.reshape(self.fixed_vectors.shape)
        eigenvals = eigenvals ** 2
        return eigenvecs,eigenvals

    def orthogonal_projection(self):
        if(self.current_estimated_rank == 0):
            return
        with torch.no_grad():
            vectors = self.vectors.reshape(1,-1)
            fixed_vectors = self.fixed_vectors.reshape((self.current_estimated_rank,-1))
            coeffs = vectors @ fixed_vectors.T
            self.vectors.data.copy_((vectors - coeffs @ fixed_vectors).view_as(self.vectors))

    def fix_vector(self):
        with torch.no_grad():
            vector = self.vectors.detach().clone()
            vector_ampl = torch.norm(vector).reshape(1)
            normalized_vector = vector/vector_ampl
            self.fixed_vectors = torch.nn.Parameter(torch.cat((self.fixed_vectors,normalized_vector),dim=0), requires_grad=False)
            self.fixed_vectors_ampl = torch.nn.Parameter(torch.cat((self.fixed_vectors_ampl,vector_ampl)), requires_grad=False)
            self.vectors.data.copy_((torch.randn((1,self.resolution,self.resolution,self.resolution),dtype=self.dtype))/(self.resolution ** 1))
            self.current_estimated_rank += 1
