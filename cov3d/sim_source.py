import torch
import numpy as np
from aspire.utils import Rotation
from aspire.volume import rotated_grids
from aspire.operators import MultiplicativeFilter,ScalarFilter
from cov3d.utils import get_torch_device
from cov3d.nufft_plan import NufftPlan,NufftPlanDiscretized
from cov3d.projection_funcs import vol_forward

class SimulatedSource():
    def __init__(self,n,vols,noise_var,whiten=True,unique_filters = None):
        self.n = n
        self.L = vols.shape[-1]
        self.num_vols = vols.shape[0]
        self.vols = vols
        self.whiten = whiten
        self.noise_var = noise_var        
        self._unique_filters = unique_filters
        self._clean_images = self._gen_clean_images()

    
    @property
    def noise_var(self):
        return self._noise_var if (not self.whiten) else 1
    
    @noise_var.setter
    def noise_var(self,noise_var):
        self._noise_var = noise_var

    @property
    def images(self):
        images = self._clean_images + torch.randn((self.n,self.L,self.L),dtype=self._clean_images.dtype,device=self._clean_images.device) * (self._noise_var ** 0.5)
        if(self.whiten):
            images /= (self._noise_var) ** 0.5

        return images.numpy()

    @property
    def unique_filters(self):
        whiten_filter = ScalarFilter(dim=2,value=self._noise_var ** (-0.5))
        return [MultiplicativeFilter(filt, whiten_filter) for filt in self._unique_filters]

    def _gen_clean_images(self,batch_size=1024):
        clean_images = torch.zeros((self.n,self.L,self.L))
        self.offsets = torch.zeros((self.n,2))
        self.amplitudes = np.ones((self.n))
        self.states = torch.tensor(np.random.choice(self.num_vols,self.n))
        self.filter_indices = np.random.choice(len(self._unique_filters),self.n)
        self.rotations = Rotation.generate_random_rotations(self.n).matrices

        unique_filters = torch.tensor(np.array([self._unique_filters[i].evaluate_grid(self.L) for i in range(len(self._unique_filters))]))
        pts_rot = torch.tensor(rotated_grids(self.L,self.rotations).copy()).reshape((3,self.n,self.L**2))
        pts_rot = pts_rot.transpose(0,1) 
        pts_rot = (torch.remainder(pts_rot + torch.pi , 2 * torch.pi) - torch.pi)

        device = get_torch_device()
        volumes = torch.tensor(self.vols.asnumpy(),device=device)
        nufft_plan = NufftPlan((self.L,)*3,batch_size = 1, dtype=volumes.dtype,device=device)

        for i in range(self.num_vols):
            idx = (self.states == i).nonzero().reshape(-1)
            for j in range(0,len(idx),batch_size):
                batch_ind = idx[j:j+batch_size]
                ptsrot = pts_rot[batch_ind].to(device)
                filter_indices = self.filter_indices[batch_ind]
                filters = unique_filters[filter_indices].to(device)

                nufft_plan.setpts(ptsrot.transpose(0,1).reshape((3,-1)))
                projected_volume = vol_forward(volumes[i].unsqueeze(0),nufft_plan,filters).squeeze(1)

                clean_images[batch_ind] = projected_volume.cpu()

        return clean_images


