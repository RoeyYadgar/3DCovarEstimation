import torch
from cov3d.projection_funcs import centered_fft3,centered_ifft3

class VolumeBase(torch.nn.Module):
    def __init__(self,resolution,dtype=torch.float32,fourier_domain=False,upsampling_factor=2):
        super().__init__()
        self.resolution = resolution
        self.dtype = dtype
        self._in_spatial_domain = not fourier_domain
        self.upsampling_factor = upsampling_factor
        self.grid_correction = None

    def init_grid_correction(self,nufft_disc):
        if(nufft_disc is None):
            self.grid_correction = None
            return
        
        pixel_pos = torch.arange(-(self.resolution // 2),(self.resolution-1)//2+1) / self.resolution
        pixel_pos = torch.pi * pixel_pos / self.upsampling_factor
        sinc_val = torch.sin(pixel_pos)/pixel_pos
        sinc_val[pixel_pos==0]=1
        sinc_val[sinc_val < 1e-6] = 1

        if(nufft_disc == 'bilinear'):
            sinc_val = sinc_val ** 2
        
        sinc_volume = torch.einsum('i,j,k->ijk', sinc_val, sinc_val, sinc_val)
        self.grid_correction = sinc_volume.to(self.device)

    def to(self,*args,**kwargs):
        super().to(*args,**kwargs)
        if(self.grid_correction is not None):
            self.grid_correction = self.grid_correction.to(*args,**kwargs)
        return self   
    
    def state_dict(self,*args,**kwargs):
        state_dict = super().state_dict(*args,**kwargs)
        state_dict.update({'_in_spatial_domain' : self._in_spatial_domain,'grid_correction':self.grid_correction})
        return state_dict
    
    def load_state_dict(self,state_dict,*args,**kwargs):
        self._in_spatial_domain = state_dict.pop('_in_spatial_domain')
        self.grid_correction = state_dict.pop('grid_correction')
        super().load_state_dict(state_dict, *args, **kwargs)
        return


class Mean(VolumeBase):
    def __init__(self,volume_init,resolution,dtype=torch.float32,fourier_domain=False,upsampling_factor=2):
        super().__init__(resolution=resolution,dtype=dtype,fourier_domain=fourier_domain,upsampling_factor=upsampling_factor)
        self.volume = torch.nn.Parameter(volume_init)

    @property
    def device(self):
        return self.volume.device

    def get_volume_fourier_domain(self):
        volume = self.volume / self.grid_correction if self.grid_correction is not None else self.volume
        return centered_fft3(volume,padding_size=(self.resolution*self.upsampling_factor,)*3)
    
    def get_volume_spatial_domain(self):
        return self.volume

    def forward(self,dummy_var = None): #dummy_var is used to make Covar module compatible with DDP - for some reason DDP requires the forward method to have an argument
        return self.get_volume_spatial_domain() if self._in_spatial_domain else self.get_volume_fourier_domain()

class Covar(VolumeBase):
    def __init__(self,resolution,rank,dtype = torch.float32,pixel_var_estimate = 1,fourier_domain = False,upsampling_factor=2,vectors = None):
        super().__init__(resolution=resolution,dtype=dtype,fourier_domain=fourier_domain,upsampling_factor=upsampling_factor)
        self.resolution = resolution
        self.rank = rank
        self.pixel_var_estimate = pixel_var_estimate
        self.dtype = dtype
        self.upsampling_factor = upsampling_factor

        vectors = self.init_random_vectors(rank) if vectors is None else torch.clone(vectors)

        self._in_spatial_domain = not fourier_domain
        self.grid_correction = None
        self.vectors = torch.nn.Parameter(vectors)

    @property
    def device(self):
        return self.vectors.device
    
    def get_vectors(self):
        return self.get_vectors_spatial_domain() if self._in_spatial_domain else self.get_vectors_fourier_domain()
    
    def init_random_vectors(self,num_vectors):
        return (torch.randn((num_vectors,) + (self.resolution,) * 3,dtype=self.dtype)) * (self.pixel_var_estimate ** 0.5)
    
    def init_random_vectors_from_psd(self,num_vectors,psd):
        vectors = (torch.randn((num_vectors,) + (self.resolution,) * 3,dtype=self.dtype))
        vectors_fourier = centered_fft3(vectors) / (self.resolution**1.5)
        vectors_fourier *= torch.sqrt(psd)
        vectors = centered_ifft3(vectors_fourier).real
        return vectors

    def forward(self,dummy_var = None): #dummy_var is used to make Covar module compatible with DDP - for some reason DDP requires the forward method to have an argument
        return self.get_vectors()
    
    @property
    def eigenvecs(self):
        with torch.no_grad():
            vectors = self.get_vectors_spatial_domain().clone().reshape(self.rank,-1)
            _,eigenvals,eigenvecs = torch.linalg.svd(vectors,full_matrices = False)
            eigenvecs = eigenvecs.reshape((self.rank,self.resolution,self.resolution,self.resolution))
            eigenvals = eigenvals ** 2
            return eigenvecs,eigenvals
        
    def get_vectors_fourier_domain(self):
        vectors = self.vectors / self.grid_correction if self.grid_correction is not None else self.vectors
        return centered_fft3(vectors,padding_size=(self.resolution*self.upsampling_factor,)*3)

    def get_vectors_spatial_domain(self):
        return self.vectors

    def orthogonal_projection(self):
        with torch.no_grad():
            vectors = self.get_vectors_spatial_domain().reshape(self.rank,-1)
            _,S,V = torch.linalg.svd(vectors,full_matrices = False)
            orthogonal_vectors  = (S.reshape(-1,1) * V).view_as(self.vectors)
            self.vectors.data.copy_(orthogonal_vectors)


            
