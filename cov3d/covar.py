import torch
from cov3d.projection_funcs import centered_fft3,centered_ifft3,crop_tensor
from cov3d.fsc_utils import expand_fourier_shell

class Covar(torch.nn.Module):
    def __init__(self,resolution,rank,dtype = torch.float32,pixel_var_estimate = 1,fourier_domain = False,upsampling_factor=2,vectors = None):
        super().__init__()
        self.resolution = resolution
        self.rank = rank
        self.pixel_var_estimate = pixel_var_estimate
        self.dtype = dtype
        self.upsampling_factor = upsampling_factor

        if(vectors is None):
            vectors = self.init_random_vectors(rank) if (not isinstance(pixel_var_estimate,torch.Tensor) or pixel_var_estimate.ndim == 0) else self.init_random_vectors_from_psd(rank,self.pixel_var_estimate) 
        else:
            vectors = torch.clone(vectors)

        self._in_spatial_domain = not fourier_domain
        self.grid_correction = None
        self.vectors = torch.nn.Parameter(vectors)

    @property
    def device(self):
        return self.vectors.device
    
    def get_vectors(self):
        return self.get_vectors_spatial_domain() if self._in_spatial_domain else self.get_vectors_fourier_domain()

    def set_vectors(self, new_vectors):
        self.vectors.data.copy_(new_vectors)
    
    def init_random_vectors(self,num_vectors):
        return (torch.randn((num_vectors,) + (self.resolution,) * 3,dtype=self.dtype)) * (self.pixel_var_estimate ** 0.5)
    
    def init_random_vectors_from_psd(self,num_vectors,psd):
        if(psd.ndim == 1):#If psd input is radial
            psd = expand_fourier_shell(psd,self.resolution,3)
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

    @property
    def grad_scale_factor(self):
        return (self.pixel_var_estimate ** 0.5)
        
    def get_vectors_fourier_domain(self):
        vectors = self.vectors / self.grid_correction if self.grid_correction is not None else self.vectors
        return centered_fft3(vectors,padding_size=(self.resolution*self.upsampling_factor,)*3)

    def get_vectors_spatial_domain(self):
        return self.vectors

    def orthogonal_projection(self):
        with torch.no_grad():
            vectors = self.get_vectors_spatial_domain().reshape(self.rank,-1)
            _,S,V = torch.linalg.svd(vectors,full_matrices = False)
            orthogonal_vectors  = (S.reshape(-1,1) * V).reshape(self.rank, self.resolution, self.resolution, self.resolution)
            self.set_vectors(orthogonal_vectors)

    def state_dict(self,*args,**kwargs):
        state_dict = super().state_dict(*args,**kwargs)
        state_dict.update({'_in_spatial_domain' : self._in_spatial_domain,'grid_correction':self.grid_correction})
        return state_dict
    
    def load_state_dict(self,state_dict,*args,**kwargs):
        self._in_spatial_domain = state_dict.pop('_in_spatial_domain')
        self.grid_correction = state_dict.pop('grid_correction')
        super().load_state_dict(state_dict, *args, **kwargs)
        return
            
    def init_grid_correction(self,nufft_disc):
        if(nufft_disc != 'bilinear' and nufft_disc != 'nearest'):
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
    


class CovarFourier(Covar):
    ''' Used to optimize the covariance eigenvecs in Fourier domain. 
    Differs from Covar with `fourier_domain=True` by keeping the underlying vectors in Fourier domain directly.
    '''
    def __init__(self,resolution,rank,dtype = torch.float32,pixel_var_estimate = 1,fourier_domain = False,upsampling_factor=2,vectors = None):
        torch.nn.Module.__init__(self)
        self.resolution = resolution
        self.rank = rank
        self.pixel_var_estimate = pixel_var_estimate
        self.dtype = dtype
        self.upsampling_factor = upsampling_factor

        if(vectors is None):
            vectors = self.init_random_vectors(rank) if (not isinstance(pixel_var_estimate,torch.Tensor) or pixel_var_estimate.ndim == 0) else self.init_random_vectors_from_psd(rank,pixel_var_estimate) 
        else:
            vectors = torch.clone(vectors)
        
        vectors = centered_fft3(vectors,padding_size=(self.resolution*self.upsampling_factor,)*3)

        self.grid_correction = None
        self._vectors_real = torch.nn.Parameter(vectors.real)
        self._vectors_imag = torch.nn.Parameter(vectors.imag)
        self._in_spatial_domain = False

    def set_vectors(self,vectors):
        if(not vectors.is_complex()):
            vectors = centered_fft3(vectors, padding_size=(self.resolution * self.upsampling_factor,) * 3)
        # Store real and imaginary parts separately
        self._vectors_real.data.copy_(vectors.real)
        self._vectors_imag.data.copy_(vectors.imag)

    def get_vectors_spatial_domain(self):
        spatial_vectors = centered_ifft3(torch.complex(self._vectors_real, self._vectors_imag)).real
        return crop_tensor(spatial_vectors, (self.resolution,) * 3,dims=[-1,-2,-3]) / (self.grid_correction if self.grid_correction is not None else 1)
    
    def get_vectors_fourier_domain(self):
        return torch.complex(self._vectors_real, self._vectors_imag)
    
    @property
    def device(self):
        return self._vectors_real.device
    
    @property
    def grad_scale_factor(self):
        return (self.pixel_var_estimate * ((self.upsampling_factor *self.resolution) ** 3)) ** 0.5