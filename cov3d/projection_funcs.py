import torch
from cov3d.nufft_plan import nufft_forward,nufft_adjoint,BaseNufftPlan
import math

def pad_tensor(tensor,size,dims=None):
    tensor_shape = tensor.shape
    if(dims is None):
        dims = [(-1-i)%tensor.ndim for i in range(len(size))]
    padded_tensor_size = torch.tensor(tensor_shape)
    padded_tensor_size[dims] = torch.tensor(size)
    padded_tensor = torch.zeros(list(padded_tensor_size),dtype=tensor.dtype,device=tensor.device)

    num_dims = len(size)
    start_ind = [math.floor(tensor_shape[dims[i]]/2) - math.floor(size[i]/2) for i in range(num_dims)]
    
    slice_ind = tuple([slice(-start_ind[i],tensor_shape[dims[i]]-start_ind[i]) for i in range(num_dims)])
    slice_ind_full = [slice(tensor.shape[i]) for i in range(tensor.ndim)]
    for i in range(num_dims):
        slice_ind_full[dims[i]] = slice_ind[i]
    padded_tensor[slice_ind_full] = tensor

    return padded_tensor

def crop_tensor(tensor,size,dims=None):
    tensor_shape = tensor.shape
    if(dims is None):
        dims = [(-1-i)%tensor.ndim for i in range(len(size))]
    
    num_dims = len(size)
    start_ind = [math.floor(tensor_shape[dims[i]]/2) - math.floor(size[i]/2) for i in range(num_dims)]
    
    slice_ind = tuple([slice(start_ind[i],size[i]+start_ind[i]) for i in range(num_dims)])
    slice_ind_full = [slice(tensor.shape[i]) for i in range(tensor.ndim)]
    for i in range(num_dims):
        slice_ind_full[dims[i]] = slice_ind[i]

    return tensor[slice_ind_full]


def centered_fft2(image,im_dim = [-1,-2]):
    return _centered_fft(torch.fft.fft2,image,im_dim)

def centered_ifft2(image,im_dim = [-1,-2]):
    return _centered_fft(torch.fft.ifft2,image,im_dim)

def centered_fft3(image,im_dim = [-1,-2,-3],padding_size = None):
    return _centered_fft(torch.fft.fftn,image,im_dim,padding_size)

def centered_ifft3(image,im_dim = [-1,-2,-3],cropping_size = None):
    tensor = _centered_fft(torch.fft.ifftn,image,im_dim)
    return crop_tensor(tensor,cropping_size,im_dim) if cropping_size is not None else tensor
    
def _centered_fft(fft_func,tensor,dim,size=None,**fft_kwargs):
    if(size is not None):
        tensor = pad_tensor(tensor,size,dim)
    return torch.fft.fftshift(fft_func(torch.fft.ifftshift(tensor,dim=dim,**fft_kwargs),dim=dim),dim=dim)


def lowpass_volume(volume,cutoff):
    fourier_vol = centered_fft3(volume)
    L = volume.shape[-1]
    fourier_mask = torch.arange(-L//2,L//2) if L % 2 == 0 else torch.arange(-L//2,L//2) + 1
    fourier_mask = torch.abs(fourier_mask) > cutoff
    fourier_vol[:,fourier_mask,:,:] = 0
    fourier_vol[:,:,fourier_mask,:] = 0
    fourier_vol[:,:,:,fourier_mask] = 0
    return centered_ifft3(fourier_vol).real

def vol_forward(volume,plan,filters = None,fourier_domain = False):
    L = plan.sz[-1]
    if(type(plan) == list or type(plan) == tuple): #When mupltiple plans are given loop through them
        volume_forward = torch.zeros((len(plan),volume.shape[0],L,L),dtype = volume.dtype,device = volume.device)
        for i in range(len(plan)):
            volume_forward[i] = vol_forward(volume,plan[i],filters[i]) if filters is not None else vol_forward(volume,plan[i])
        return volume_forward
    elif(isinstance(plan,BaseNufftPlan)):
        vol_nufft = nufft_forward(volume,plan)
        vol_nufft = vol_nufft.reshape((*volume.shape[:-3],-1,L,L)).transpose(0,1)
        batch_size = vol_nufft.shape[1]
        
        if(L % 2 == 0):
            vol_nufft_clone = vol_nufft.clone()
            vol_nufft_clone[:,:,0,:] = 0
            vol_nufft_clone[:,:,:,0] = 0
        else:
            vol_nufft_clone = vol_nufft

        if(filters is not None):
            vol_nufft_clone = vol_nufft_clone * filters.unsqueeze(1)

        if(batch_size == 1):
            vol_nufft_clone = vol_nufft_clone.squeeze(0)

        volume_forward = centered_ifft2(vol_nufft_clone).real if (not fourier_domain) else vol_nufft_clone

        return volume_forward/L


def im_backward(image,plan):
    L = image.shape[-1]
    im_fft = centered_fft2(image/L**2)

    if(L % 2 == 0):
        im_fft[:,0,:] = 0
        im_fft[:,:,0] = 0

    image_backward = nufft_adjoint(im_fft,plan)

    return torch.real(image_backward)/L

    

    