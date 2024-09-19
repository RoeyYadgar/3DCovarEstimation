import torch
from nufft_plan import nufft_forward,nufft_adjoint,BatchNufftPlan,NufftPlan


def centered_fft2(image,im_dim = [-1,-2]):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image,dim=im_dim),dim=im_dim),dim=im_dim)

def centered_ifft2(image,im_dim = [-1,-2]):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(image,dim=im_dim),dim=im_dim),dim=im_dim)

def centered_fft3(image,im_dim = [-1,-2,-3]):
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(image,dim=im_dim),dim=im_dim),dim=im_dim)

def centered_ifft3(image,im_dim = [-1,-2,-3]):
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(image,dim=im_dim),dim=im_dim),dim=im_dim)

def vol_forward(volume,plan,filters = None):
    L = volume.shape[-1]
    if(type(plan) == BatchNufftPlan): #When mupltiple plans are given loop through them
        volume_forward = torch.zeros((len(plan),volume.shape[0],L,L),dtype = volume.dtype,device = volume.device)
        for i in range(len(plan)):
            volume_forward[i] = vol_forward(volume,plan[i],filters[i]) if filters is not None else vol_forward(volume,plan[i])
        return volume_forward
    elif(type(plan) == NufftPlan):
        vol_nufft = nufft_forward(volume,plan)
        vol_nufft = vol_nufft.reshape((*volume.shape[:-3],L,L))
        
        if(L % 2 == 0):
            vol_nufft_clone = vol_nufft.clone()
            vol_nufft_clone[:,0,:] = 0
            vol_nufft_clone[:,:,0] = 0
        else:
            vol_nufft_clone = vol_nufft

        if(filters is not None):
            vol_nufft_clone = vol_nufft_clone * filters

        volume_forward = centered_ifft2(vol_nufft_clone)

        return torch.real(volume_forward)/L


def im_backward(image,plan):
    L = image.shape[-1]
    im_fft = centered_fft2(image/L**2)

    if(L % 2 == 0):
        im_fft[:,0,:] = 0
        im_fft[:,:,0] = 0

    image_backward = nufft_adjoint(im_fft,plan)

    return torch.real(image_backward)/L

    

    