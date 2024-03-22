import torch
from nufft_plan import nufft_forward,nufft_adjoint


def centered_fft2(image,im_dim = [-1,-2]):
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(image,dim=im_dim),dim=im_dim),dim=im_dim)

def centered_ifft2(image,im_dim = [-1,-2]):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(image,dim=im_dim),dim=im_dim),dim=im_dim)

def vol_forward(volume,plan):
    L = volume.shape[-1]
    if(type(plan) == list or type(plan) == tuple): #When mupltiple plans are given loop through them
        volume_forward = torch.zeros((len(plan),volume.shape[0],L,L),dtype = volume.dtype,device = volume.device)
        for i in range(len(plan)):
            volume_forward[i] = vol_forward(volume,plan[i])
        return volume_forward
    else:
        vol_nufft = nufft_forward(volume,plan)
        vol_nufft = vol_nufft.reshape((*volume.shape[:-3],L,L))
        
        if(L % 2 == 0):
            vol_nufft[:,0,:] = 0
            vol_nufft[:,:,0] = 0

        
        volume_forward = centered_ifft2(vol_nufft)

        return torch.real(volume_forward)/L


def im_backward(image,plan):
    L = image.shape[-1]
    im_fft = centered_fft2(image/L**2)

    if(L % 2 == 0):
        im_fft[:,0,:] = 0
        im_fft[:,:,0] = 0

    image_backward = nufft_adjoint(im_fft,plan)

    return torch.real(image_backward)/L

    

    