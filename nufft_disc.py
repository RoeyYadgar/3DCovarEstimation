import torch
from torch.nn.functional import grid_sample


def nufft_disc(volume,points,mode='bilinear'):
    """
    Implementation of NUFFT discretization with nearest/linear interpolation of fourier domain
    """
    #mode should be 'bilinear'/'nearest' (Note : 'bilinear' will actually be 'trilinear' interpolation)
    #Volume should be with shape (N,L,L,L)
    #points should be in range [-1,1] with shape (n,3,L^2)
    
    L = volume.shape[-1]
    points = points.transpose(-2,-1).reshape(1,-1,L,L,3)
    n = points.shape[1]
    points = points.flip(-1)

    #For even image sizes fourier points are [-L/2, ... , L/2-1]/(L/2)*pi while torch grid_sample treats grid as [-1 , ... , 1]
    #For add image sizes fourier points are [-(L-1)/2,...,(L-1)/2]/(L/2)*pi
    if(L % 2 == 0): 
        points = (points + 1/L)
    points*=(L/(L-1))/torch.pi
    
    #For some reason grid_sample does not support complex data. Instead the real and imaginary parts are splitted into different 'channels'
    volume = volume.unsqueeze(1)
    volume_real_imag_split = torch.cat((volume.real,volume.imag),dim=1).reshape(-1,L,L,L) #Shape of (N*2,L,L,L)
    #Grid sample's batch is used when we need to sample different volumes with different grids, here however we want to sample all volumes with different grids so we use the grid_sample channels instead.
    output = grid_sample(input=volume_real_imag_split.unsqueeze(0),grid=points,mode=mode,align_corners=True) #Shape of (1,N*2,n,L,L)

    #Put it back into its complex form
    output = output.reshape(-1,2,n,L,L)
    output = torch.complex(output[:,0],output[:,1]) #Shape of (N,n,L,L)


    return output

    



