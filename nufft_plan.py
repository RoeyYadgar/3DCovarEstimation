import torch
from cufinufft import Plan as cuPlan 
from finufft import Plan
import numpy as np
import math
from abc import ABC

def get_half_fourier_grid(points,dim):
    """
    Converts a 2d rotated grid of points in fourier space into half a grid that represents only positive frequencies in the first dimension.
    This is used to reduce number of points to compute in  NUFFT since real signals have conjugate symmetry in fourier space
    Args:
        points (torch.Tensor) : Batch of 2d rotated grids of size (a,...,b,L,L,c,...,d)
        dim (int) : Two tuple of dimensions of the actual grid (For example for shape (a,b,c,L,L) dim=(3,4))
    Returns:
        half_points (torch.Tensor) : Batch of 2d rotated half-grids of size (a,...,b,L/2,L,c,...,d)
    """
    L = points.shape[dim[0]]
    indices = [slice(None)] * points.ndim
    indices[dim[0]] = slice(L//2,L)
    if(L % 2 == 0): #In even image size fourier points in index 0 corrspond to the 'most negative' frequency which doesnt have a positive counterpart
        indices[dim[1]] = slice(1,L) 
    return points[tuple(indices)]

class BaseNufftPlan(ABC):

    def setpts(self,points):
        raise NotImplementedError

    def execute_forward(self,volume):
        raise NotImplementedError

class NufftPlanDiscretized(BaseNufftPlan):
    """
    Discretized version of the NUFFT operator (specifficaly for the usage of the projection operator and not general case of NUFFT).
    Uses pytorch's `grid_sample` function to interpolate from the fourier tranform of the given volume.
    Assumes input volume is already given in frequency domain.
    """
    def __init__(self,sz,upsample_factor=1,mode='bilinear',use_half_grid = True):
        self.sz = sz
        self.upsample_factor = upsample_factor
        self.mode = mode
        self.use_half_grid = use_half_grid

    def setpts(self,points):
        L = self.sz[0]
        self.points = points.transpose(-2,-1).reshape(1,-1,L,L,3)
        self.points = self.points.flip(-1) #grid_sample uses xyz convention unlike the zyx given by aspire's `rotated_grids`
        self.points = get_half_fourier_grid(self.points,(2,3)) if self.use_half_grid else self.points

        vol_L = L*self.upsample_factor
        #For even image sizes fourier points are [-L/2, ... , L/2-1]/(L/2)*pi while torch grid_sample treats grid as [-1 , ... , 1]
        #For add image sizes fourier points are [-(L-1)/2,...,(L-1)/2]/(L/2)*pi
        self.points *= (vol_L/(vol_L-1)) / torch.pi
        if(vol_L % 2 == 0):
            self.points = (self.points + 1/(vol_L-1))

        self.batch_points = self.points.shape[1]

    @torch.compile
    def execute_forward(self,volume):
        """
        Assumes volume is given in fourier domain with shape (N,L,L,L) either as complex tensor or as tuple pair of real and imag tensors
        """
        L = self.sz[0]
        #For some reason grid_sample does not support complex data. Instead the real and imaginary parts are splitted into different 'channels'
        if(isinstance(volume,tuple)):
            volume_real = volume[0].unsqueeze(1)
            volume_imag = volume[1].unsqueeze(1)
        else:
            volume = volume.unsqueeze(1)
            volume_real = volume.real
            volume_imag = volume.imag
        volume_L = L*self.upsample_factor #Size of volume can be different than the L since might have been upsampled (or downsampled)
        volume_real_imag_split = torch.cat((volume_real,volume_imag),dim=1).reshape(-1,volume_L,volume_L,volume_L) #Shape of (N*2,volume_L,volume_L,volume_L)
        #Grid sample's batch is used when we need to sample different volumes with different grids, here however we want to sample all volumes with different grids so we use the grid_sample channels instead.
        output = torch.nn.functional.grid_sample(input=volume_real_imag_split.unsqueeze(0),grid=self.points,mode=self.mode,align_corners=True) #Shape of (1,N*2,n,L,L) (or (1,N*2,n,L/2,L) if self.use_half_grid=True)

        half_L = math.ceil(L/2) if self.use_half_grid else L
        full_L = (L-1) if (self.use_half_grid and L % 2 == 0) else L
        #Put it back into its complex form
        output = output.reshape(-1,2,self.batch_points,half_L,full_L)
        output = torch.complex(output[:,0],output[:,1]) #Shape of (N,n,L,L) (or (N,n,L/2,L))

        if(self.use_half_grid): #Fill the full grid by flipping & conjugating the output and concateting with original output
            output_full = torch.concat((output[:,:,1:,:].flip(dims=(2,3)).conj(),output),dim=2)
            if(L % 2 == 0): #When image size is even we need to pad the image with additional row and col of zeros
                zeros_row = torch.zeros((output.shape[:2] + (1,L-1)),dtype=output.dtype,device=output.device)
                output_full = torch.concat((zeros_row,output_full),dim=2)
                zeros_col = torch.zeros((output.shape[:2] + (L,1)),dtype=output.dtype,device=output.device)
                output_full = torch.concat((zeros_col,output_full),dim=3) 

            output = output_full
        return output



    
class NufftPlan(BaseNufftPlan):
    def __init__(self,sz,batch_size = 1,eps = 1e-6,dtype = torch.float32,device = torch.device('cpu'),**kwargs):
        self.sz = sz
        self.batch_size = batch_size
        self.device = device
        if(dtype == torch.float32 or dtype == torch.complex64):
            self.dtype = torch.float32
            self.complex_dtype = torch.complex64
            np_dtype = np.float32
        elif(dtype == torch.float64 or dtype == torch.complex128):
            self.dtype = torch.float64
            self.complex_dtype = torch.complex128
            np_dtype = np.float64
            
        eps = max(eps, np.finfo(np_dtype).eps * 10) #dtype determines determines the eps bottleneck

        if(device == torch.device('cpu')):
            self.forward_plan = Plan(nufft_type = 2,n_modes_or_dim=self.sz,n_trans=batch_size, eps = eps, dtype = np_dtype,**kwargs)
            self.adjoint_plan = Plan(nufft_type = 1,n_modes_or_dim=self.sz,n_trans=batch_size, eps = eps, dtype = np_dtype,**kwargs)
        else: #GPU device
            #TODO : check gpu_sort effect
            default_kwargs = {'gpu_method':1,'gpu_sort' : 0}
            default_kwargs.update(kwargs)
            self.forward_plan = cuPlan(nufft_type = 2,n_modes = self.sz,n_trans=batch_size,eps = eps,dtype=np_dtype,gpu_device_id = device.index,**default_kwargs)
            self.adjoint_plan = cuPlan(nufft_type = 1,n_modes = self.sz,n_trans=batch_size,eps = eps,dtype=np_dtype,gpu_device_id = device.index,**default_kwargs)


    def setpts(self,points):
        points = (torch.remainder(points + torch.pi , 2 * torch.pi) - torch.pi).contiguous()
        if(self.device == torch.device('cpu')):
            points = points.numpy() #finufft plan does not accept torch tensors
        #Clean references to past points (preventing a memory leak)
        self.forward_plan._references = []
        self.adjoint_plan._references = []
                
        self.forward_plan.setpts(*points)
        self.adjoint_plan.setpts(*points)

    def execute_forward(self,signal):
        signal = signal.type(self.complex_dtype).contiguous()
        if(self.device == torch.device('cpu')):
            signal = signal.numpy()
        forward_signal = self.forward_plan.execute(signal).reshape((self.batch_size,-1))
        if(self.device == torch.device('cpu')):
            forward_signal = torch.from_numpy(forward_signal)
        return forward_signal
    
    def execute_adjoint(self,signal):
        signal = signal.type(self.complex_dtype).contiguous()
        if(self.device == torch.device('cpu')):
            signal = signal.numpy()
        adjoint_signal = self.adjoint_plan.execute(signal.reshape((self.batch_size,-1)))
        if(self.device == torch.device('cpu')):
            adjoint_signal = torch.from_numpy(adjoint_signal)
        return adjoint_signal
        

class TorchNufftForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,signal,nufft_plan):
        ctx.signal_shape = signal.shape
        ctx.nufft_plan = nufft_plan
        ctx.complex_input = signal.is_complex()
        return nufft_plan.execute_forward(signal)
    
    @staticmethod
    def backward(ctx,grad_output): #Since nufft_plan referenced is saved on the context, set_pts cannot be called before using backward (otherwise it will compute the wrong adjoint transformation)
        nufft_plan = ctx.nufft_plan
        signal_grad = nufft_plan.execute_adjoint(grad_output).reshape(ctx.signal_shape)
        if(not ctx.complex_input): #If input to forward method is real the gradient should also be real
            signal_grad = signal_grad.real
        return signal_grad , None
    

class TorchNufftAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,signal,nufft_plan):
        ctx.signal_shape = signal.shape
        ctx.nufft_plan = nufft_plan
        ctx.complex_input = signal.is_complex()
        return nufft_plan.execute_adjoint(signal)
    
    @staticmethod
    def backward(ctx,grad_output):
        nufft_plan = ctx.nufft_plan
        signal_grad = nufft_plan.execute_forward(grad_output).reshape(ctx.signal_shape)
        if(not ctx.complex_input): #If input to forward method is real the gradient should also be real
            signal_grad = signal_grad.real
        return signal_grad, None
    

def nufft_forward(signal,nufft_plan):
    return TorchNufftForward.apply(signal,nufft_plan) if isinstance(nufft_plan,NufftPlan) else nufft_plan.execute_forward(signal) #NufftPlan.execute_forward does not have autograd and so we must pass it into the autograd class

#TODO : implement adjoint operator for NufftPlanDiscretized? (Not really needed)
nufft_adjoint = TorchNufftAdjoint.apply