import torch
from cufinufft import Plan as cuPlan 
from finufft import Plan
import numpy as np
from abc import ABC

class NufftPlanAbstract(ABC):

    def setpts(self,points):
        pass

    def execute_forward(self,volume):
        pass

class NufftPlanDiscretized(NufftPlanAbstract):
    """
    Discretized version of the NUFFT operator (specifficaly for the usage of the projection operator and not general case of NUFFT).
    Uses pytorch's `grid_sample` function to interpolate from the fourier tranform of the given volume.
    Assumes input volume is already given in frequency domain.
    """
    def __init__(self,sz,upsample_factor=1,mode='bilinear'):
        self.sz = sz
        self.upsample_factor = upsample_factor
        self.mode = mode

    def setpts(self,points):
        L = self.sz[0]
        self.points = points.transpose(-2,-1).reshape(1,-1,L,L,3)
        self.points = self.points.flip(-1) #grid_sample uses xyz convention unlike the zyx given by aspire's `rotated_grids`

        vol_L = L*self.upsample_factor
        #For even image sizes fourier points are [-L/2, ... , L/2-1]/(L/2)*pi while torch grid_sample treats grid as [-1 , ... , 1]
        #For add image sizes fourier points are [-(L-1)/2,...,(L-1)/2]/(L/2)*pi
        self.points *= (vol_L/(vol_L-1)) / torch.pi
        if(vol_L % 2 == 0):
            self.points = (self.points + 1/(vol_L-1))

        self.batch_points = self.points.shape[1]

    @torch.compile()
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
        output = torch.nn.functional.grid_sample(input=volume_real_imag_split.unsqueeze(0),grid=self.points,mode=self.mode,align_corners=True) #Shape of (1,N*2,n,L,L)

        #Put it back into its complex form
        output = output.reshape(-1,2,self.batch_points,L,L)
        output = torch.complex(output[:,0],output[:,1]) #Shape of (N,n,L,L)
        return output



    
class NufftPlan(NufftPlanAbstract):
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