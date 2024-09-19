import torch
from cufinufft import Plan as cuPlan 
from finufft import Plan
import numpy as np


class NufftPlan():
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
        

class BatchNufftPlan():
    def __init__(self,batch_plan,sz,batch_size = 1,eps = 1e-6,dtype = torch.float32,device = torch.device('cpu'),**kwargs):
        self.batch_size = batch_plan
        self.nufft_plan = NufftPlan(sz,batch_size,eps,dtype,device,**kwargs)
        self.points = None
        
    def setpts(self,points):
        self.points = points

    def __getitem__(self,index):
        self.nufft_plan.setpts(self.points[index])
        return self.nufft_plan
    
    def __len__(self):
        return self.points.shape[0] if self.points is not None else 0

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
    

nufft_forward = TorchNufftForward.apply
nufft_adjoint = TorchNufftAdjoint.apply