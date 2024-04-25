import torch
from cufinufft import Plan 
import numpy as np


class NufftPlan():
    def __init__(self,sz,batch_size = 1,eps = 1e-8,dtype = torch.float32,**kwargs):
        self.sz = sz
        self.batch_size = batch_size
        if(dtype == torch.float32 or dtype == torch.complex64):
            self.dtype = torch.float32
            self.complex_dtype = torch.complex64
            np_dtype = np.float32
        elif(dtype == torch.float64 or dtype == torch.complex128):
            self.dtype = torch.float64
            self.complex_dtype = torch.complex128
            np_dtype = np.float64
            
        eps = max(eps, np.finfo(np_dtype).eps) #dtype determines determines the eps bottleneck

        self.forward_plan = Plan(nufft_type = 2,n_modes = self.sz,n_trans=batch_size,eps = eps,dtype=np_dtype,**kwargs)
        self.adjoint_plan = Plan(nufft_type = 1,n_modes = self.sz,n_trans=batch_size,eps = eps,dtype=np_dtype,**kwargs)


    def setpts(self,points):
        #Clean references to past points (preventing a memory leak)
        self.forward_plan._references = []
        self.adjoint_plan._references = []
        points = (torch.remainder(points + torch.pi , 2 * torch.pi) - torch.pi).contiguous()        
        self.forward_plan.setpts(*points)
        self.adjoint_plan.setpts(*points)

    def execute_forward(self,signal):
        signal = signal.type(self.complex_dtype).contiguous()
        forward_signal = self.forward_plan.execute(signal).reshape((self.batch_size,-1))

        return forward_signal
    
    def execute_adjoint(self,signal):
        signal = signal.type(self.complex_dtype).contiguous()
        adjoint_signal = self.adjoint_plan.execute(signal.reshape((self.batch_size,-1)))

        return adjoint_signal
        



class TorchNufftForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,signal,nufft_plan):
        ctx.signal_shape = signal.shape
        ctx.nufft_plan = nufft_plan
        ctx.complex_input = signal.is_complex()
        return nufft_plan.execute_forward(signal)
    
    @staticmethod
    def backward(ctx,grad_output):
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