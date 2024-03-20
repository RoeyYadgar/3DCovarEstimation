import torch
from cufinufft import Plan 



class Nufft():
    def __init__(self,sz,batch_size,eps = 1e-8,**kwargs):
        self.sz = sz
        self.batch_size = batch_size
        self.forward_plan = Plan(nufft_type = 2,n_modes = self.sz,n_trans=batch_size,eps = eps,**kwargs)
        self.adjoint_plan = Plan(nufft_type = 1,n_modes = self.sz,n_trans=batch_size,eps = eps,**kwargs)


    def setpts(self,points):
        points = (torch.remainder(points + torch.pi , 2 * torch.pi) - torch.pi).contiguous()
        self.forward_plan.setpts(*points)
        self.adjoint_plan.setpts(*points)

    def execute_forward(self,signal):
        forward_signal = self.forward_plan.execute(signal).reshape((self.batch_size,-1))
        '''
        if(self.batch_size == 1):
            return forward_signal[0]
        else:
            return forward_signal
        '''
        return forward_signal
    
    def execute_adjoint(self,signal):
        adjoint_signal = self.adjoint_plan.execute(signal.reshape((self.batch_size,-1)))
        if(self.batch_size == 1):
            return adjoint_signal[0]
        else:
            return adjoint_signal
        



class TorchNufftForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx,signal,nufft_plan):
        ctx.signal_shape = signal.shape
        ctx.nufft_plan = nufft_plan
        return nufft_plan.execute_forward(signal)
    
    @staticmethod
    def backward(ctx,grad_output):
        nufft_plan = ctx.nufft_plan
        return nufft_plan.execute_adjoint(grad_output).reshape(ctx.signal_shape) , None
    

class TorchNufftAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,signal,nufft_plan):
        ctx.signal_shape = signal.shape
        ctx.nufft_plan = nufft_plan
        return nufft_plan.execute_adjoint(signal)
    
    @staticmethod
    def backward(ctx,grad_output):
        nufft_plan = ctx.nufft_plan
        return nufft_plan.execute_forward(grad_output).reshape(ctx.signal_shape) , None
    

nufft_forward = TorchNufftForward.apply
nufft_adjoint = TorchNufftAdjoint.apply