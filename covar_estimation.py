
from aspire.volume import Volume
from aspire.utils import Rotation
import aspire
from aspire.image.image import Image
import numpy as np
from numpy import matmul,mean,power
from numpy.linalg import norm
import torch
from utils import *


def covar_cost_gradient(vols,vols_backproject,images_backproject,images_volsforward_prod ,volsforward_prod):
    rank,batch_size,L,_,_ = vols_backproject.shape
    
    vols_backproject = vols_backproject.asnumpy().reshape((rank,batch_size,-1))
    images_backproject = images_backproject.asnumpy().reshape((1,batch_size,-1))
    
    batch_size = images_backproject.shape[1]
    
    #newvols = np.zeros(vols_backproject.shape)
    grad = (np.matmul(volsforward_prod,vols_backproject,axes=[(0,1),(0,2),(0,2)])
            - np.matmul(images_volsforward_prod,images_backproject,axes=[(0,1),(0,2),(0,2)]))
    
    
    grad = 4*np.mean(grad,axis=(1)).reshape(rank,L,L,L)
    
    
    
    return grad/(L**3)
    
def covar_cost(vols_forward,images):
    #TODO check need for ordering in reshape
    
    rank,batch_size,L,_ = vols_forward.shape
    
    
    vols_forward_np = vols_forward.asnumpy().reshape((rank,batch_size,-1))
    images_np = images.asnumpy().reshape((1,batch_size,-1))
    
    norm_images_term = np.power(norm(images_np,axis=(2)),4)
    
    images_volsforward_prod = np.matmul(vols_forward_np,images_np,axes = [(0,2),(2,0),(0,2)]).transpose(0,2,1)
    
    volsforward_prod = np.matmul(vols_forward_np,vols_forward_np,axes = [(0,2),(2,0),(0,2)]).transpose(0,2,1)
    

    
    cost_val = (norm_images_term 
                - 2*np.sum(np.power(images_volsforward_prod,2),axis=0)
                + np.sum(np.power(volsforward_prod,2),axis=(0,1)))
                
    
    return np.mean(cost_val)/(L**2) , images_volsforward_prod , volsforward_prod
    


def vol_stack_forward(vols,src,image_ind,image_num):
    L = vols.resolution
    vols_num = len(vols)
    vols_forward = Image(np.zeros((vols_num,image_num,L,L),dtype=vols.dtype))
    
    for i in range(vols_num):
        vols_forward[i] = src.vol_forward(vols[i],image_ind,image_num)
    
    
    return vols_forward
    
def im_stack_backward(ims,src,image_ind):
    
    L = ims.resolution
    
    stack_shape = ims.stack_shape
    im_backward = Volume(np.zeros(np.concatenate((stack_shape,(L,L,L))),dtype=ims.dtype))
    
    for i in range(stack_shape[0]):
        for j in range(stack_shape[1]):
            
            im_backward[i,j] = src.im_backward(ims[i][j],image_ind+j).T
            
            
    
    return im_backward
    




class CovarCost(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input,src,image_ind,images):
        
        #ctx.save_for_backward(input)
        ctx.images = images
        ctx.src = src
        ctx.image_ind = image_ind
        
        vols = Volume(input.numpy())
        vols_forward = vol_stack_forward(vols,src,image_ind,images.shape[0])
        
        ctx.vols = vols
        ctx.vols_forward = vols_forward
        
        cost_val,images_volsforward_prod , volsforward_prod = covar_cost(vols_forward,images)
        
        ctx.images_volsforward_prod = images_volsforward_prod
        ctx.volsforward_prod = volsforward_prod
        
        return torch.tensor(cost_val)
    
    @staticmethod
    def backward(ctx,grad_output):
        #input,target,rotations = ctx.saved_tensors
        batch_size = ctx.images.shape[0]
        vols_backproject = im_stack_backward(ctx.vols_forward, ctx.src, ctx.image_ind)
        images_backproject = im_stack_backward(ctx.images.stack_reshape((1,batch_size)), ctx.src, ctx.image_ind)
        
        
        grad_input_np = covar_cost_gradient(ctx.vols, vols_backproject, images_backproject, ctx.images_volsforward_prod, ctx.volsforward_prod)
        grad_input = torch.tensor(grad_input_np,dtype= np2torchDtype(ctx.vols.dtype))
        
    
        return grad_input * grad_output, None, None,None
    
    
    