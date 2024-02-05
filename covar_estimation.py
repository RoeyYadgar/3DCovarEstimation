
from aspire.volume import Volume
from aspire.utils import Rotation
import aspire
from aspire.image.image import Image
import numpy as np
from numpy import matmul,mean
from numpy.linalg import norm
import torch



def covar_cost_gradient(vols,src,image_ind,images):
    
    batch_size = images.n_images
    
    vols_forward = vol_stack_forward(vols,src,image_ind,batch_size)
    vols_backproject = im_stack_backward(vols_forwrard, src, image_ind)
    
    vols_forward_norm = norm(vols_forward.asnumpy(), axis = (2,3))
    
    p_u = project_stack(u,rots)
    pbp_u = backproject_stack(p_u, inverted_rots)
    inner_prod_u_pbpu = np.tensordot(pbp_u,u,axes=([2,3,4],[1,2,3]))
    pbpSigma_bpbU_prod = np.tensordot(inner_prod_u_pbpu,pbp_u,axes=([0,1],[0,1]))
    
    
    if(mean_vol is not None):
        p_mean = project_stack(mean_vol,rots)
        zero_mean_projections = projections - p_mean
    else:
        zero_mean_projections = projections
        
            
    bp_y = backproject_stack(
                            (zero_mean_projections).stack_reshape(1,projections.stack_shape[0]),
                             inverted_rots)  
    inner_prod_u_bpy = np.tensordot(bp_y,u,axes=([2,3,4],[1,2,3]))
    bpyCovar_bpY_prod = np.tensordot(inner_prod_u_bpy,bp_y,axes=([0,1],[0,1]))
    
    
    #grad = 4*(bpyCovar_bpY_prod - pbpSigma_bpbU_prod)/((u.resolution**3) * projections.stack_shape[0])
    grad = (bpyCovar_bpY_prod - pbpSigma_bpbU_prod)
    grad = grad/norm(grad)
    
    return grad
    
def covar_cost(projected_vols,projections):
    #TODO check need for ordering
    rank,batch_size,L,_ = projected_vols.shape
    projected_vols = projected_vols.asnumpy().reshape((rank,batch_size,-1))#,order = 'F') 
    projections = projections.asnumpy().reshape((batch_size,-1))#,order='F')
    
    norm_proj_term = mean(np.power(
                norm(projections,axis=(1)),4))
    
    inner_prod_term = np.sum(np.power(np.sum(projected_vols*projections,axis=2),2),axis=(0,1))/batch_size
    
    
    norm_projvols_term = np.sum(np.power(
                np.matmul(projected_vols,projected_vols,axes = [(0,2),(2,0),(0,2)]),2),axis=(0,1,2))/batch_size
    
    #proj_inner_prod = np.matmul(projected_vols,projected_vols,axes = [(0,2),(2,0),(0,2)]).transpose(0,2,1)
    
    cost_val = norm_proj_term - 2*inner_prod_term + norm_projvols_term
    
    return cost_val/(L**4)
    
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
    def forward(ctx, input,target,rotations):
        #ctx.save_for_backward(input,target,rotations)
        ctx.save_for_backward(input)
        ctx.target = target
        ctx.rotations = rotations
        #return torch.tensor(np.linalg.norm(covar_cost_gradient(Volume(input.numpy()),target,rotations)))
        projected_vols = project_stack(Volume(input.numpy()), rotations)
        cost_val = covar_cost(projected_vols,target)
        return torch.tensor(cost_val)
    
    @staticmethod
    def backward(ctx,grad_output):
        #input,target,rotations = ctx.saved_tensors
        input, = ctx.saved_tensors
        target = ctx.target
        rotations = ctx.rotations
        grad_input_np  = (covar_cost_gradient(Volume(input.numpy()),target,rotations)) #TODO take into account mean
        grad_input = torch.tensor(grad_input_np,dtype=torch.float32)
        
    
        return -grad_input, None, None
    
    
    
if __name__ == "__main__":
    from torch.autograd import gradcheck
    input = torch.randn((1,L,L,L), requires_grad=True)
    test = gradcheck(CovarCost.apply, (input,projections,rots), eps=1e-6, atol=1e-4)