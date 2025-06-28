import torch
from scipy.stats import chi2
from cov3d.nufft_plan import NufftPlan,NufftPlanDiscretized
from cov3d.projection_funcs import vol_forward,centered_fft3
from tqdm import tqdm
import math

def wiener_coords(dataset,eigenvecs,eigenvals,batch_size = 1024,start_ind = None,end_ind = None,return_eigen_forward = False):
    if(start_ind is None):
        start_ind = 0
    if(end_ind is None):
        end_ind = len(dataset)
    vol_shape = eigenvecs.shape
    L = vol_shape[-1]
    rank = eigenvecs.shape[0]
    dtype = eigenvecs.dtype
    device = eigenvecs.device
    
    filters = dataset.unique_filters
    if(len(filters) < 10000): #TODO : set the threhsold based on available memory of a single GPU
        filters = filters.to(device)
    covar_noise = dataset.noise_var * torch.eye(rank,device = device)
    if(len(eigenvals.shape) == 1):
        eigenvals = torch.diag(eigenvals)

    nufft_plans = NufftPlan((L,)*3,batch_size=rank,dtype = dtype,device=device)
    coords = torch.zeros((end_ind-start_ind,rank),device=device)
    if(return_eigen_forward):
        eigen_forward_images = torch.zeros((end_ind-start_ind,rank,L,L),dtype=dtype)

    pbar = tqdm(total=math.ceil(coords.shape[0]/batch_size), desc=f'Computing latent coordinates')
    for i in range(0,coords.shape[0],batch_size):
        images,pts_rot,filter_indices,_ = dataset[start_ind + i:min(start_ind + i + batch_size,end_ind)]
        num_ims = images.shape[0]
        pts_rot = pts_rot.to(device)
        images = images.to(device).reshape(num_ims,-1)
        batch_filters = filters[filter_indices].to(device) if len(filters) > 0 else None
        nufft_plans.setpts(pts_rot)
        
        eigen_forward = vol_forward(eigenvecs,nufft_plans,batch_filters)
        if(return_eigen_forward):
            eigen_forward_images[i:i+num_ims] = eigen_forward.to('cpu')
        eigen_forward = eigen_forward.reshape((num_ims,rank,-1))

        for j in range(num_ims):
            eigen_forward_Q , eigen_forward_R = torch.linalg.qr(eigen_forward[j].T)
            image_coor = (images[j] @ eigen_forward_Q)
            image_coor_covar = eigen_forward_R @ eigenvals @ eigen_forward_R.T + covar_noise
            image_coor = eigenvals @ eigen_forward_R.T @ torch.inverse(image_coor_covar) @ image_coor
            coords[i+j,:] = image_coor
        
        pbar.update(1)
    pbar.close()


    if(not return_eigen_forward):
        return coords
    else:
        return coords,eigen_forward_images



    
def latentMAP(dataset,eigenvecs,eigenvals,batch_size=1024,start_ind = None,end_ind = None,return_coords_covar = False,nufft_plan=NufftPlan,**nufft_plan_kwargs):
    if(start_ind is None):
        start_ind = 0
    if(end_ind is None):
        end_ind = len(dataset)
    vol_shape = eigenvecs.shape
    L = vol_shape[-1]
    rank = eigenvecs.shape[0]
    dtype = eigenvecs.dtype
    device = eigenvecs.device
    
    filters = dataset.unique_filters
    if(len(filters) < 10000): #TODO : set the threhsold based on available memory of a single GPU
        filters = filters.to(device)
    if(len(eigenvals.shape) == 1):
        eigenvals = torch.diag(eigenvals)

    eigenvals_inv = torch.inverse(eigenvals + 1e-6 * torch.eye(rank,device=device,dtype=dtype)) #add a small value to avoid numerical instability

    if(nufft_plan == NufftPlan):
        nufft_plans = NufftPlan((L,)*3,batch_size=rank,dtype = dtype,device=device,**nufft_plan_kwargs)
    elif(nufft_plan == NufftPlanDiscretized):
        default_kwargs = {'upsample_factor' : 2, 'mode' : 'bilinear'}
        default_kwargs.update(nufft_plan_kwargs)
        nufft_plans = NufftPlanDiscretized((L,)*3,**default_kwargs)
        eigenvecs = centered_fft3(eigenvecs,padding_size = (L*default_kwargs['upsample_factor'],)*3)

    coords = torch.zeros((end_ind-start_ind,rank),device=device)
    if(return_coords_covar):
        coords_covar_inv = torch.zeros((end_ind-start_ind,rank,rank),dtype=dtype)

    pbar = tqdm(total=math.ceil(coords.shape[0]/batch_size), desc=f'Computing latent coordinates')
    for i in range(0,coords.shape[0],batch_size):
        images,pts_rot,filter_indices,_ = dataset[start_ind + i:min(start_ind + i + batch_size,end_ind)]
        pts_rot = pts_rot.to(device)
        images = images.to(device)
        batch_filters = filters[filter_indices].to(device) if len(filters) > 0 else None
        nufft_plans.setpts(pts_rot)
        
        eigen_forward = vol_forward(eigenvecs,nufft_plans,batch_filters)

        latent_coords,m,_ = compute_latentMAP_batch(images,eigen_forward,dataset.noise_var,eigenvals_inv)
        coords[i:(i+batch_size)] = latent_coords.squeeze(-1)

        if(return_coords_covar):
            coords_covar_inv[i:(i+batch_size)] = m.to('cpu')

        pbar.update(1)
    pbar.close()

    del nufft_plans

    if(not return_coords_covar):
        return coords
    else:
        return coords,coords_covar_inv
    

def compute_latentMAP_batch(images,eigen_forward,noise_var,eigenvals_inv = None):
    n = images.shape[0]
    r = eigen_forward.shape[1]
    if(eigenvals_inv is None):
        eigenvals_inv = torch.eye(r,device=eigen_forward.device,dtype=eigen_forward.dtype)
    images = images.reshape(n,-1,1)
    eigen_forward = eigen_forward.reshape(n,r,-1)

    m = eigen_forward.conj() @ eigen_forward.transpose(1,2) / noise_var + eigenvals_inv
    
    projected_images = torch.matmul(eigen_forward.conj(),images) / noise_var #size (batch, rank,1)

    #There can be numerical instability with inverting the matrix m due to small entries - correct it here by normalizing the matrix by trace(m)/size(m) before inversion
    mean_m = (m.diagonal(dim1=-2,dim2=-1).abs().sum(dim=1)/m.shape[-1])
    latent_coords = torch.linalg.solve(m/mean_m.reshape(-1,1,1),projected_images) / mean_m.reshape(-1,1,1)#size (batch, rank,1)

    return latent_coords,m,projected_images

def mahalanobis_distance(coords,coords_mean,coords_covar_inv):
    mean_centered_coords = coords - coords_mean
    dist = torch.sum((mean_centered_coords @ (coords_covar_inv)) * mean_centered_coords,dim=1)

    return dist

def mahalanobis_threshold(coords,coords_mean,coords_covar_inv,q=0.95):
    dist = mahalanobis_distance(coords,coords_mean,coords_covar_inv)
    return dist < chi2.ppf(q,df=coords.shape[1])