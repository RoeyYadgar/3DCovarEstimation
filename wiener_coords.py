import torch
from scipy.stats import chi2
from nufft_plan import BatchNufftPlan
from projection_funcs import vol_forward



def wiener_coords(dataset,eigenvecs,eigenvals,batch_size = 8,start_ind = None,end_ind = None,return_eigen_forward = False):
    if(start_ind is None):
        start_ind = 0
    if(end_ind is None):
        end_ind = len(dataset)
    vol_shape = eigenvecs.shape
    L = vol_shape[-1]
    rank = eigenvecs.shape[0]
    dtype = eigenvecs.dtype
    device = eigenvecs.device
    
    filters = dataset.unique_filters.to(device)
    covar_noise = dataset.noise_var * torch.eye(rank,device = device)
    if(len(eigenvals.shape) == 1):
        eigenvals = torch.diag(eigenvals)

    nufft_plans = BatchNufftPlan(batch_size,(L,)*3,batch_size=rank,dtype = dtype,device=device)
    coords = torch.zeros((end_ind-start_ind,rank),device=device)
    if(return_eigen_forward):
        eigen_forward_images = torch.zeros((end_ind-start_ind,rank,L,L),dtype=dtype)
    for i in range(0,coords.shape[0],batch_size):
        images,pts_rot,filter_indices = dataset[start_ind + i:min(start_ind + i + batch_size,end_ind)]
        num_ims = images.shape[0]
        pts_rot = pts_rot.to(device)
        images = images.to(device).reshape(num_ims,-1)
        batch_filters = filters[filter_indices] if len(filters) > 0 else None
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


    if(not return_eigen_forward):
        return coords
    else:
        return coords,eigen_forward_images



    
def latentMAP(dataset,eigenvecs,eigenvals,batch_size=8,start_ind = None,end_ind = None,return_coords_covar = False):
    if(start_ind is None):
        start_ind = 0
    if(end_ind is None):
        end_ind = len(dataset)
    vol_shape = eigenvecs.shape
    L = vol_shape[-1]
    rank = eigenvecs.shape[0]
    dtype = eigenvecs.dtype
    device = eigenvecs.device
    
    filters = dataset.unique_filters.to(device)
    if(len(eigenvals.shape) == 1):
        eigenvals = torch.diag(eigenvals)

    nufft_plans = BatchNufftPlan(batch_size,(L,)*3,batch_size=rank,dtype = dtype,device=device)
    coords = torch.zeros((end_ind-start_ind,rank),device=device)
    if(return_coords_covar):
        coords_covar = torch.zeros((end_ind-start_ind,rank,rank),dtype=dtype)
    for i in range(0,coords.shape[0],batch_size):
        images,pts_rot,filter_indices = dataset[start_ind + i:min(start_ind + i + batch_size,end_ind)]
        num_ims = images.shape[0]
        pts_rot = pts_rot.to(device)
        images = images.to(device).reshape(num_ims,-1)
        batch_filters = filters[filter_indices] if len(filters) > 0 else None
        nufft_plans.setpts(pts_rot)
        
        eigen_forward = vol_forward(eigenvecs,nufft_plans,batch_filters)

        eigen_forward = eigen_forward.reshape((num_ims,rank,-1))

        for j in range(num_ims):
            m = eigen_forward[j] @ eigen_forward[j].T / dataset.noise_var + torch.inverse(eigenvals)
            im_coor = eigen_forward[j] @ images[j] / dataset.noise_var
            coord_covar = torch.inverse(m)
            coords[i+j,:] = coord_covar @ im_coor
            if(return_coords_covar):
                coords_covar[i+j] = coord_covar.to('cpu')

    if(not return_coords_covar):
        return coords
    else:
        return coords,coords_covar
    

def mahalanobis_distance(coords,coords_mean,coords_covar):
    mean_centered_coords = coords - coords_mean
    dist = torch.sum((mean_centered_coords @ torch.inverse(coords_covar)) * mean_centered_coords,dim=1)

    return dist

def mahalanobis_threshold(coords,coords_mean,coords_covar,q=0.95):
    dist = mahalanobis_distance(coords,coords_mean,coords_covar)
    return dist < chi2.ppf(q,df=coords.shape[1])