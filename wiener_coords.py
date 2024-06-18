import torch
from nufft_plan import NufftPlan
from projection_funcs import vol_forward



def wiener_coords(dataset,eigenvecs,eigenvals,batch_size = 8):
    vol_shape = eigenvecs.shape
    L = vol_shape[-1]
    rank = eigenvecs.shape[0]
    dtype = eigenvecs.dtype
    device = eigenvecs.device
    
    filters = dataset.unique_filters.to(device)
    covar_noise = dataset.noise_var * torch.eye(rank,device = device)
    if(len(eigenvals.shape) == 1):
        eigenvals = torch.diag(eigenvals)

    nufft_plans = [NufftPlan((L,)*3,batch_size=rank,dtype = dtype,gpu_device_id = device.index,gpu_method = 1,gpu_sort = 0) for i in range(batch_size)]
    coords = torch.zeros((len(dataset),rank),device=device)
    for i in range(0,len(dataset),batch_size):
        images,pts_rot,filter_indices = dataset[i:i+batch_size]
        num_ims = images.shape[0]
        pts_rot = pts_rot.to(device)
        images = images.to(device).reshape(batch_size,-1)
        batch_filters = filters[filter_indices] if len(filters) > 0 else None
        for j in range(num_ims):
            nufft_plans[j].setpts(pts_rot[j])
        
        eigen_forward = vol_forward(eigenvecs,nufft_plans,batch_filters).reshape((batch_size,rank,-1))

        for j in range(num_ims):
            eigen_forward_Q , eigen_forward_R = torch.linalg.qr(eigen_forward[j].T)
            image_coor = (images[j] @ eigen_forward_Q).T
            image_coor_covar = eigen_forward_R @ eigenvals @ eigen_forward_R.T + covar_noise
            image_coor = eigenvals @ eigen_forward_R.T @ torch.inverse(image_coor_covar) @ image_coor
            coords[i+j,:] = image_coor

    return coords



    
