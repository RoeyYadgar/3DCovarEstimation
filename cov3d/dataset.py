import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from aspire.utils import support_mask
import numpy as np
import copy
from aspire.volume import Volume,rotated_grids
from aspire.utils import Rotation
from cov3d.utils import soft_edged_kernel,get_torch_device,get_complex_real_dtype
from cov3d.nufft_plan import NufftPlan
from cov3d.projection_funcs import vol_forward,centered_fft2,centered_ifft2
from cov3d.fsc_utils import average_fourier_shell,sum_over_shell


class CovarDataset(Dataset):
    def __init__(self,src,noise_var,mean_volume = None,mask=None,invert_data = False):
        self.resolution = src.L
        self.rot_vecs = torch.tensor(Rotation(src.rotations).as_rotvec().astype(src.rotations.dtype))
        self.pts_rot = self.compute_pts_rot(self.rot_vecs)
        self.noise_var = noise_var
        self.data_inverted = invert_data

        self.filter_indices = torch.tensor(src.filter_indices.astype(int)) #For some reason ASPIRE store filter_indices as string for some star files
        num_filters = len(src.unique_filters)
        self.unique_filters = torch.zeros((num_filters,src.L,src.L))
        for i in range(num_filters):
            self.unique_filters[i] = torch.tensor(src.unique_filters[i].evaluate_grid(src.L))
   
        if(mean_volume is not None):
            self.images = torch.tensor(self.preprocess_images(src,mean_volume))
        else:
            self.images = torch.tensor(src.images[:].asnumpy())
        self.estimate_filters_gain()
        self.estimate_signal_var()
        self.mask_images(mask)

        if(self.data_inverted):
            self.images = -1*self.images

        self.dtype = self.images.dtype
        self._in_spatial_domain = True
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        return self.images[idx] , self.pts_rot[idx] , self.filter_indices[idx],idx
    
    def compute_pts_rot(self,rotvecs):
        rotations = Rotation.from_rotvec(rotvecs.numpy())
        pts_rot = torch.tensor(rotated_grids(self.resolution,rotations.matrices).copy()).reshape((3,rotvecs.shape[0],self.resolution**2)) #TODO : replace this with torch affine_grid with size (N,1,L,L,1)
        pts_rot = pts_rot.transpose(0,1) 
        pts_rot = (torch.remainder(pts_rot + torch.pi , 2 * torch.pi) - torch.pi) #After rotating the grids some of the points can be outside the [-pi , pi]^3 cube

        return pts_rot
    
    def preprocess_images(self,src,mean_volume,batch_size=512):
        device = get_torch_device()
        mean_volume = torch.tensor(mean_volume.asnumpy(),device=device)
        nufft_plan = NufftPlan((self.resolution,)*3,batch_size = 1, dtype=mean_volume.dtype,device=device)

        images = src.images[:]
        images = images.shift(-src.offsets)
        images = images/(src.amplitudes[:,np.newaxis,np.newaxis].astype(images.dtype))
        if(mean_volume is not None): #Substracted projected mean from images. Using own implemenation of volume projection since Aspire implemention is too slow
            for i in range(0,src.n,batch_size): #TODO : do this with own wrapper of nufft to improve run time
                pts_rot = self.pts_rot[i:(i+batch_size)]
                filter_indices = self.filter_indices[i:(i+batch_size)]
                filters = self.unique_filters[filter_indices].to(device) if len(self.unique_filters) > 0 else None
                pts_rot = pts_rot.to(device)
                nufft_plan.setpts(pts_rot.transpose(0,1).reshape((3,-1)))
                projected_mean = vol_forward(mean_volume,nufft_plan,filters).squeeze(1)


                images[i:min(i+batch_size,src.n)] -= projected_mean.cpu().numpy().astype(images.dtype)


        return images
    
    def get_subset(self,idx):
        subset = self.copy()
        subset.images = subset.images[idx]
        subset.pts_rot = subset.pts_rot[idx]
        subset.filter_indices = subset.filter_indices[idx]
        subset.rot_vecs = subset.rot_vecs[idx]

        return subset
    
    def half_split(self,permute = True):
        data_size = len(self)
        if(permute):
            permutation = torch.randperm(data_size)
        else:
            permutation = torch.arange(0,data_size)

        ds1 = self.get_subset(permutation[:data_size//2])
        ds2 = self.get_subset(permutation[data_size//2:])

        return ds1,ds2
    

    def get_total_gain(self):
        """Returns a 3D tensor represntaing the total gain of each frequency, observed by the dataset"""
        pts_rot_grid = torch.round(self.pts_rot / torch.pi * (self.resolution//2)).to(torch.long)
        gain_tensor = torch.zeros((self.resolution,)*3,dtype=self.unique_filters.dtype)
        filters = self.unique_filters[self.filter_indices]
        
        coords = (pts_rot_grid + (self.resolution // 2)) % self.resolution
        indices = coords[:,0] * self.resolution**2 + coords[:,1] * self.resolution + coords[:,2]
        gain_tensor = gain_tensor.view(-1)
        gain_tensor.scatter_add_(0,indices.reshape(-1),filters.reshape(-1)**2)

        gain_tensor = gain_tensor.reshape((self.resolution,)*3)

        return gain_tensor

    def remove_vol_from_images(self,vol,coeffs = None,copy_dataset = False):
        device = vol.device
        num_vols = vol.shape[0]
        if(coeffs is None):
            coeffs = torch.ones(num_vols,len(self))
        dataset = self.copy() if copy_dataset else self

        
        nufft_plan = NufftPlan((self.resolution,)*3,batch_size=num_vols,dtype=vol.dtype,device = device)

        for i in range(len(dataset)):
            _,pts_rot,filter_ind,_ = dataset[i]
            pts_rot = pts_rot.to(device)
            filt = dataset.unique_filters[filter_ind].to(device)
            nufft_plan.setpts(pts_rot)

            vol_proj = vol_forward(vol,nufft_plan,filt).reshape(num_vols,dataset.resolution,dataset.resolution)

            dataset.images[i] -= torch.sum(vol_proj * coeffs[i].reshape((-1,1,1)),dim=0).cpu()

        return dataset

    def copy(self):
        return copy.deepcopy(self)
    

    def to_fourier_domain(self):
        if(self._in_spatial_domain):
            self.images = centered_fft2(self.images)
            #TODO : transform points into grid_sample format here instead of in discretization function?
            self.noise_var *= self.resolution**2 #2-d Fourier transform scales everything by a factor of L (and the variance scaled by L**2)
            self.dtype = get_complex_real_dtype(self.dtype)
            self._in_spatial_domain = False

    def to_spatial_domain(self):
        if(not self._in_spatial_domain):
            self.images = centered_ifft2(self.images).real
            self.noise_var /= self.resolution**2
            self.dtype = get_complex_real_dtype(self.dtype)
            self._in_spatial_domain = True

    def estimate_signal_var(self,support_radius = None,batch_size=512):
        #Estimates the signal variance per pixel
        mask = torch.tensor(support_mask(self.resolution,support_radius))
        mask_size = torch.sum(mask)
        
        signal_psd = torch.zeros((self.resolution,self.resolution))
        for i in range(0,len(self.images),batch_size):
            images_masked = self.images[i:i+batch_size][:,mask]
            images_masked = self.images[i:i+batch_size] * mask
            signal_psd += torch.sum(torch.abs(centered_fft2(images_masked))**2,axis=0)
        signal_psd /= len(self.images) * (self.resolution ** 2) * mask_size
        signal_rpsd = average_fourier_shell(signal_psd)

        noise_psd = torch.ones((self.resolution,self.resolution)) * self.noise_var / (self.resolution**2) 
        noise_rpsd = average_fourier_shell(noise_psd)

        self.signal_rpsd = (signal_rpsd - noise_rpsd)/(self.radial_filters_gain)
        self.signal_rpsd[self.signal_rpsd < 0] = 0 #in low snr setting the estimatoin for high radial resolution might not be accurate enough
        self.signal_var = sum_over_shell(self.signal_rpsd,self.resolution,2).item()
            
    def estimate_filters_gain(self):
        average_filters_gain_spectrum = torch.mean(self.unique_filters ** 2,axis=0) 
        radial_filters_gain = average_fourier_shell(average_filters_gain_spectrum)
        estimated_filters_gain = sum_over_shell(radial_filters_gain,self.resolution,2).item() / (self.resolution**2)

        self.filters_gain = estimated_filters_gain
        self.radial_filters_gain = radial_filters_gain


    def mask_images(self,mask,batch_size=512):
        if(mask is None):
            self.mask = None
            return

        device = get_torch_device()

        self.mask = torch.tensor(mask.asnumpy(),device=device) if isinstance(mask,Volume) else torch.tensor(mask,device=device)

        softening_kernel = soft_edged_kernel(radius=5,L=self.resolution,dim=2)
        softening_kernel = torch.tensor(softening_kernel,device=device)
        softening_kernel_fourier = centered_fft2(softening_kernel)

        nufft_plan = NufftPlan((self.resolution,)*3,batch_size = 1, dtype=self.mask.dtype,device=device)

        for i in range(0,len(self.images),batch_size):
            _,pts_rot,_,_ = self[i:(i+batch_size)]
            pts_rot = pts_rot.to(device)
            nufft_plan.setpts(pts_rot.transpose(0,1).reshape((3,-1)))
            projected_mask = vol_forward(self.mask,nufft_plan).squeeze(1)

            if(i == 0): #Use first batch to determine threshold
                vals = projected_mask.reshape(-1).cpu().numpy()
                threshold = np.percentile(vals[vals > 10 ** (-1.5)],10) #filter values which aren't too close to 0 and take a threhosld that captures 90% of the projected mask
            
            mask_binary = projected_mask > threshold
            mask_binary_fourier = centered_fft2(mask_binary)

            soft_mask_binary = centered_ifft2(mask_binary_fourier * softening_kernel_fourier).real

            self.images[i:min(i+batch_size,len(self.images))] *= soft_mask_binary.cpu()

        self.mask = self.mask.cpu()

@dataclass
class GTData:
    """
    Class to hold the ground truth data to compute metrics against ground truth.
    """
    eigenvecs: torch.Tensor = None
    rotations : torch.Tensor = None

    def __post_init__(self):
        if self.eigenvecs is not None:
            self.eigenvecs = torch.tensor(self.eigenvecs)
        if self.rotations is not None:
            self.rotations = torch.tensor(self.rotations)


