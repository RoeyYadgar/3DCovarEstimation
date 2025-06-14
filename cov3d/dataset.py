import torch
from dataclasses import dataclass
from torch.utils.data import Dataset
from aspire.utils import support_mask
import numpy as np
import random
import copy
from tqdm import tqdm
from aspire.volume import Volume,rotated_grids
from aspire.utils import Rotation
from cov3d.utils import soft_edged_kernel,get_torch_device,get_complex_real_dtype
from cov3d.nufft_plan import NufftPlan,NufftPlanDiscretized
from cov3d.projection_funcs import vol_forward,im_backward,centered_fft2,centered_ifft2,get_mask_threshold,preprocess_image_batch
from cov3d.fsc_utils import average_fourier_shell,sum_over_shell
from cov3d.covar import Mean
from cov3d.poses import PoseModule

class CovarDataset(Dataset):
    def __init__(self,src,noise_var,mean_volume = None,mask=None,invert_data = False,apply_preprocessing = True):
        self.resolution = src.L
        self.rot_vecs = torch.tensor(Rotation(src.rotations).as_rotvec().astype(src.rotations.dtype))
        self.offsets = torch.tensor(src.offsets,dtype=self.rot_vecs.dtype)
        self.pts_rot = self.compute_pts_rot(self.rot_vecs)
        self.noise_var = noise_var
        self.data_inverted = invert_data
        self._in_spatial_domain = True

        self.filter_indices = torch.tensor(src.filter_indices.astype(int)) #For some reason ASPIRE store filter_indices as string for some star files
        num_filters = len(src.unique_filters)
        self.unique_filters = torch.zeros((num_filters,src.L,src.L))
        for i in range(num_filters):
            self.unique_filters[i] = torch.tensor(src.unique_filters[i].evaluate_grid(src.L))

        self.images = torch.tensor(src.images[:].asnumpy())

        #TODO: signal var should be estimated after removing projected mean but before applying masking
        self.estimate_filters_gain()
        self.estimate_signal_var()
        if(apply_preprocessing):
            self.preprocess_from_modules(*self.construct_mean_pose_modules(mean_volume,mask,self.rot_vecs,self.offsets))
            self.offsets[:] = 0 #After preprocessing images have no offsets

        if(self.data_inverted):
            self.images = -1*self.images



        self.dtype = self.images.dtype
        self.mask = torch.tensor(mask.asnumpy()) if mask is not None else None

        
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
    

    def construct_mean_pose_modules(self,mean_volume,mask,rot_vecs,offsets):
        L = self.resolution
        device = get_torch_device()
        mean = Mean(torch.tensor(mean_volume.asnumpy()),L,
                    fourier_domain=False,
                    volume_mask=torch.tensor(mask.asnumpy()) if mask is not None else None,
                    )
        pose = PoseModule(rot_vecs,offsets,L)
        nufft_plan = NufftPlan((self.resolution,)*3,batch_size = 1, dtype=mean.dtype,device=device)

        return mean,pose,nufft_plan

    def preprocess_from_modules(self,mean_module,pose_module,nufft_plan=None,batch_size=1024):
        device = get_torch_device()
        mean_module = mean_module.to(device)
        pose_module = pose_module.to(device)
        if(nufft_plan is None):
            nufft_plan = NufftPlan((self.resolution,)*3,batch_size = 1, dtype=mean_module.dtype,device=device) if mean_module._in_spatial_domain else \
                    NufftPlanDiscretized((self.resolution,)*3,upsample_factor=mean_module.upsampling_factor, mode='bilinear')

        softening_kernel_fourier = soft_edged_kernel(radius=5,L=self.resolution,dim=2,in_fourier=True).to(device)

        with torch.no_grad():
            mask = mean_module.get_volume_mask()
            mean_volume = mean_module(None)
            idx = torch.arange(min(batch_size,len(self)),device=device)
            nufft_plan.setpts(pose_module(idx)[0].transpose(0,1).reshape((3,-1)))
            mask_threshold = get_mask_threshold(mask,nufft_plan) if mask is not None else 0
            pbar = tqdm(total=np.ceil(len(self)/batch_size), desc=f'Applying preprocessing on dataset images')
            for i in range(0,len(self),batch_size): 
                idx = torch.arange(i,min(i+batch_size,len(self)))
                images,_,filters,_ = self[idx]
                idx = idx.to(device)
                images = images.to(device)
                filters = self.unique_filters[filters].to(device) if len(self.unique_filters) > 0 else None
                if(pose_module.use_contrast):
                    #If pose module containts contrasts - correct images
                    pts_rot,phase_shift,contrasts = pose_module(idx)
                    images = images / contrasts.reshape(-1,1,1)
                else:
                    pts_rot,phase_shift = pose_module(idx)
                self.images[idx] = preprocess_image_batch(images,nufft_plan,filters,
                                                          (pts_rot,phase_shift),mean_volume,
                                                          mask,mask_threshold,softening_kernel_fourier,fourier_domain=not self._in_spatial_domain).cpu()
                
                pbar.update(1)
            pbar.close()

    
    def get_subset(self,idx):
        subset = self.copy()
        subset.images = subset.images[idx]
        subset.pts_rot = subset.pts_rot[idx]
        subset.filter_indices = subset.filter_indices[idx]
        subset.rot_vecs = subset.rot_vecs[idx]
        subset.offsets = subset.offsets[idx]

        return subset
    
    def half_split(self,permute = True):
        data_size = len(self)
        if(permute):
            permutation = torch.randperm(data_size)
        else:
            permutation = torch.arange(0,data_size)

        ds1 = self.get_subset(permutation[:data_size//2])
        ds2 = self.get_subset(permutation[data_size//2:])

        return ds1,ds2,permutation
    

    def get_total_gain(self,batch_size=1024,device=None):
        """
        Returns a 3D tensor represntaing the total gain of each frequency, observed by the dataset = diag(sum(P_i^T P_i))
        """
        L = self.resolution
        upsample_factor=1
        nufft_plan = NufftPlanDiscretized((L,)*3,upsample_factor=upsample_factor,mode='nearest',use_half_grid=False)
        device = get_torch_device() if device is None else device
        gain_tensor = torch.zeros((L*upsample_factor,)*3,device=device,dtype=self.dtype)

        for i in range(0,len(self),batch_size):
            _,pts_rot,filter_indices,_ = self[i:min(i+batch_size,len(self))]
            pts_rot = pts_rot.to(device)
            filters = self.unique_filters[filter_indices].to(device) if self.unique_filters is not None else None

            nufft_plan.setpts(pts_rot.transpose(0,1).reshape((3,-1)))

            gain_tensor += im_backward(torch.complex(filters,torch.zeros_like(filters)),nufft_plan,filters,fourier_domain=True).squeeze().abs()

        gain_tensor /= L

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

class BatchIndexSampler(torch.utils.data.Sampler):
    def __init__(self, data_size, batch_size, shuffle=True,idx=None):
        self.data_size = data_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        if(idx is None):
            idx = list(range(self.data_size))
        else:
            self.data_size = len(idx)
        if self.shuffle:
            random.shuffle(idx)
        self.idx = torch.tensor(idx)

    def __iter__(self):
        for i in range(0, self.data_size, self.batch_size):
            yield self.idx[i:i + self.batch_size]

    def __len__(self):
        return (self.data_size + self.batch_size - 1) // self.batch_size

def identity_collate(batch):
    return batch

def create_dataloader(dataset, batch_size,idx=None, **dataloader_kwargs):
    sampler = dataloader_kwargs.pop('sampler', None)
    if sampler is None:
        sampler = BatchIndexSampler(len(dataset), batch_size, shuffle=False,idx=idx)
    else:
        sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size,drop_last=False)
    batch_size = None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=identity_collate, #Can't use lambda function here since it will be pickled and sent to other processes when using DDP
        **dataloader_kwargs
    )

def get_dataloader_batch_size(dataloader):
    batch_size = dataloader.batch_size
    if(batch_size is None):
        batch_size = dataloader.sampler.batch_size
    return batch_size


@dataclass
class GTData:
    """
    Class to hold the ground truth data to compute metrics against ground truth.
    """
    eigenvecs: torch.Tensor = None
    mean : torch.Tensor = None
    rotations : torch.Tensor = None
    offsets : torch.Tensor = None
    contrasts: torch.Tensor = None

    def __post_init__(self):
        if self.eigenvecs is not None:
            self.eigenvecs = torch.tensor(self.eigenvecs)
        if self.mean is not None:
            self.mean = torch.tensor(self.mean)
        if self.rotations is not None:
            self.rotations = torch.tensor(self.rotations)
        if self.offsets is not None:
            self.offsets = torch.tensor(self.offsets)
        if self.contrasts is not None:
            self.contrasts = torch.tensor(self.contrasts)


    def half_split(self, permutation = None):
        rotations_present = self.rotations is not None
        offsets_present = self.offsets is not None
        contrasts_present = self.contrasts is not None
        if not (rotations_present or offsets_present):
            return self, self
        
        n = self.rotations.shape[0] if rotations_present else self.offsets.shape[0]
        if permutation is None:
            permutation = torch.arange(n)
        perm = permutation[:n//2], permutation[n//2:]

        rotations1 = self.rotations[perm[0]] if rotations_present else None
        offsets1 = self.offsets[perm[0]] if offsets_present else None
        contrasts1 = self.contrasts[perm[0]] if contrasts_present else None
        rotations2 = self.rotations[perm[1]] if rotations_present else None
        offsets2 = self.offsets[perm[1]] if offsets_present else None
        contrasts2 = self.contrasts[perm[1]] if contrasts_present else None

        gt1 = GTData(
            eigenvecs=self.eigenvecs,
            mean=self.mean,
            rotations=rotations1,
            offsets=offsets1,
            contrasts=contrasts1,
        )
        gt2 = GTData(
            eigenvecs=self.eigenvecs,
            mean=self.mean,
            rotations=rotations2,
            offsets=offsets2,
            contrasts=contrasts2,
        )
        return gt1, gt2
        



