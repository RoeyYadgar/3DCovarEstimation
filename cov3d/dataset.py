from typing import Iterable,Optional,Tuple
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
from cov3d.utils import soft_edged_kernel,get_torch_device,set_module_grad
from cov3d.source import ImageSource
from cov3d.nufft_plan import NufftPlan,NufftPlanDiscretized
from cov3d.projection_funcs import vol_forward,im_backward,centered_fft2,centered_ifft2,get_mask_threshold,preprocess_image_batch
from cov3d.fsc_utils import average_fourier_shell,sum_over_shell
from cov3d.covar import Mean
from cov3d.poses import PoseModule,rotvec_to_rotmat

class CovarDataset(Dataset):
    def __init__(self,src,noise_var,mean_volume = None,mask=None,invert_data = False,apply_preprocessing = True):
        if(isinstance(src,ImageSource)):
            self._init_from_source(src)
        else:
            self._init_from_aspire_source(src)
        self.pts_rot = self.compute_pts_rot(self.rot_vecs)
        self.noise_var = noise_var
        self.data_inverted = invert_data
        self._in_spatial_domain = True

        if(self.data_inverted):
            self.images = -1*self.images

        #TODO: signal var should be estimated after removing projected mean but before applying masking
        self.estimate_filters_gain()
        self.estimate_signal_var()
        if(apply_preprocessing):
            self.preprocess_from_modules(*self.construct_mean_pose_modules(mean_volume,mask,self.rot_vecs,self.offsets))
            self.offsets[:] = 0 #After preprocessing images have no offsets


        self.dtype = self.images.dtype
        self.mask = torch.tensor(mask.asnumpy()) if mask is not None else None

    def _init_from_source(self,source):
        self.resolution = source.resolution
        #TODO: replace with non ASPIRE implemntation?
        self.rot_vecs = torch.tensor(Rotation(source.rotations.numpy()).as_rotvec(),dtype=source.rotations.dtype)
        self.offsets = source.offsets
        self.images = source.images(torch.arange(0,len(source)))
        self.filters = source.get_ctf(torch.arange(0,len(source)))

    def _init_from_aspire_source(self,source):
        self.resolution = source.L
        self.rot_vecs = torch.tensor(Rotation(source.rotations).as_rotvec().astype(source.rotations.dtype))
        self.offsets = torch.tensor(source.offsets,dtype=self.rot_vecs.dtype)
        self.images = torch.tensor(source.images[:].asnumpy())

        filter_indices = torch.tensor(source.filter_indices.astype(int)) #For some reason ASPIRE store filter_indices as string for some star files
        num_filters = len(source.unique_filters)
        unique_filters = torch.zeros((num_filters,source.L,source.L))
        for i in range(num_filters):
            unique_filters[i] = torch.tensor(source.unique_filters[i].evaluate_grid(source.L))

        self.filters = unique_filters[filter_indices]

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self,idx):
        return self.images[idx] , self.pts_rot[idx] , self.filters[idx],idx
    
    def compute_pts_rot(self,rotvecs):
        rotations = Rotation.from_rotvec(rotvecs.numpy())
        pts_rot = torch.tensor(rotated_grids(self.resolution,rotations.matrices).copy()).reshape((3,rotvecs.shape[0],self.resolution**2)) #TODO : replace this with torch affine_grid with size (N,1,L,L,1)
        pts_rot = pts_rot.transpose(0,1) 
        pts_rot = (torch.remainder(pts_rot + torch.pi , 2 * torch.pi) - torch.pi) #After rotating the grids some of the points can be outside the [-pi , pi]^3 cube

        return pts_rot
    

    def construct_mean_pose_modules(self,mean_volume,mask,rot_vecs,offsets,fourier_domain=False):
        L = self.resolution
        device = get_torch_device()

        if isinstance(mask,Volume):
            mask = torch.tensor(mask.asnumpy())

        mean = Mean(torch.tensor(mean_volume.asnumpy()),L,
                    fourier_domain=fourier_domain,
                    volume_mask= mask if mask is not None else None,
                    )
        pose = PoseModule(rot_vecs,offsets,L)
        nufft_plan = NufftPlan((self.resolution,)*3,batch_size = 1, dtype=mean.dtype,device=device) if not fourier_domain else \
                NufftPlanDiscretized((self.resolution,)*3,upsample_factor=mean.upsampling_factor, mode='bilinear')
        if fourier_domain:
            mean.init_grid_correction('bilinear')

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
            nufft_plan.setpts(pose_module(idx)[0])
            mask_threshold = get_mask_threshold(mask,nufft_plan) if mask is not None else 0
            pbar = tqdm(total=np.ceil(len(self)/batch_size), desc=f'Applying preprocessing on dataset images')
            for i in range(0,len(self),batch_size): 
                idx = torch.arange(i,min(i+batch_size,len(self)))
                images,_,filters,_ = self[idx]
                idx = idx.to(device)
                images = images.to(device)
                filters = filters.to(device)
                if(pose_module.use_contrast):
                    #If pose module containts contrasts - correct images
                    #TODO: The ideal way to use it is to apply the contrast on the filters.
                    # However, the dataset only contains the unique filters.
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
        subset.filters = subset.filters[idx]
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
            _,pts_rot,filters,_ = self[i:min(i+batch_size,len(self))]
            pts_rot = pts_rot.to(device)
            filters = filters.to(device)

            nufft_plan.setpts(pts_rot)

            gain_tensor += im_backward(torch.complex(filters,torch.zeros_like(filters)),nufft_plan,filters,fourier_domain=True).squeeze().abs()

        gain_tensor /= L

        return gain_tensor


    def get_total_covar_gain(self,batch_size=None,device=None):
        """
        Returns a 2D tensor represnting the total gain of each frequency pair in the covariance least squares problem.
        """
        
        L = self.resolution
        upsample_factor=1
        nufft_plan = NufftPlanDiscretized((L,)*3,upsample_factor=upsample_factor,mode='nearest',use_half_grid=False)
        device = get_torch_device() if device is None else device
        gain_tensor = torch.zeros((L*upsample_factor,)*3,device=device,dtype=self.dtype)

        s = average_fourier_shell(gain_tensor).shape[0]
        covar_shell_gain = torch.zeros((s,s),device=device,dtype=self.dtype)

        torch.cuda.empty_cache()

        if batch_size is None:
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            available_memory = total_mem - reserved_mem
            available_batch_size = available_memory / (L**3 * self.dtype.itemsize*2) # self.dtype.itemsize*2 bytes per value for complex dtype
            
            batch_size = int(available_batch_size / 6)
            batch_size = min(batch_size,256)
            print(f'Using batch size of {batch_size} to compute dataset covar gain')
            print((torch.cuda.memory_reserved(device),torch.cuda.memory_allocated(device)))


        for i in range(0,len(self),batch_size):
            _,pts_rot,filters,_ = self[i:min(i+batch_size,len(self))]
            pts_rot = pts_rot.to(device)
            filters = filters.to(device)

            nufft_plan.setpts(pts_rot)

            if(L % 2 == 0):
                filters[:,0,:] = 0
                filters[:,:,0] = 0
            gain_tensor = nufft_plan.execute_adjoint_unaggregated(torch.complex(filters**2,torch.zeros_like(filters))).abs() / L**2

            averaged_gain_tensor = average_fourier_shell(*gain_tensor)
            covar_shell_gain += averaged_gain_tensor.T @ averaged_gain_tensor

        torch.cuda.empty_cache()

        return covar_shell_gain

    def copy(self):
        return copy.deepcopy(self)
    

    def to_fourier_domain(self):
        if(self._in_spatial_domain):
            self.images = centered_fft2(self.images)
            #TODO : transform points into grid_sample format here instead of in discretization function?
            self.noise_var *= self.resolution**2 #2-d Fourier transform scales everything by a factor of L (and the variance scaled by L**2)
            self._in_spatial_domain = False

    def to_spatial_domain(self):
        if(not self._in_spatial_domain):
            self.images = centered_ifft2(self.images).real
            self.noise_var /= self.resolution**2
            self._in_spatial_domain = True

    def estimate_signal_var(self,support_radius = None,batch_size=512):
        #Estimates the signal variance per pixel
        mask = torch.tensor(support_mask(self.resolution,support_radius))
        mask_size = torch.sum(mask)
        
        signal_psd = torch.zeros((self.resolution,self.resolution))
        for i in range(0,len(self),batch_size):
            images_masked = self._get_images_for_signal_var(i, batch_size) * mask
            signal_psd += torch.sum(torch.abs(centered_fft2(images_masked))**2,axis=0)
        signal_psd /= len(self) * (self.resolution ** 2) * mask_size
        signal_rpsd = average_fourier_shell(signal_psd)

        noise_psd = torch.ones((self.resolution,self.resolution)) * self.noise_var / (self.resolution**2) 
        noise_rpsd = average_fourier_shell(noise_psd)

        self.signal_rpsd = (signal_rpsd - noise_rpsd)/(self.radial_filters_gain)
        self.signal_rpsd[self.signal_rpsd < 0] = 0 #in low snr setting the estimatoin for high radial resolution might not be accurate enough
        self.signal_var = sum_over_shell(self.signal_rpsd,self.resolution,2).item()
    
    def _get_images_for_signal_var(self, start_idx, batch_size):
        """Helper method to get images for signal variance estimation.
        Subclasses can override this to provide different image access patterns."""
        end_idx = min(start_idx + batch_size, len(self))
        return self.images[start_idx:end_idx]
            
    def estimate_filters_gain(self,batch_size=1024):

        average_filters_gain_spectrum = torch.zeros((self.resolution,self.resolution))
        for i in range(0,len(self),batch_size):
            filters = self._get_filters_for_filters_gain(i,batch_size)
            average_filters_gain_spectrum += torch.sum(filters ** 2,axis=0)
        average_filters_gain_spectrum /= len(self)

        radial_filters_gain = average_fourier_shell(average_filters_gain_spectrum)
        estimated_filters_gain = sum_over_shell(radial_filters_gain,self.resolution,2).item() / (self.resolution**2)

        self.filters_gain = estimated_filters_gain
        self.radial_filters_gain = radial_filters_gain

    def _get_filters_for_filters_gain(self, start_idx, batch_size):
        """Helper method to get filters for filter gain estimation.
        Subclasses can override this to provide different CTF access patterns."""
        end_idx = min(start_idx + batch_size, len(self))
        return self.filters[start_idx:end_idx]


    def update_pose(self,pose_module : PoseModule,batch_size : int = 1024):
        """Updates dataset's particle pose information from a given PoseModule

        Args:
            pose_module (PoseModule): PoseModule instance to update from
        """
        with torch.no_grad():
            for i in range(0,len(self),batch_size):                
                idx = torch.arange(i,min(i + batch_size,len(self)),device=pose_module.device)
                self.pts_rot[idx.cpu()] = pose_module(idx)[0].detach().cpu()
                self.offsets[idx.cpu()] = pose_module.get_offsets()[idx.cpu()].detach().cpu()

class LazyCovarDataset(CovarDataset):
    def __init__(self,src,noise_var,mean_volume = None,mask=None,invert_data = False,apply_preprocessing = True):
        if not isinstance(src,ImageSource):
            raise ValueError(f'input src is of type {type(src)}. LazyCovarDataset only supports ImageSource')
        self.src = src
        self.resolution = self.src.resolution
        self.noise_var = noise_var
        self.data_inverted = invert_data
        self._in_spatial_domain = True
        self.apply_preprocessing = apply_preprocessing

        self.estimate_filters_gain()
        self.estimate_signal_var()

        self.mask = torch.tensor(mask.asnumpy()) if mask is not None else None
        self.mean_volume = mean_volume

        #Decalare additional attributes that will be initialized in post_init_setup
        self._mean_volume = None
        self._mask = None
        self._pose_module = None
        self._nufft_plan = None
        self._softening_kernel_fourier = None
        self._mask_threshold = None

    @property
    def dtype(self):
        return self.src.dtype


    def post_init_setup(self,fourier_domain):
        """Performs additional setup after constructor.
        It inits a nufft plan that is used internally to compute projections of the mean volume and mask.
        This must happen after class construction since when we use DDP we pass this object which cannot have
        tensors already on the GPU.
        """
        #TODO: should better handle case where apply_preprocessing=False, this will use uncessery GPU mem
        rot_vecs = torch.tensor(Rotation(self.src.rotations).as_rotvec(),dtype=self.dtype) #TODO: use a torch implementation?
        mean_module,pose_module,nufft_plan = self.construct_mean_pose_modules(self.mean_volume,self.mask,rot_vecs,self.src.offsets,fourier_domain=fourier_domain)
        self._set_internal_preprocessing_modules(mean_module,pose_module,nufft_plan)


    def preprocess_from_modules(self, mean_module, pose_module, nufft_plan=None, batch_size=1024):
        """Overrides superclass method, since this is a lazy dataset implementation, this does not actually perform any preprocessing
            but it update the internal objects required preprocessing on demand.
        """

        mean_module = copy.deepcopy(mean_module)
        pose_module = copy.deepcopy(pose_module)

        if mean_module._in_spatial_domain != self._in_spatial_domain:
            domain_name = lambda val : 'Spatial' if val else 'Fourier'
            print(f'Warning: Mean module is in {domain_name(mean_module._in_spatial_domain)} domain while dataset is in {domain_name(self._in_spatial_domain)}. Changing domain of the mean mean module to fit dataset')

            mean_module._in_spatial_domain = self._in_spatial_domain


        device = get_torch_device()
        if(nufft_plan is None):
            nufft_plan = NufftPlan((self.resolution,)*3,batch_size = 1, dtype=mean_module.dtype,device=device) if mean_module._in_spatial_domain else \
                    NufftPlanDiscretized((self.resolution,)*3,upsample_factor=mean_module.upsampling_factor, mode='bilinear')

        self._set_internal_preprocessing_modules(mean_module,pose_module,nufft_plan)

    def _set_internal_preprocessing_modules(self,mean_module,pose_module,nufft_plan):
        set_module_grad(mean_module,False)
        set_module_grad(pose_module,False)
        device = get_torch_device()
        self.src = self.src.to(device)
        mean_module = mean_module.to(device)

        #_mean_volume and _mask are different than the original mean_volume and mask in that they are in fourier domain(if fourier_domain is True)
        self._mean_volume = mean_module()
        self._mask = mean_module.get_volume_mask()


        self._pose_module = pose_module.to(device)
        self._nufft_plan = nufft_plan


        #Compute mask related variables
        with torch.no_grad():
            idx = torch.arange(min(1024,len(self)),device=device)
            nufft_plan.setpts(pose_module(idx)[0])
            self._mask_threshold = get_mask_threshold(self._mask,nufft_plan) if self._mask is not None else 0
            #Mask softening kernel should be in fourier space regardless of the value of fourier_domain
            self._softening_kernel_fourier = soft_edged_kernel(radius=5,L=self.resolution,dim=2,in_fourier=True).to(device)

    def __len__(self):
        return len(self.src)


    def __getitem__(self,idx):
        return self._get_images(idx) + (idx,)

    def _get_images(self,idx : Iterable,filters : Optional[torch.Tensor] = None) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        """Returns images by idx after pre-processing (if needed) which includes substracting projected mean and masking.

        Args:
            idx (Iterable): image index to be returned
            filters (Optional[torch.Tensor]): CTF filters to be used when computing the mean projection.
                If None are given CTFs are computed by the corresponding image idx.

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: Tuple containing images,rotated grid of fourier components and filters
        """
        device = self._mean_volume.device
        images = self.src.images(idx,fourier = not self._in_spatial_domain)

        image_sign = -1 if self.data_inverted else 1
        images *= image_sign

        #Compute CTF if not provided
        if filters is None:
            filters = self.src.get_ctf(idx)

        if not isinstance(idx,torch.Tensor):
            if isinstance(idx, slice):
                idx = torch.arange(len(self.src))[idx]
            else: 
                idx = torch.tensor(idx)

        idx = idx.to(device)

        #Compute pts
        if self._pose_module.use_contrast:
            pts_rot,phase_shift,contrasts = self._pose_module(idx)
            filters = filters * contrasts.reshape(-1,1,1) #TODO: need to handle case where filters are provided, should contrast be applied?
        else:
            pts_rot,phase_shift = self._pose_module(idx)

        if not self.apply_preprocessing:
            return images,pts_rot,filters

        images = images.to(device)
        filters = filters.to(device)

        with torch.no_grad():
            images = preprocess_image_batch(images,self._nufft_plan,filters,
                                            (pts_rot,phase_shift),self._mean_volume,
                                            self._mask,self._mask_threshold,self._softening_kernel_fourier,fourier_domain=not self._in_spatial_domain)

        return images,pts_rot,filters

    @property
    def filters(self):
        return self.src.get_ctf(torch.arange(0,len(self)))


    @property
    def rot_vecs(self):
        return torch.tensor(Rotation(self.src.rotations.numpy()).as_rotvec(),dtype=self.src.rotations.dtype)

    @property
    def offsets(self):
        return self.src.offsets

    def _get_images_for_signal_var(self, start_idx, batch_size):
        """Override to use lazy loading for signal variance estimation."""
        end_idx = min(start_idx + batch_size, len(self))
        return self.src.images(torch.arange(start_idx, end_idx))

    def _get_filters_for_filters_gain(self, start_idx, batch_size):
        """Override to use lazy loading for signal variance estimation."""
        end_idx = min(start_idx + batch_size, len(self))
        return self.src.get_ctf(torch.arange(start_idx, end_idx))

    def get_subset(self,idx):
        subset = self.copy()
        subset.src = self.src.get_subset(idx)

        #Create a new pose module for the subset
        if self._pose_module is not None:
            rot_vecs = torch.tensor(Rotation(subset.src.rotations).as_rotvec(),dtype=self.dtype)
            offsets = subset.src.offsets
            subset._pose_module = PoseModule(rot_vecs,offsets,self.resolution)


        return subset

    def to_fourier_domain(self):
        if self._in_spatial_domain:
            self.post_init_setup(fourier_domain=True)
            self._in_spatial_domain = False


    def to_spatial_domain(self):
        if not self._in_spatial_domain:
            self.post_init_setup(fourier_domain=False)
            self._in_spatial_domain = True

    def update_pose(self,pose_module : PoseModule,batch_size : int = None):
        """Updates dataset's particle pose information from a given PoseModule

        Args:
            pose_module (PoseModule): PoseModule instance to update from
        """
        rot_vecs = pose_module.get_rotvecs().detach()
        offsets = pose_module.get_offsets().detach().cpu()

        self.src.rotations = rotvec_to_rotmat(rot_vecs).cpu()
        self.src.offsets = offsets

        #Update pose on exisiting internal pose module
        if self._pose_module is not None:
            self._pose_module.set_rotvecs(rot_vecs)
            self._pose_module.set_offsets(offsets)

            if (self._pose_module.use_contrast and pose_module.use_contrast):
                self._pose_module.set_contrasts(pose_module.get_contrasts().detach().cpu())
        

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

    #Cannot use num workers > 1 with lazy dataset since it requires GPU usage
    #TODO: find a better solution that enables the use of more workers
    num_workers = dataloader_kwargs.pop('num_workers',0)
    if (isinstance(dataset,LazyCovarDataset)):
        print(f'Warning: cannot use {num_workers} > 1 num_workers with Lazy dataset. setting num_workers to 0 and prefetch_factor to None')
        dataloader_kwargs['prefetch_factor'] = None
        dataloader_kwargs['persistent_workers'] = False
        dataloader_kwargs['pin_memory'] = False
        dataloader_kwargs['pin_memory_device'] = ''
        num_workers = 0

    batch_size = None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
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
        



