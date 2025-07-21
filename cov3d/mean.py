import torch
import torch.distributed as dist
from torch.utils.data import DataLoader 
from cov3d.projection_funcs import im_backward,centered_ifft3,centered_fft3
from cov3d.dataset import CovarDataset,create_dataloader
from cov3d.nufft_plan import NufftPlanDiscretized
from cov3d.poses import get_phase_shift_grid,offset_to_phase_shift
from cov3d.fsc_utils import rpsd,upsample_and_expand_fourier_shell,average_fourier_shell,vol_fsc
from cov3d.utils import get_torch_device
from tqdm import tqdm
from typing import Union

def reconstruct_mean(dataset : Union[CovarDataset,DataLoader],init_vol = None,mask=None,upsampling_factor=2,batch_size=1024,idx=None,return_lhs_rhs=False):
    
    if(not isinstance(dataset,DataLoader)):
        if(idx is None):
            idx = torch.arange(len(dataset))
        dataloader = create_dataloader(dataset,batch_size=batch_size,idx=idx,pin_memory=True)
    else:
        dataloader = dataset
        assert idx is None, "If input dataset is a dataloader, idx cannot be specified"
    dataset = dataloader.dataset

    device = get_torch_device() if init_vol is None else init_vol.device
        

    L = dataset.resolution
    nufft_plan = NufftPlanDiscretized((L,)*3,upsample_factor=upsampling_factor,mode='nearest',use_half_grid=False)

    is_dataset_in_fourier = not dataset._in_spatial_domain


    if(not is_dataset_in_fourier):
        dataset.to_fourier_domain()

    backproj_im = torch.zeros((L*upsampling_factor,)*3,device=device,dtype=dataset.dtype)
    backproj_ctf = torch.zeros((L*upsampling_factor,)*3,device=device,dtype=dataset.dtype)
    phase_shift_grid = get_phase_shift_grid(L, dtype=backproj_im.real.dtype,device=device)


    for batch in tqdm(dataloader,desc='Reconstructing mean volume'):
        images,pts_rot,filter_indices,idx = batch
        image_offsets = dataset.offsets[idx].to(device).to(pts_rot.dtype)
        images = images.to(device) * offset_to_phase_shift(-image_offsets, phase_shift_grid=phase_shift_grid)
        pts_rot = pts_rot.to(device)
        filters = dataset.unique_filters[filter_indices].to(device) if dataset.unique_filters is not None else None

        nufft_plan.setpts(pts_rot)

        backproj_im += im_backward(images,nufft_plan,filters,fourier_domain=True)[0]
        backproj_ctf += im_backward(torch.complex(filters,torch.zeros_like(filters)),nufft_plan,filters,fourier_domain=True)[0]

    backproj_ctf /= L #normalization by L is needed because backproj_ctf represnts diag(\sum P^T P) and the projection operator and since we only use P^T we are not taking into account a division by L in vol_forward

    if(not is_dataset_in_fourier):
        dataset.to_spatial_domain()


    if init_vol is not None:
        init_vol_rpsd = rpsd(init_vol.squeeze(0))
        reg = dataset.noise_var / upsample_and_expand_fourier_shell(init_vol_rpsd,L * upsampling_factor,3)
        reg /= L
        from matplotlib import pyplot as plt
        v = average_fourier_shell(reg,backproj_ctf)
        fig = plt.figure()
        plt.plot(v.cpu().T)
        plt.yscale('log')
        fig.savefig('test.jpg')
    else:
        reg = torch.ones_like(backproj_ctf) * 1e-1#3 #TODO: determine this constant adapatively

    mean_volume =  backproj_im / (backproj_ctf + reg)

    mean_volume = centered_ifft3(mean_volume,cropping_size=(L,)*3).real

    if(mask is not None):
        mean_volume *= mask.squeeze(0)

    if(not return_lhs_rhs):
        return mean_volume
    else:
        return mean_volume,backproj_im,backproj_ctf


def reconstruct_mean_from_halfsets(dataset : CovarDataset,idx = None, **reconstruction_kwargs):

    reconstruction_kwargs['return_lhs_rhs'] = True
    if(idx is None):
        idx = torch.arange(len(dataset))

    mean_half1,lhs1,rhs1 = reconstruct_mean(dataset,idx=idx[:len(idx)//2],**reconstruction_kwargs)
    mean_half2,lhs2,rhs2 = reconstruct_mean(dataset,idx=idx[len(idx)//2:],**reconstruction_kwargs)

    return regularize_mean_from_halfsets(mean_half1,lhs1,rhs1,
                                        mean_half2,lhs2,rhs2,
                                        mask=reconstruction_kwargs.get('mask',None))

def reconstruct_mean_from_halfsets_DDP(dataset : DataLoader,ranks = None, **reconstruction_kwargs):
    #This function assumes the input dataloader has a distributed sampler. Each node will only pass on its corresponding samples determined by the sampler.
    if(ranks is None):
        ranks = [i for i in range(dist.get_world_size())]

    if(len(ranks) == 1):
        #In the case there's only one rank, we call the non DDP version using the internal dataset of the dataloader, and idx selected by the sampler
        return reconstruct_mean_from_halfsets(dataset.dataset,idx=list(iter(dataset.sampler)),**reconstruction_kwargs)
    reconstruction_kwargs['return_lhs_rhs'] = True


    world_size = len(ranks)
    rank = dist.get_rank()

    result = reconstruct_mean(dataset, **reconstruction_kwargs)
    mean, backproj_im, backproj_ctf = result



    #Sum backproj_im and backproj_ctf only on the group of the corresponding half set
    group1 = dist.new_group(ranks=ranks[:world_size//2])
    group2 = dist.new_group(ranks=ranks[world_size//2:])
    rank_group = group1 if rank in ranks[:world_size//2] else group2
    #print(f'DEVICE : {backproj_im.device} , {torch.norm(backproj_im)}')
    dist.all_reduce(backproj_im,op=dist.ReduceOp.SUM,group=rank_group)
    #print(f'DEVICE : {backproj_im.device} , {torch.norm(backproj_im)}')
    dist.all_reduce(backproj_ctf,op=dist.ReduceOp.SUM,group=rank_group)

    mean_volume =  backproj_im / (backproj_ctf + 1e-1)
    mean_volume = centered_ifft3(mean_volume,cropping_size=(mean.shape[-1],)*3).real

    #Sync mean_volume,backproj_im,backproj_ctf across the two groups
    half1 = []
    half2 = []
    for i,tensor in enumerate([mean_volume,backproj_im,backproj_ctf]):
        #TODO: this is inefficent since tensor is the same across each group. This can be done instead by sending an receiveing the tensor for rank pairs from the two groups
        tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        tensor = tensor.contiguous()
        dist.all_gather(tensor_list, tensor)
        
        half1.append(tensor_list[ranks[0]])
        half2.append(tensor_list[ranks[-1]])

    dist.destroy_process_group(group1)
    dist.destroy_process_group(group2)

    #print(f'DEVICE : {backproj_im.device} , {torch.norm(half1[1])}, {torch.norm(half2[1])}')
    return regularize_mean_from_halfsets(*half1,*half2,reconstruction_kwargs.get('mask',None))
        

def regularize_mean_from_halfsets(mean_half1,lhs1,rhs1,mean_half2,lhs2,rhs2,mask=None):

    L = mean_half1.shape[-1]

    filter_gain = centered_fft3(centered_ifft3((rhs1 + rhs2),cropping_size=(L,)*3)).abs() / 2

    averaged_filter_gain = average_fourier_shell(1 / filter_gain).to(mean_half1.device)

    mean_fsc = vol_fsc(mean_half1,mean_half2)
    fsc_epsilon = 1e-6
    mean_fsc[mean_fsc < fsc_epsilon] = fsc_epsilon
    mean_fsc[mean_fsc > 1-fsc_epsilon] = 1-fsc_epsilon

    fourier_reg = 1/((mean_fsc / (1-mean_fsc)) * averaged_filter_gain)

    fourier_reg = upsample_and_expand_fourier_shell(fourier_reg.unsqueeze(0),lhs1.shape[-1],3) / (L ** 1) #TODO: check normalization constant

    mean_volume = ((lhs1 / (rhs1 + fourier_reg)) + (lhs2 / (rhs2 + fourier_reg)))/2

    mean_volume = centered_ifft3(mean_volume,cropping_size=(L,)*3).real

    if(mask is not None):
        mean_volume *= mask.squeeze(0)



    return mean_volume


if __name__ == "__main__":
    import pickle
    from aspire.volume import Volume
    from aspire.utils import Rotation
    import os
    #dataset_path = 'data/pose_opt_exp'
    dataset_path = 'data/scratch_data/igg_1d/images/snr0.01/downsample_L128/abinit_refine'
    dataset = pickle.load(open(os.path.join(dataset_path,'result_data/dataset.pkl'),'rb'))
    dataset.to_fourier_domain()

    USE_GT = True

    if(USE_GT):
        gt_data = pickle.load(open(os.path.join(dataset_path,'result_data/gt_data.pkl'),'rb'))
        vol = gt_data.mean.unsqueeze(0).to('cuda:0')
        dataset.pts_rot = dataset.compute_pts_rot(torch.tensor(Rotation(gt_data.rotations.numpy()).as_rotvec()).to(torch.float32))
    else:
        vol = torch.tensor(Volume.load('data/pose_opt_exp/relion_noisy_pose.mrc').asnumpy()).to('cuda:0')

    mask = torch.tensor(Volume.load('data/scratch_data/igg_1d/init_mask/mask.mrc').downsample(vol.shape[-1]).asnumpy()).to('cuda:0').squeeze(0)
    rec_vol = reconstruct_mean_from_halfsets(dataset,batch_size=2048,mask=mask)

    from fsc_utils import vol_fsc,rpsd
    fsc = vol_fsc(vol.squeeze(0),rec_vol.squeeze(0))
    mean_fsc = fsc[:vol.shape[-1]//2].mean()
    mse_error = (torch.norm(vol - rec_vol).cpu().numpy()/torch.norm(vol).cpu().numpy())

    print(f'Mean FSC : {mean_fsc}, MSE : {mse_error}')

    print(f'GT NORM : {torch.norm(vol)} ,EST NORM : {torch.norm(rec_vol)}')

    vols_rpsd = rpsd(*torch.concat((vol,rec_vol.unsqueeze(0)),dim=0)).cpu()

    from matplotlib import pyplot as plt
    fig,axs = plt.subplots(2,2)

    axs[0,0].plot(fsc.cpu())
    axs[0,1].plot(vols_rpsd.T.cpu()[:32])
    axs[0,1].set_yscale('log')
    axs[1,0].plot(vols_rpsd[0,:32]/vols_rpsd[1,:32])
    axs[1,0].set_yscale('log')

    fig.savefig('reconstruct.jpg')

    Volume(vol.cpu().numpy()).save('exp_data/mean1.mrc',overwrite=True)
    Volume(rec_vol.cpu().numpy()).save('exp_data/mean2.mrc',overwrite=True)




    
    
    