import torch
from cov3d.projection_funcs import im_backward,centered_ifft3
from cov3d.dataset import CovarDataset
from cov3d.nufft_plan import NufftPlanDiscretized
from cov3d.fsc_utils import rpsd,upsample_and_expand_fourier_shell,average_fourier_shell,vol_fsc
from cov3d.utils import get_torch_device
from tqdm import tqdm

def reconstruct_mean(dataset : CovarDataset,init_vol = None,mask=None,upsampling_factor=2,batch_size=1024,start_ind = None,end_ind = None,return_lhs_rhs=False):

    if(start_ind is None):
        start_ind = 0
    if(end_ind is None):
        end_ind = len(dataset)

    L = dataset.resolution
    nufft_plan = NufftPlanDiscretized((L,)*3,upsample_factor=upsampling_factor,mode='nearest',use_half_grid=False)

    is_dataset_in_fourier = not dataset._in_spatial_domain

    device = get_torch_device() if init_vol is None else init_vol.device

    if(not is_dataset_in_fourier):
        dataset.to_fourier_domain()

    backproj_im = torch.zeros((L*upsampling_factor,)*3,device=device,dtype=dataset.dtype)
    
    backproj_ctf = torch.zeros((L*upsampling_factor,)*3,device=device,dtype=dataset.dtype)

    for i in tqdm(range(start_ind,end_ind,batch_size),desc='Reconstructing mean volume'):
        images,pts_rot,filter_indices,_ = dataset[i:min(i + batch_size,end_ind)]
        images = images.to(device)
        pts_rot = pts_rot.to(device)
        filters = dataset.unique_filters[filter_indices].to(device) if dataset.unique_filters is not None else None

        nufft_plan.setpts(pts_rot.transpose(0,1).reshape((3,-1)))

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


def reconstruct_mean_from_halfsets(dataset : CovarDataset, **reconstruction_kwargs):

    reconstruction_kwargs['return_lhs_rhs'] = True
    mean_half1,lhs1,rhs1 = reconstruct_mean(dataset,start_ind=0,end_ind=len(dataset)//2,**reconstruction_kwargs)
    mean_half2,lhs2,rhs2 = reconstruct_mean(dataset,start_ind=len(dataset)//2,end_ind=len(dataset),**reconstruction_kwargs)

    #filter_gain = dataset.get_total_gain() / dataset.resolution ** 2
    from projection_funcs import centered_fft3
    filter_gain = centered_fft3(centered_ifft3((rhs1 + rhs2),cropping_size=(dataset.resolution,)*3)).abs()

    averaged_filter_gain = 1 / average_fourier_shell(filter_gain).to(mean_half1.device)

    mean_fsc = vol_fsc(mean_half1,mean_half2)
    fsc_epsilon = 1e-6
    mean_fsc[mean_fsc < fsc_epsilon] = fsc_epsilon
    mean_fsc[mean_fsc > 1-fsc_epsilon] = 1-fsc_epsilon

    fourier_reg = 1/((mean_fsc / (1-mean_fsc)) * averaged_filter_gain)

    fourier_reg = upsample_and_expand_fourier_shell(fourier_reg.unsqueeze(0),lhs1.shape[-1],3)

    mean_volume = ((lhs1 / (rhs1 + fourier_reg)) + (lhs2 / (rhs2 + fourier_reg)))/2

    mean_volume = centered_ifft3(mean_volume,cropping_size=(dataset.resolution,)*3).real

    mask = reconstruction_kwargs.get('mask',None)
    if(mask is not None):
        mean_volume *= mask.squeeze(0)

    from matplotlib import pyplot as plt
    fig = plt.figure()
    v = average_fourier_shell(rhs1.real,rhs2.real,fourier_reg)
    plt.plot(v.cpu().T)
    #v = average_fourier_shell(fourier_reg)
    #plt.plot(torch.arange(len(v))*2,v.cpu().T)
    plt.yscale('log')
    fig.savefig('test2.jpg')
    fig = plt.figure()
    plt.plot(mean_fsc.cpu())
    fig.savefig('fsc.jpg')



    return mean_volume


if __name__ == "__main__":
    import pickle
    from aspire.volume import Volume
    from aspire.utils import Rotation
    dataset = pickle.load(open('data/pose_opt_exp/result_data/dataset.pkl','rb'))

    USE_GT = True

    if(USE_GT):
        gt_data = pickle.load(open('data/pose_opt_exp/result_data/gt_data.pkl','rb'))
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




    
    
    