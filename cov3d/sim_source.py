import torch
import numpy as np
from aspire.utils import Rotation
from aspire.volume import rotated_grids,LegacyVolume,Volume
from aspire.operators import MultiplicativeFilter,ScalarFilter,ArrayFilter,RadialCTFFilter
from cov3d.utils import get_torch_device
from cov3d.nufft_plan import NufftPlan,NufftPlanDiscretized
from cov3d.projection_funcs import vol_forward
from cov3d.covar_sgd import CovarDataset
from cov3d.workflow import covar_processing
from cov3d.analyze import analyze
from cov3d.utils import volsCovarEigenvec
from cov3d.fsc_utils import rpsd
import os
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

class SimulatedSource():
    def __init__(self,n,vols,noise_var,whiten=True,unique_filters = None):
        self.n = n
        self.L = vols.shape[-1]
        self.num_vols = vols.shape[0]
        self.vols = vols
        self.whiten = whiten
        self.noise_var = noise_var        
        self._unique_filters = unique_filters
        self._clean_images = self._gen_clean_images()

    
    @property
    def noise_var(self):
        return self._noise_var if (not self.whiten) else 1
    
    @noise_var.setter
    def noise_var(self,noise_var):
        self._noise_var = noise_var

    @property
    def images(self):
        images = self._clean_images + torch.randn((self.n,self.L,self.L),dtype=self._clean_images.dtype,device=self._clean_images.device) * (self._noise_var ** 0.5)
        if(self.whiten):
            images /= (self._noise_var) ** 0.5

        return images.numpy()

    @property
    def unique_filters(self):
        whiten_filter = ScalarFilter(dim=2,value=self._noise_var ** (-0.5))
        return [MultiplicativeFilter(filt, whiten_filter) for filt in self._unique_filters]

    def _gen_clean_images(self,batch_size=1024):
        clean_images = torch.zeros((self.n,self.L,self.L))
        self.offsets = torch.zeros((self.n,2))
        self.amplitudes = np.ones((self.n))
        self.states = torch.tensor(np.random.choice(self.num_vols,self.n))
        self.filter_indices = np.random.choice(len(self._unique_filters),self.n)
        self.rotations = Rotation.generate_random_rotations(self.n).matrices

        unique_filters = torch.tensor(np.array([self._unique_filters[i].evaluate_grid(self.L) for i in range(len(self._unique_filters))]))
        pts_rot = torch.tensor(rotated_grids(self.L,self.rotations).copy()).reshape((3,self.n,self.L**2))
        pts_rot = pts_rot.transpose(0,1) 
        pts_rot = (torch.remainder(pts_rot + torch.pi , 2 * torch.pi) - torch.pi)

        device = get_torch_device()
        volumes = torch.tensor(self.vols.asnumpy(),device=device)
        nufft_plan = NufftPlan((self.L,)*3,batch_size = 1, dtype=volumes.dtype,device=device)

        for i in range(self.num_vols):
            idx = (self.states == i).nonzero().reshape(-1)
            for j in range(0,len(idx),batch_size):
                batch_ind = idx[j:j+batch_size]
                ptsrot = pts_rot[batch_ind].to(device)
                filter_indices = self.filter_indices[batch_ind]
                filters = unique_filters[filter_indices].to(device)

                nufft_plan.setpts(ptsrot.transpose(0,1).reshape((3,-1)))
                projected_volume = vol_forward(volumes[i].unsqueeze(0),nufft_plan,filters).squeeze(1)

                clean_images[batch_ind] = projected_volume.cpu()

        return clean_images


def inspect_dataset(dataset,output_path):
    L = dataset.resolution
    sample_images = dataset.images[:5].transpose(0,1).reshape(L,-1)

    is_vectors_gd = dataset.vectorsGD is not None

    fig = plt.figure(figsize=(8,6))
    gs = GridSpec(2,3,figure=fig)
    ax1 = fig.add_subplot(gs[0,:])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[1,1])

    ax1.imshow(sample_images)
    cbar = fig.colorbar(ax1.imshow(sample_images), ax=ax1, orientation='vertical')
    cbar.set_label('Intensity')
    ax1.set_title('Image samples')


    ax2.plot(dataset.signal_rpsd)
    ax2.set_yscale('log')
    ax2.set_title('Signal RPSD estimate')

    ax3.plot(dataset.radial_filters_gain)
    ax3.set_yscale('log')
    ax3.set_title('Filter radial gain')

    if(is_vectors_gd):
        ax4 = fig.add_subplot(gs[1,2])
        ax4.plot(rpsd(*dataset.vectorsGD.reshape(-1,L,L,L)).T)
        ax4.set_yscale('log')
        ax4.set_title('Groundtruth eigen volumes RPSD')

    fig.savefig(output_path)

def simulateExp(folder_name = None,no_ctf=False):
    os.makedirs(folder_name,exist_ok=True)

    L = 64
    n = 100000
    r = 5
    pixel_size = 3 * 128/ L

    if(not no_ctf):
        filters = [RadialCTFFilter(defocus=d,pixel_size=pixel_size) for d in np.linspace(8e3, 2.5e4, 927)]
    else:
        filters = [ArrayFilter(np.ones((L,L)))]


    voxels = LegacyVolume(L=int(L*0.7),C=r+1,K=64,dtype=np.float32,pixel_size=pixel_size).generate()
    padded_voxels = np.zeros((r+1, L, L, L), dtype=np.float32)
    pad_width = (L - voxels.shape[1]) // 2
    padded_voxels[:, pad_width:pad_width+voxels.shape[1], pad_width:pad_width+voxels.shape[2], pad_width:pad_width+voxels.shape[3]] = voxels
    voxels = Volume(padded_voxels)
    voxels.save(os.path.join(folder_name,'gt_vols.mrc'),overwrite=True) 

    sim = SimulatedSource(n,vols=voxels,unique_filters=filters,noise_var = 0)
    var = torch.var(sim._clean_images).item()
    
   
    vectorsGD = volsCovarEigenvec(voxels)    
    snr_vals = 10**np.arange(0,-3.5,-0.5)
    for snr in snr_vals:
        noise_var = var / snr
        print(f'Signal power : {var}. Using noise variance of {noise_var} to achieve SNR of {snr}')

        sim.noise_var = noise_var
        noise_var = sim.noise_var
        dataset = CovarDataset(sim,noise_var,vectorsGD=vectorsGD,mean_volume=Volume(voxels.asnumpy().mean(axis=0)))
        dataset.starfile = 'tmp.star'

        dir_name = os.path.join(folder_name,'obj_ls',f'algorithm_output_{snr}')
        os.makedirs(dir_name,exist_ok=True)  
        inspect_dataset(dataset,os.path.join(dir_name,'dataset.jpg'))
        covar_processing(dataset,r,dir_name,max_epochs=20,lr=1e-2,objective_func='ls',num_reg_update_iters=1)
        analysis_figures = analyze(os.path.join(dir_name,'recorded_data.pkl'),output_dir=dir_name,analyze_with_gt=True,skip_reconstruction=True,gt_labels=sim.states,num_clusters=0)


if __name__=="__main__":
    simulateExp('data/rank5_covar_estimate')