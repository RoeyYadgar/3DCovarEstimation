import unittest
import numpy as np
import torch
from torch.autograd import gradcheck
from aspire.volume import Volume,LegacyVolume,rotated_grids
from aspire.source import Simulation


import nufft_plan
#from covar_estimation import CovarCost
from covar_sgd import Covar,cost,CovarDataset
from utils import volsCovarEigenvec,generateBallVoxel
from projection_funcs import vol_forward
from aspire.operators import RadialCTFFilter,ArrayFilter
from aspire.utils import Rotation



class TestTorchImpl(unittest.TestCase):
    
    def setUp(self):
        self.img_size = 32
        self.num_imgs = 100
        c = 5
        self.vols = LegacyVolume(
            L=self.img_size,
            C=c,
            dtype=np.float32,
        ).generate()
        self.vols -= np.mean(self.vols,axis=0)

        self.sim = Simulation(
            n=self.num_imgs,
            vols=self.vols,
            dtype=np.float32,
            amplitudes=1,
            offsets = 0,
            unique_filters=[RadialCTFFilter(defocus=d,pixel_size=3) for d in np.linspace(8e3, 2.5e4, 7)]
        )
        self.dataset = CovarDataset(self.sim,noise_var=0)
        rots = self.sim.rotations[:]
        pts_rot = rotated_grids(self.img_size,rots)
        self.pts_rot = pts_rot.reshape((3,-1))

        self.device = torch.device('cuda:0')
    
    def test_cost_gradient_scaling(self):

        #Test scaling of volumes 
        num_ims = 50
        rank = 4
        reg = 1e-5
        noise_var = 1
        scaling_param = 10


        vols = torch.randn((rank,self.img_size,self.img_size,self.img_size),dtype = torch.float32,requires_grad = True,device=self.device)
        images,pts_rot,filter_inds = self.dataset[:num_ims]
        pts_rot = pts_rot.to(self.device)
        images = images.to(self.device)
        filters = self.dataset.unique_filters[filter_inds].to(self.device)
        plans = []
        for i in range(num_ims):
            plan = nufft_plan.NufftPlan((self.img_size,)*3,batch_size = rank,dtype = torch.float32,device=self.device)
            plan.setpts(pts_rot[i]) 
            plans.append(plan)

        cost_val = cost(vols,images,plans,filters,noise_var,reg = reg)
        cost_val.backward()

        vols_grad = vols.grad.clone()

        scaled_vols = torch.tensor(vols.data * scaling_param,requires_grad = True,device = self.device)
        #When the volumes are scaled by alpha (and SNR is preserved) the cost gradient should scale by alpha ** 3
        cost_val = cost(scaled_vols,images*scaling_param,plans,filters,noise_var * (scaling_param ** 2),reg=reg)
        cost_val.backward()

        scaled_vols_grad = scaled_vols.grad.clone()
        torch.testing.assert_close((scaled_vols_grad).to('cpu'), (scaling_param**3) * vols_grad.to('cpu'), rtol=5e-3,atol=5e-3)

        vols.grad.zero_()
        #When the filters are scaled by alpha (and SNR is preseverd) the cost gradient should scale by alpha ** 4 as well as the regularization parameter.
        cost_val = cost(vols,images*scaling_param,plans,filters*scaling_param,noise_var * (scaling_param ** 2),reg=reg*(scaling_param**4))
        cost_val.backward()
        scaled_filters_grid = vols.grad.clone()
        torch.testing.assert_close((scaled_filters_grid).to('cpu'), (scaling_param**4) * vols_grad.to('cpu'), rtol=5e-3,atol=5e-3)

    def test_cost_gradient_resolution_scaling(self):
        #TODO : check if When the resolution is scaled by alpha is the gradient scaled by alpha ** 3?
        num_ims = 20
        rank = 4
        reg = 0
        noise_var = 0

        rots = Rotation.generate_random_rotations(num_ims).matrices

        pts_rot = torch.tensor(rotated_grids(self.img_size,rots).copy(),device=self.device,dtype=torch.float32).reshape((3,num_ims,self.img_size**2))
        pts_rot = pts_rot.transpose(0,1)

        vols = torch.randn((rank,self.img_size,self.img_size,self.img_size),dtype = torch.float32,requires_grad = True,device=self.device)
        plans = []
        for i in range(num_ims):
            plan = nufft_plan.NufftPlan((self.img_size,)*3,batch_size = rank,dtype = torch.float32,device=self.device)
            plan.setpts(pts_rot[i]) 
            plans.append(plan)

        images = torch.zeros(num_ims,self.img_size,self.img_size,device = self.device,dtype=torch.float32)
        for i in range(num_ims):
            images[i] = vol_forward(vols,plans)[i,i%rank]

        cost_val = cost(vols,images,plans,None,noise_var,reg = reg)
        cost_val.backward()
        vols_grad = vols.grad.clone()

        #Downsampling    
        img_size_ds = int(self.img_size/2)
        vols_ds = Volume(vols.detach().cpu().numpy()).downsample(img_size_ds)
        vols_ds = torch.tensor(vols_ds.asnumpy(),device=self.device,dtype=torch.float32,requires_grad = True)
        pts_rot = torch.tensor(rotated_grids(img_size_ds,rots).copy(),device=self.device,dtype=torch.float32).reshape((3,num_ims,img_size_ds**2))
        pts_rot = pts_rot.transpose(0,1)
        
        plans = []
        for i in range(num_ims):
            plan = nufft_plan.NufftPlan((img_size_ds,)*3,batch_size = rank,dtype = torch.float32,device=self.device)
            plan.setpts(pts_rot[i]) 
            plans.append(plan)

        images_ds = torch.zeros(num_ims,img_size_ds,img_size_ds,device = self.device,dtype=torch.float32)
        for i in range(num_ims):
            images_ds[i] = vol_forward(vols_ds,plans)[i,i%rank]

        cost_val = cost(vols_ds,images_ds,plans,None,noise_var,reg = reg)
        cost_val.backward()
        vols_grad_ds = vols_ds.grad.clone()
        
        print(torch.norm(vols_grad))
        print(torch.norm(vols_grad_ds))
        print(torch.norm(vols_grad)/torch.norm(vols_grad_ds))

    def test_projection_resolution_scaling(self):
        n = 8
        #Projection scales values of volume by 1/(L^*1.5)
        
        L1 = 15
        voxels = Volume.from_vec((generateBallVoxel([-0.6,0,0],0.5,L1)))
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
        rots = sim.rotations
        
        norm_vol1 = (np.linalg.norm(voxels))
        norm_images1 = np.linalg.norm(sim.images[:],axis=(1,2))
        norm_backproj1 = np.linalg.norm(sim.images[0].backproject(rots[0].reshape((1,3,3))))                                
        
        L2 = 200
        voxels = Volume.from_vec((generateBallVoxel([-0.6,0,0],0.5,L2)))
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,angles=Rotation.from_matrix(rots).as_rotvec())
        
        
        
        norm_vol2 = (np.linalg.norm(voxels))
        norm_images2 = np.linalg.norm(sim.images[:],axis=(1,2))
        norm_backproj2 = np.linalg.norm(sim.images[0].backproject(rots[0].reshape((1,3,3))))
        
        print(norm_vol2/norm_vol1)
        print(norm_images2/norm_images1)
        print(norm_backproj2/norm_backproj1)

if __name__ == "__main__":
    
    unittest.main()


