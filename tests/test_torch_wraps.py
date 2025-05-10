#%%
import unittest
import numpy as np
import torch
from torch.autograd import gradcheck
from aspire.volume import Volume,LegacyVolume,rotated_grids
from aspire.utils import Rotation
from aspire.source import Simulation
from aspire.operators import RadialCTFFilter
from aspire.image import Image
from aspire.nufft import nufft as aspire_nufft
from aspire.nufft import anufft as aspire_anufft

from cov3d import nufft_plan
from cov3d import projection_funcs
from cov3d.poses import PoseModule
from cov3d.covar import Mean

class TestTorchWraps(unittest.TestCase):
    
    def setUp(self):
        self.img_size = 15
        self.num_imgs = 2048
        c = 5
        self.vols = LegacyVolume(
            L=self.img_size,
            C=c,
            dtype=np.float32,
        ).generate() * 100

        self.sim = Simulation(
            n=self.num_imgs,
            vols=self.vols,
            dtype=np.float32,
            amplitudes=1,
            offsets = 0
        )

        rots = self.sim.rotations[:]
        pts_rot = rotated_grids(self.img_size,rots)
        self.pts_rot = pts_rot.reshape((3,-1))

        self.device = torch.device("cuda:0")
        
    def test_centered_fft_pad_crop(self):
        us = 2
        vol_torch = torch.tensor(self.vols.asnumpy(),device=self.device)
        vol_torch_fft_padded = projection_funcs.centered_fft3(vol_torch,padding_size=(self.img_size*us,)*3)
        vol_torch_ifft_cropped = projection_funcs.centered_ifft3(vol_torch_fft_padded,cropping_size=(self.img_size,)*3).real

        torch.testing.assert_close(vol_torch_ifft_cropped,vol_torch,rtol=1e-5,atol=1e-4)


    def test_nufft_forward(self):
        vols = self.vols      
        pts_rot = self.pts_rot[:,:self.img_size ** 2]

        #singleton validation
        nufft_forward_aspire = aspire_nufft(vols[0].asnumpy(),pts_rot).reshape(1,-1)

        vol_torch = torch.tensor(vols[0].asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,device = self.device)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        nufft_forward_torch = nufft_plan.nufft_forward(vol_torch,plan)
        nufft_forward_torch = nufft_forward_torch.cpu().numpy()

        pts_rot_torch = (torch.remainder(torch.tensor(pts_rot.copy(),device=self.device) + torch.pi , 2 * torch.pi) - torch.pi)
        us = 2
        plan = nufft_plan.NufftPlanDiscretized((self.img_size,)*3,upsample_factor=us,mode='bilinear')
        plan.setpts(pts_rot_torch)
        nufft_forward_disc = plan.execute_forward(projection_funcs.centered_fft3(vol_torch,padding_size=(self.img_size*us,)*3))
        nufft_forward_disc = nufft_forward_disc.reshape(1,-1).cpu().numpy()


        threshold = np.mean(np.abs(nufft_forward_aspire))        
        np.testing.assert_allclose(nufft_forward_torch,nufft_forward_aspire,rtol = 1e-3,atol=threshold*0.01)
        print(np.linalg.norm(nufft_forward_disc - nufft_forward_torch)/np.linalg.norm(nufft_forward_torch))
        np.testing.assert_array_less(np.linalg.norm(nufft_forward_disc - nufft_forward_torch)/np.linalg.norm(nufft_forward_torch),0.2)

        #stack validation
        nufft_forward_aspire = aspire_nufft(vols,pts_rot)

        vol_torch = torch.tensor(vols.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,vols.shape[0],device = self.device)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        nufft_forward_torch = nufft_plan.nufft_forward(vol_torch,plan)
        nufft_forward_torch = nufft_forward_torch.cpu().numpy()

        pts_rot_torch = (torch.remainder(torch.tensor(pts_rot.copy(),device=self.device) + torch.pi , 2 * torch.pi) - torch.pi)
        plan = nufft_plan.NufftPlanDiscretized((self.img_size,)*3,upsample_factor=us)
        plan.setpts(pts_rot_torch)
        nufft_forward_disc = plan.execute_forward(projection_funcs.centered_fft3(vol_torch,padding_size=(self.img_size*us,)*3))
        nufft_forward_disc = nufft_forward_disc.reshape(vols.shape[0],-1).cpu().numpy()

        np.testing.assert_allclose(nufft_forward_torch,nufft_forward_aspire,rtol = 1e-3,atol=threshold*0.01)
        print(np.linalg.norm(nufft_forward_disc - nufft_forward_torch)/np.linalg.norm(nufft_forward_torch))
        np.testing.assert_array_less(np.linalg.norm(nufft_forward_disc - nufft_forward_torch)/np.linalg.norm(nufft_forward_torch),0.2)
        
    def test_nufft_adjoint(self):
        #TODO : figure out why the difference between aspire's and the torch binding has rtol > 1e-4
        num_ims = 5

        #singleton validation
        pts_rot = self.pts_rot[:,:self.img_size ** 2]
        images = self.sim.images[0]
        from aspire.numeric import fft, xp
        from aspire.image import Image
        images = Image(xp.asnumpy(fft.centered_fft2(xp.asarray(images))))
        nufft_adjoint_aspire = aspire_anufft(images.asnumpy().reshape((1,-1)),pts_rot,(self.img_size,)*3)
        threshold = np.mean(np.abs(nufft_adjoint_aspire.real))*0.1


        im_torch = torch.tensor(images.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,device = self.device)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        nufft_adjoint_torch = nufft_plan.nufft_adjoint(im_torch,plan)
        nufft_adjoint_torch = nufft_adjoint_torch.cpu().numpy()[0]

        np.testing.assert_allclose(nufft_adjoint_torch,nufft_adjoint_aspire,rtol = 1e-2,atol=threshold)

        # Testing with NufftPlanDiscretized
        us = 4
        pts_rot_torch = (torch.remainder(torch.tensor(pts_rot.copy(), device=self.device) + torch.pi, 2 * torch.pi) - torch.pi)
        plan_disc = nufft_plan.NufftPlanDiscretized((self.img_size,) * 3, upsample_factor=us, mode='bilinear')
        plan_disc.setpts(pts_rot_torch)
        nufft_adjoint_disc = plan_disc.execute_adjoint(im_torch)
        nufft_adjoint_disc = projection_funcs.centered_ifft3(nufft_adjoint_disc,cropping_size=(self.img_size,)*3).cpu().numpy()[0]

        m = Mean(torch.tensor(nufft_adjoint_disc),15,upsampling_factor=us)
        m.init_grid_correction('bilinear')

        np.testing.assert_allclose(nufft_adjoint_disc.real * (us * self.img_size) ** 3 /  m.grid_correction.numpy(), nufft_adjoint_aspire.real, rtol=1e-2, atol=threshold * 3)


        # Stack validation
        pts_rot = self.pts_rot[:, :self.img_size ** 2 * num_ims]
        images = self.sim.images[:num_ims]
        images = Image(xp.asnumpy(fft.centered_fft2(xp.asarray(images))))
        threshold = np.mean(np.abs(nufft_adjoint_aspire.real))*0.1
        nufft_adjoint_aspire = aspire_anufft(images.asnumpy().reshape((1, -1)), pts_rot, (self.img_size,) * 3)
        threshold = np.mean(np.abs(nufft_adjoint_aspire.real))*0.1

        im_torch = torch.tensor(images.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, device=self.device)
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device))
        nufft_adjoint_torch = nufft_plan.nufft_adjoint(im_torch.reshape(num_ims,-1), plan)
        nufft_adjoint_torch = nufft_adjoint_torch.cpu().numpy()[0]

        np.testing.assert_allclose(nufft_adjoint_torch, nufft_adjoint_aspire, rtol=1e-2, atol=threshold)

        # Stack validation with NufftPlanDiscretized
        pts_rot_torch = (torch.remainder(torch.tensor(pts_rot.copy(), device=self.device) + torch.pi, 2 * torch.pi) - torch.pi)
        plan_disc = nufft_plan.NufftPlanDiscretized((self.img_size,) * 3, upsample_factor=us, mode='bilinear')
        plan_disc.setpts(pts_rot_torch)
        nufft_adjoint_disc = plan_disc.execute_adjoint(im_torch)
        nufft_adjoint_disc = projection_funcs.centered_ifft3(nufft_adjoint_disc,cropping_size=(self.img_size,)*3).cpu().numpy()[0]

        np.testing.assert_allclose(nufft_adjoint_disc.real * (us * self.img_size) ** 3 /  m.grid_correction.numpy(), nufft_adjoint_aspire.real, rtol=1e-2, atol=threshold * 3)

    def test_grad_forward(self):
        pts_rot = np.float64(self.pts_rot[:,:self.img_size ** 2])
        vol = torch.randn((self.img_size,)*3,dtype = torch.double, device = self.device)
        vol.requires_grad = True
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,dtype=torch.float64,device = self.device)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        gradcheck(nufft_plan.nufft_forward, (vol,plan), eps=1e-6, rtol=1e-4,nondet_tol= 1e-5)

    def test_grad_adjoint(self):
        pts_rot = np.float64(self.pts_rot[:,:self.img_size ** 2])
        im = torch.randn((self.img_size,)*2,dtype = torch.double, device = self.device)
        im.requires_grad = True
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,dtype=torch.float64,device = self.device)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        gradcheck(nufft_plan.nufft_adjoint, (im,plan), eps=1e-6, rtol=1e-4,nondet_tol= 1e-5)


    def test_vol_project(self):
        pts_rot = self.pts_rot[:,:self.img_size ** 2]

        vol_forward_aspire = self.sim.vol_forward(self.vols[0],0,1)

        pts_rot = self.pts_rot[:,:self.img_size ** 2]
        vol_torch = torch.tensor(self.vols[0].asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,device = self.device)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        vol_forward_torch = projection_funcs.vol_forward(vol_torch,plan)
        vol_forward_torch = vol_forward_torch.cpu().numpy()

        np.testing.assert_allclose(vol_forward_torch,vol_forward_aspire,rtol = 1e-3,atol=1e-3)

    def test_im_backproject(self):
  
        pts_rot = self.pts_rot[:,:self.img_size ** 2]
        imgs = self.sim.images[0]
        im_backproject_aspire = self.sim.im_backward(imgs,0).asnumpy()

        im_torch = torch.tensor(imgs.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,device = self.device)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        im_backproject_torch = projection_funcs.im_backward(im_torch,plan)
        im_backproject_torch = im_backproject_torch.cpu().numpy()

        np.testing.assert_allclose(im_backproject_torch,im_backproject_aspire,rtol = 1e-3,atol=1e-3)        


    def test_vol_project_ctf(self):
        sim = Simulation(
            n=1,
            vols=self.vols,
            dtype=np.float32,
            amplitudes=1,
            offsets = 0,
            unique_filters=[RadialCTFFilter(defocus=1.5e4)]
        )
        rots = sim.rotations[:]
        pts_rot = rotated_grids(self.img_size,rots)
        pts_rot = pts_rot.reshape((3,-1))
        pts_rot = pts_rot[:,:self.img_size ** 2]
        filter = torch.tensor(sim.unique_filters[0].evaluate_grid(self.img_size)).unsqueeze(0).to(self.device)
        vol_forward_aspire = sim.vol_forward(self.vols[0],0,1)

        vol_torch = torch.tensor(self.vols[0].asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,device = self.device)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        vol_forward_torch = projection_funcs.vol_forward(vol_torch,plan,filter)
        vol_forward_torch = vol_forward_torch.cpu().numpy()

        np.testing.assert_allclose(vol_forward_torch,vol_forward_aspire,rtol = 1e-3,atol=1e-3)

    def test_vol_project_fourier_slice(self):
        vol = torch.tensor(self.vols[0].asnumpy(),device=self.device)
        rot = np.array([np.eye(3)],self.vols.dtype)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,device=self.device)
        plan.setpts(torch.tensor(rotated_grids(self.img_size,rot).copy(),device=self.device).reshape((3,-1)))
        
        vol_forward = projection_funcs.vol_forward(vol,plan)
        vol_forward_fourier = projection_funcs.centered_fft2(vol_forward)[0]

        vol_fourier = projection_funcs.centered_fft3(vol)
        vol_fourier_slice = vol_fourier[0][self.img_size//2]

        torch.testing.assert_close(vol_fourier_slice, vol_forward_fourier*self.img_size, rtol=5e-3,atol=5e-3)


    def test_batch_nufft_grad(self):
        batch_size = 8
        pts_rot = self.pts_rot[:,:batch_size * self.img_size ** 2].reshape((3,batch_size,-1))
        pts_rot = torch.tensor(pts_rot.copy(),device=self.device).transpose(0,1)
        vol_torch = torch.tensor(self.vols.asnumpy(),device=self.device,requires_grad=True)
        num_vols = vol_torch.shape[0]
        plans = [nufft_plan.NufftPlan((self.img_size,)*3,num_vols,device = self.device) for i in range(batch_size)]
        for i in range(batch_size):
            plans[i].setpts(pts_rot[i])
        vol_forward = torch.zeros((batch_size,num_vols,self.img_size,self.img_size),dtype=vol_torch.dtype,device=self.device)
        for i in range(batch_size):
            vol_forward[i] = projection_funcs.vol_forward(vol_torch,plans[i])

        v1 = torch.norm(vol_forward)
        v1.backward()
        vol_forward_grad = vol_torch.grad

        vol_torch = torch.tensor(self.vols.asnumpy(),device=self.device,requires_grad=True)
        batch_plans = nufft_plan.NufftPlan((self.img_size,)*3,num_vols,device = self.device)
        batch_plans.setpts(pts_rot.transpose(0,1).reshape((3,-1)))
        batch_vol_forward = projection_funcs.vol_forward(vol_torch,batch_plans)

        v2 = torch.norm(batch_vol_forward)
        v2.backward()
        batch_vol_forward_grad = vol_torch.grad


        torch.testing.assert_close(vol_forward,batch_vol_forward,rtol=5e-3,atol=5e-3)
        torch.testing.assert_close(vol_forward_grad,batch_vol_forward_grad,rtol=5e-3,atol=5e-3)


    def test_pose_module_rots(self):
        rotations = self.sim.rotations
        init_rotvec = torch.tensor(Rotation.from_matrix(rotations).as_rotvec(),dtype=torch.float32)
        pose_module = PoseModule(init_rotvec,torch.zeros(len(init_rotvec),2),self.img_size)
        index = torch.tensor([5,13,192,153])
        pts_rot = torch.tensor(self.pts_rot.copy()).reshape(3,-1,self.img_size**2)[:,index].transpose(0,1)
        module_pts_rot,_ = pose_module(index)
        torch.testing.assert_close(pts_rot,module_pts_rot,rtol=1e-3,atol=1e-3)

    def test_pose_module_offsets(self):
        N = 100  
        offsets = torch.randn((N,2),dtype=torch.float32) * 5
        init_rotvec = torch.tensor(Rotation.from_matrix(self.sim.rotations[:N]).as_rotvec(),dtype=torch.float32)
        pose_module = PoseModule(init_rotvec,offsets,self.img_size)

        images = torch.randn((N,self.img_size,self.img_size),dtype=torch.float32)
        _,phase_shift = pose_module(torch.arange(N))
        module_shifted_images = projection_funcs.centered_ifft2(projection_funcs.centered_fft2(images) * phase_shift).real

        aspire_shifted_images = Image(images.numpy()).shift(-offsets).asnumpy()

        torch.testing.assert_close(module_shifted_images,torch.tensor(aspire_shifted_images),rtol=1e-3,atol=1e-3)

if __name__ == "__main__":
    
    unittest.main()


