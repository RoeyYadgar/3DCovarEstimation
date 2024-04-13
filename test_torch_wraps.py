import unittest
import numpy as np
import torch
from torch.autograd import gradcheck
from aspire.volume import Volume,LegacyVolume,rotated_grids
from aspire.utils import Rotation
from aspire.source import Simulation
from aspire.operators import RadialCTFFilter
from aspire.utils import Rotation
from aspire.nufft import nufft as aspire_nufft
from aspire.nufft import anufft as aspire_anufft

import nufft_plan
import projection_funcs

class TestTorchWraps(unittest.TestCase):
    
    def setUp(self):
        self.img_size = 15
        self.num_imgs = 100
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
        
    def test_nufft_forward(self):
        num_ims = 5
        pts_rot = self.pts_rot[:,:self.img_size ** 2]

        #singleton validation
        nufft_forward_aspire = aspire_nufft(self.vols[0],pts_rot)

        vol_torch = torch.tensor(self.vols[0].asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        nufft_forward_torch = nufft_plan.nufft_forward(vol_torch,plan)
        nufft_forward_torch = nufft_forward_torch.cpu().numpy()

        np.testing.assert_allclose(nufft_forward_torch,nufft_forward_aspire,rtol = 1e-6)

        #stack validation
        nufft_forward_aspire = aspire_nufft(self.vols,pts_rot)

        vol_torch = torch.tensor(self.vols.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,self.vols.shape[0])
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        nufft_forward_torch = nufft_plan.nufft_forward(vol_torch,plan)
        nufft_forward_torch = nufft_forward_torch.cpu().numpy()

        np.testing.assert_allclose(nufft_forward_torch,nufft_forward_aspire,rtol = 1e-6)

    def test_nufft_adjoint(self):
        #TODO : figure out why the difference between aspire's and the torch binding has rtol > 1e-4
        num_ims = 5

        #singleton validation
        pts_rot = self.pts_rot[:,:self.img_size ** 2]
        images = self.sim.images[0]
        nufft_adjoint_aspire = aspire_anufft(images.asnumpy().reshape((1,-1)),pts_rot,(self.img_size,)*3)

        im_torch = torch.tensor(images.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        nufft_adjoint_torch = nufft_plan.nufft_adjoint(im_torch,plan)
        nufft_adjoint_torch = nufft_adjoint_torch.cpu().numpy()

        np.testing.assert_allclose(nufft_adjoint_torch,nufft_adjoint_aspire,rtol = 1e-3,atol=1e-3)

        #singleton validation
        num_ims = 5
        pts_rot = self.pts_rot[:,:self.img_size ** 2]
        images = self.sim.images[:num_ims]
        nufft_adjoint_aspire = aspire_anufft(images.asnumpy().reshape((num_ims,-1)),pts_rot,(self.img_size,)*3)

        im_torch = torch.tensor(images.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,num_ims)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        nufft_adjoint_torch = nufft_plan.nufft_adjoint(im_torch,plan)
        nufft_adjoint_torch = nufft_adjoint_torch.cpu().numpy()

        np.testing.assert_allclose(nufft_adjoint_torch,nufft_adjoint_aspire,rtol = 1e-3,atol=1e-3)

    def test_grad_forward(self):
        pts_rot = np.float64(self.pts_rot[:,:self.img_size ** 2])
        vol = torch.randn((self.img_size,)*3,dtype = torch.double, device = self.device)
        vol.requires_grad = True
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,dtype=torch.float64)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        gradcheck(nufft_plan.nufft_forward, (vol,plan), eps=1e-6, rtol=1e-4,nondet_tol= 1e-5)

    def test_grad_adjoint(self):
        pts_rot = np.float64(self.pts_rot[:,:self.img_size ** 2])
        im = torch.randn((self.img_size,)*2,dtype = torch.double, device = self.device)
        im.requires_grad = True
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1,dtype=torch.float64)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        gradcheck(nufft_plan.nufft_adjoint, (im,plan), eps=1e-6, rtol=1e-4,nondet_tol= 1e-5)


    def test_vol_project(self):
        pts_rot = self.pts_rot[:,:self.img_size ** 2]

        vol_forward_aspire = self.sim.vol_forward(self.vols[0],0,1)

        pts_rot = self.pts_rot[:,:self.img_size ** 2]
        vol_torch = torch.tensor(self.vols[0].asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        vol_forward_torch = projection_funcs.vol_forward(vol_torch,plan)
        vol_forward_torch = vol_forward_torch.cpu().numpy()

        np.testing.assert_allclose(vol_forward_torch,vol_forward_aspire,rtol = 1e-3,atol=1e-3)

    def test_im_backproject(self):
  
        pts_rot = self.pts_rot[:,:self.img_size ** 2]
        imgs = self.sim.images[0]
        im_backproject_aspire = self.sim.im_backward(imgs,0).T

        im_torch = torch.tensor(imgs.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1)
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
        filter = torch.tensor(sim.unique_filters[0].evaluate_grid(self.img_size)).to(self.device)
        vol_forward_aspire = sim.vol_forward(self.vols[0],0,1)

        vol_torch = torch.tensor(self.vols[0].asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,)*3,1)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        vol_forward_torch = projection_funcs.vol_forward(vol_torch,plan,filter)
        vol_forward_torch = vol_forward_torch.cpu().numpy()

        np.testing.assert_allclose(vol_forward_torch,vol_forward_aspire,rtol = 1e-3,atol=1e-3)



if __name__ == "__main__":
    
    unittest.main()


