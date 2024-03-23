import unittest
import numpy as np
import torch
from torch.autograd import gradcheck
from aspire.volume import Volume,LegacyVolume,rotated_grids
from aspire.source import Simulation


import nufft_plan
from covar_estimation import CovarCost
from covar_sgd import Covar
from utils import volsCovarEigenvec

class TestTorchImpl(unittest.TestCase):
    
    def setUp(self):
        self.img_size = 15
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
            offsets = 0
        )

        rots = self.sim.rotations[:]
        pts_rot = rotated_grids(self.img_size,rots)
        self.pts_rot = pts_rot.reshape((3,-1))

        
        
    def test_cost_function(self):
        num_ims = 50
        rank = 4
        vectors = torch.tensor(volsCovarEigenvec(self.vols).asnumpy(),dtype=torch.float32,requires_grad = True)
        #vectors = torch.randn((rank,self.img_size,self.img_size,self.img_size),dtype = torch.float32,requires_grad = True)
        images = self.sim.images[:num_ims]
        old_impl_cost = CovarCost.apply(vectors,self.sim,0,images)

        device = torch.device("cuda:0")
        vectors_new = torch.tensor(vectors.detach(),device=device)

        plans = []
        pts_rot = torch.tensor(self.pts_rot.copy()).reshape((3,self.num_imgs,self.img_size**2))
        for i in range(num_ims):
            plan = nufft_plan.NufftPlan((self.img_size,)*3,batch_size = rank,dtype = torch.float32,eps = 1e-6)
            plan.setpts(pts_rot[:,i].to(device)) 
            plans.append(plan)
        images = torch.tensor(images.asnumpy()).to(device)

        covar = Covar(self.img_size,rank,dtype=torch.float32,vectors=vectors_new)
        new_impl_cost = covar.cost(images,plans)

        old_impl_cost.backward()
        new_impl_cost.backward()

        grad_old = vectors.grad
        grad_new = covar.vectors.grad.cpu()

        torch.testing.assert_allclose(grad_old, grad_new, rtol=1e-05,atol=1e-5)


if __name__ == "__main__":
    
    unittest.main()


