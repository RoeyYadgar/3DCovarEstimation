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

import nufft_plan

class TestNufftPlan(unittest.TestCase):
    
    def setUp(self):
        self.img_size = 15
        self.num_imgs = 100
        c = 5
        self.vols = LegacyVolume(
            L=self.img_size,
            C=c,
            dtype=np.float32,
        ).generate()

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
        pts_rot = self.pts_rot[:,:num_ims*self.img_size ** 2]

        #singleton validation
        nufft_forward_aspire = aspire_nufft(self.vols[0],pts_rot)

        vol_torch = torch.tensor(self.vols[0].asnumpy(),dtype = torch.complex64).to(self.device)
        plan = nufft_plan.Nufft((self.img_size,)*3,1)
        plan.setpts(torch.tensor(pts_rot.copy(),device=self.device))
        nufft_forward_torch = nufft_plan.nufft_forward(vol_torch,plan)
        print(nufft_forward_aspire.shape)

        nufft_forward_torch = nufft_forward_torch.cpu().numpy()

        np.testing.assert_allclose(nufft_forward_torch,nufft_forward_aspire,rtol = 1e-5)

        

if __name__ == "__main__":
    
    unittest.main()


