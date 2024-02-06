import unittest
import scipy
from aspire.volume import Volume
from aspire.utils import Rotation
from aspire.source import Simulation

from covar_estimation import *
from utils import *


class TestCovarGradient(unittest.TestCase):
    
    def setUp(self):
        pass
        
    @unittest.skip("not needed")
    def testProjectionBackprojection(self):
        '''
        When projecting and backprojecting, the backprojected volume should be tranposed
        chimeraX views the volume with from 'up view'
        to use voxelSurf from matlab on the volume, the volume dimensions should be permuted with permute(x,[2,3,1])
        '''
        
        self.voxel = Volume.from_vec((generateCylinderVoxel([-0.5,0],0.5,15)))
        rot = Rotation.from_rotvec([np.pi/20,np.pi/4,-np.pi],dtype=float)
        sim = Simulation(n = 1 , vols = self.voxel,amplitudes= 1,offsets = 0,angles = rot.as_rotvec())
        
        backprojected_voxel= sim.im_backward(sim.images[:],0).T
        
        self.voxel.save('data/test/voxel.mrc',overwrite= True)
        backprojected_voxel.save('data/test/backprojected_voxel.mrc',overwrite= True)
        sim.images[0].show()
        
        scipy.io.savemat('data/test/voxel.mat',{'voxel':np.array(self.voxel)[0]})
    
    
    @unittest.skip("not needed")
    def testCovarGradientComputation(self):
        L = 15
        n = 1024
        voxels = Volume.from_vec(np.concatenate((generateBallVoxel([-0.6,0,0],0.5,L),
                                         -generateBallVoxel([-0.6,0,0],0.5,L))))
        
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
        
        
        vec = voxels.to_vec()[0].reshape((1,L,L,L))
        vec[0,0,0,0] = 1
        vec = torch.tensor(vec,requires_grad=True)
        cost_val = CovarCost.apply(vec,sim,0,sim.images[:])
        cost_val.backward()
        
        Volume(vec.detach().numpy()).save('data/test/grad_input.mrc',overwrite = True)
        Volume(vec.grad.detach().numpy()).save('data/test/grad_output.mrc',overwrite = True)


    def testCovarGradcheck(self):
        #TODO : check why gradient is non-deterministic
        L = 15
        n = 16

        voxels = Volume.from_vec(np.double(np.concatenate((generateBallVoxel([0,0,0],0.5,L), #Use double dtype when using gradcheck
                                         -generateBallVoxel([0,0,0],0.5,L)))))
        
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
        
        from torch.autograd import gradcheck
        input = torch.randn((1,L,L,L), requires_grad=True,dtype=torch.double)
        gradcheck(CovarCost.apply, (input,sim,0,sim.images[:]), eps=1e-6, atol=1e-4,nondet_tol= 1e-5)



if __name__ == "__main__":
    
    unittest.main()


