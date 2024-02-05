import unittest
import scipy
from aspire.volume import Volume
from aspire.utils import Rotation
from aspire.source import Simulation

from covar_estimation import *
from utils import *


class TestCovarGradient(unittest.TestCase):
    
    def setUp(self):
        self.voxel = Volume.from_vec((generateCylinderVoxel([-0.5,0],0.5,15)))
        
    def testProjectionBackprojection(self):
        '''
        When projecting and backprojecting, the backprojected volume should be tranposed
        chimeraX views the volume with from 'up view'
        to use voxelSurf from matlab on the volume, the volume dimensions should be permuted with permute(x,[2,3,1])
        '''
        
        
        rot = Rotation.from_rotvec([np.pi/20,np.pi/4,-np.pi],dtype=float)
        sim = Simulation(n = 1 , vols = self.voxel,amplitudes= 1,offsets = 0,angles = rot.as_rotvec())
        
        backprojected_voxel= sim.im_backward(sim.images[:],0).T
        
        self.voxel.save('data/test/voxel.mrc',overwrite= True)
        backprojected_voxel.save('data/test/backprojected_voxel.mrc',overwrite= True)
        sim.images[0].show()
        
        scipy.io.savemat('data/test/voxel.mat',{'voxel':np.array(self.voxel)[0]})
    






if __name__ == "__main__":
    
    unittest.main()


