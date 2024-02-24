import unittest
import scipy
from aspire.volume import Volume
from aspire.utils import Rotation
from aspire.source import Simulation
from aspire.operators import RadialCTFFilter
from torch.autograd import gradcheck,gradgradcheck

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

    @unittest.skip("not needed")
    def testCovarGradcheck(self):
        #TODO : check why gradient is non-deterministic
        #TODO : check regularization gradient numerical error
        L = 15
        n = 8
        r = 2

        voxels = Volume.from_vec(np.double(np.concatenate((generateBallVoxel([0,0,0],0.5,L), #Use double dtype when using gradcheck
                                         -generateBallVoxel([0,0,0],0.5,L)))))
        
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
        
        input = torch.randn((r,L,L,L), requires_grad=True,dtype=torch.double)
        gradcheck(CovarCost.apply, (input,sim,0,sim.images[:],0.01), eps=1e-6, atol=1e-4,nondet_tol= 1e-5)
        
    def testCovarGradcheckCTF(self):
        L = 5
        n = 8
        r = 2

        voxels = Volume.from_vec(np.double(np.concatenate((generateBallVoxel([0,0,0],0.5,L), #Use double dtype when using gradcheck
                                         -generateBallVoxel([0,0,0],0.5,L)))))
        
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])
        
        
        input = torch.randn((r,L,L,L), requires_grad=True,dtype=torch.double)
        #gradcheck(CovarCost.apply, (input,sim,0,sim.images[:],0.01), eps=1e-6, atol=1e-4,nondet_tol= 1e-5)
        #gradgradcheck(CovarCost.apply, (input,sim,0,sim.images[:],0.01), eps=1e-6, atol=1e-4,nondet_tol= 1e-5)
        c = CovarCost.apply(input,sim,0,sim.images[:],0.01)
        grads = torch.autograd.grad(c,input,create_graph=True)[0]
        #hessian = torch.autograd.grad(grads,input,grad_outputs=torch.ones_like(grads))[0]
        hessian = []
        for grad in grads.reshape((-1)):
            hessian_row = torch.autograd.grad(grad,input,create_graph = True)
            hessian.append(hessian_row)
        print(hessian.shape)
        

    @unittest.skip("not needed")  
    def testAmplitudeEffect(self):
        #gradient scales with ampltitude^3 - learning_rate should scale with 1/ampltidue^2
        n = 8 
        L = 15
        amp = 100
        voxels = Volume.from_vec((generateBallVoxel([-0.6,0,0],0.5,L)))
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
        
        v1 = torch.randn((1,L,L,L),dtype=torch.float)
        v2 = amp * v1
        v1.requires_grad = True
        v2.requires_grad = True
        cost_val = CovarCost.apply(v1,sim,0,sim.images[:],1e-4)
        cost_val.backward()
        cost_val = CovarCost.apply(v2,sim,0,sim.images[:] * amp,1e-4)
        cost_val.backward()
        
        
        self.assertFalse(bool(torch.any(torch.abs(torch.div(v1.grad,v2.grad) * (amp ** 3) - 1) > 1e-3)))
        
        
    @unittest.skip("not needed")    
    def testResolutionNorm(self):
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


