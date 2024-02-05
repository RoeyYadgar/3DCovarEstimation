from utils import *
from covar_estimation import *
from covar_sgd import *

from aspire.volume import Volume
from aspire.utils import Rotation
from aspire.source import Simulation
import scipy


L = 15
n = 2048
voxels = Volume.from_vec(np.concatenate((generateBallVoxel([-0.6,0,0],0.5,15),
                                         generateBallVoxel([-0.6,0,0],0.5,15))))
#voxels = Volume.from_vec((generateBallVoxel([-0.6,0,0],0.5,15)))
 

mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
#voxels.save('tmp.mrc',overwrite=True)


sim = Simulation(n = n , vols = voxels,amplitudes= rademacherDist(n),offsets = 0)
projections = sim.images[:]
rots = Rotation.from_euler(sim.angles)
#projections[:5].show()

#covar_cost_gradient(voxels,sim,0,projections[:10])
vols_forward = vol_stack_forward(voxels,sim,0,10)
vol_backproject = im_stack_backward(vols_forward, sim, 0)


covar_cost(vols_forward,projections[:10])

'''
rots[0] = Rotation.from_matrix(np.eye(3))[0]
projected_vols = project_stack(voxels,rots)#.transpose()
projected_vols[0,0].show()
backprojected_vols = backproject_stack(projected_vols, rots.invert())

#backprojected_vols[0,0].save('data/test.mrc',overwrite=True)
scipy.io.savemat('data/test.mat',{'og_vol' : voxels.asnumpy()[0],'proj_vol' : projected_vols.asnumpy()[0],'bp_vols' : backprojected_vols.asnumpy()[0]})
print('computing_grad')
'''

'''
v = voxels.to_vec()
#v[0,(3*(L**2))]+=10
c = Covar(L,1,mean_voxel,vectors= None)



c.toVol().save('data/test_before.mrc',overwrite= True)
c.train(projections,rots,batch_size = 1,epoch_num = 1)
estimated_vol = c.toVol()
estimated_vol.save('data/test.mrc',overwrite= True)
#c.cost(projections,rots).backward()
#Volume(np.array(c.vectors.grad)).save('data/grad.mrc',overwrite = True)
#scipy.io.savemat('data/grad.mat',{'grad_vol':np.array(c.vectors.grad)[0]})
'''