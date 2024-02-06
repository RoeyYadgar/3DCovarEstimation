from utils import *
from covar_estimation import *
from covar_sgd import Covar

from aspire.volume import Volume
from aspire.utils import Rotation
from aspire.source import Simulation
import scipy
import torch
import pickle
import os

L = 15
n = 2048
voxels = Volume.from_vec(np.concatenate((generateBallVoxel([-0.6,0,0],0.5,15),
                                         -generateBallVoxel([-0.6,0,0],0.5,15))))
#voxels = Volume.from_vec((generateBallVoxel([-0.6,0,0],0.5,15)))
 

mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
#voxels.save('tmp.mrc',overwrite=True)


sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
#projections = sim.images[:]
#rots = Rotation.from_euler(sim.angles)
#projections[:5].show()


#v = voxels.to_vec()[0].reshape((1,L,L,L))
#v[0,0,0,0]+=10
'''
c = Covar(L,1,mean_voxel,sim,vectors= None,vectorsGD = voxels[0])


c.toVol().save('data/test_before.mrc',overwrite= True)
c.train(batch_size = 1,epoch_num = 1,lr= 100)
estimated_vol = c.toVol()
estimated_vol.save('data/test.mrc',overwrite= True)
'''

learning_rate = [1000, 10, 100 ,1000]
momentum = [0.8, 0.9, 0.95]


for lr in learning_rate:
    for mu in momentum:
        filepath = f'data/results/covar_est_lr{lr}_momentum{mu}.bins'
        if(not os.path.isfile(filepath)):
            c = Covar(L,1,mean_voxel,sim,vectors= None,vectorsGD = voxels[0])
            c.train(batch_size = 1,epoch_num = 3,lr= lr,momentum=mu)
            pickle.dump(c,open(filepath,'wb'))