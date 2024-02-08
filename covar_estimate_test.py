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
import pandas as pd

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

learning_rate = [10]
momentum = [0.9]
regularization = [1e-5,1e-4, 1e-3 ,1e-2]


covar_dataframe = []
for lr in learning_rate:
    for mu in momentum:
        for reg in regularization:
            filepath = f'data/results/covar_est_lr{lr}_momentum{mu}_reg{reg}.bin'
            
            if(not os.path.isfile(filepath)):
                covar_dataframe.append([filepath,lr,mu,reg])
                c = Covar(L,1,mean_voxel,sim,vectors= None,vectorsGD = voxels[0])
                c.train(batch_size = 1,epoch_num = 10,lr= lr,momentum=mu , reg = reg)
                pickle.dump(c,open(filepath,'wb'))
                
            
covar_dataframe = pd.DataFrame(covar_dataframe,columns = ['filename','learning_rate','momentum','reg'])
#covar_dataframe.to_csv('data/results/results.csv',mode = 'a',header = False)
appendCSV(covar_dataframe, 'data/results/results.csv')
            
