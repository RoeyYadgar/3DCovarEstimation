from utils import *
from covar_estimation import *
from covar_sgd import Covar

from aspire.volume import Volume,LegacyVolume
from aspire.utils import Rotation
from aspire.source import Simulation
import scipy
import torch
import pickle
import os
import pandas as pd
aspire.config['logging']['console_level'] = 'WARNING'

L = 15
n = 2048
r = 2


voxels = LegacyVolume(
    L=L,
    C=r+1,
    dtype=np.float32,
).generate()

voxels -= np.mean(voxels,axis=0)

_,voxelsSTD,voxelsSpan = np.linalg.svd((voxels).asnumpy().reshape((r+1,-1)),full_matrices=False)
voxelsSpan = Volume.from_vec(voxelsSpan[:r,:] * voxelsSTD[:r,np.newaxis]) # r + 1 voxels with 0 mean can be spanned by r voxels
 

mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))


sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

c = Covar(L,2,mean_voxel,sim,vectors= None,vectorsGD = voxels[0])



lr_normalization_factor = np.mean(np.linalg.norm(sim.images[:],axis=(1,2))) ** 2

learning_rate = [5,0.5,50]
momentum = [0.9,0.95,0.8]
regularization = [1e-6,1e-5,1e-4, 1e-3 ,1e-2]

folder_name = 'data/results3'
covar_dataframe = []
for lr in learning_rate:
    for mu in momentum:
        for reg in regularization:
            filepath = os.path.join(folder_name,f'covar_est_lr{lr}_momentum{mu}_reg{reg}.bin')
            
            if(not os.path.isfile(filepath)):
                print(f'Running SGD on : {filepath}')
                open(filepath,'wb') #Create the 'decoy' file so other threads will not run the same training with the same parameters
                c = Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = voxelsSpan)
                try:
                    c.train(batch_size = 1,epoch_num = 10,lr= lr / lr_normalization_factor,momentum=mu , reg = reg)
                    pickle.dump(c,open(filepath,'wb'))
                    appendCSV(pd.DataFrame([[filepath,lr,mu,reg]],columns = ['filename','learning_rate','momentum','reg']),
                              os.path.join(folder_name,'results.csv'))
                except Exception:
                    print('Covar training got nan cost value, skipping training')
                    os.remove(filepath) #Remove the 'decoy' file
                
            
            

