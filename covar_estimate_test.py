from utils import *
from covar_estimation import *
from covar_sgd import Covar

from aspire.volume import Volume,LegacyVolume
from aspire.utils import Rotation
from aspire.source import Simulation
from aspire.operators import RadialCTFFilter
import scipy
import torch
import pickle
import os
import pandas as pd
from itertools import product


def run_all_hyperparams(init_covar,folder_name,param_names,*args):
    
    param_dict = {param_name : index for index,param_name in enumerate(param_names)}
    param_combinations = list(product(*args))
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    for i,param_vals in enumerate(param_combinations):
        filename = "_".join(f"{param_name}{param_vals[param_dict[param_name]]}" for param_name in param_dict)
        filepath = os.path.join(folder_name,filename)
        
        if(not os.path.isfile(filepath)):
            print(f'Running SGD on : {filepath}')
            open(filepath,'wb') #Create the 'decoy' file so other threads will not run the same training with the same parameters
            c = init_covar()
            try:
                c.train(batch_size = 1,epoch_num = 10,
                        lr = param_vals[param_dict['lr']],
                        momentum = param_vals[param_dict['momentum']],
                        reg = param_vals[param_dict['reg']],
                        gamma_lr = param_vals[param_dict['gammaLr']],
                        gamma_reg = param_vals[param_dict['gammaReg']],
                        )
                pickle.dump(c,open(filepath,'wb'))
                df_row = pd.DataFrame([[filepath] + (list(param_vals))],
                                      columns = ['filename'] + (param_names))
                appendCSV(df_row,
                          os.path.join(folder_name,'results.csv'))
            except Exception:
                print('Covar training got nan cost value, skipping training')
                os.remove(filepath) #Remove the 'decoy' file
    
    


def rank2_lr_params_test(folder_name = None):

    if(folder_name == None):
        folder_name = 'data/rank2_L15_lr_test'

    L = 15
    n = 2048
    r = 2
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate()
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

    learning_rate = [5e-4 ,1e-4 ,5e-5 ,1e-5 ,5e-6]
    momentum = [0.9]
    regularization = [1e-6,1e-5,1e-4]
    gamma_lr = [1]
    gamma_reg = [1]

    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))

    run_all_hyperparams(covar_init,folder_name,
                    ['lr','momentum','reg','gammaLr','gammaReg'],learning_rate,momentum,regularization,gamma_lr,gamma_reg)

def rank2_gamma_params_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_L15_gamma_test'

    L = 15
    n = 2048
    r = 2
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate()
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

    learning_rate = [1e-4,1e-5]
    momentum = [0.9]
    regularization = [1e-4,1e-5]
    gamma_lr = [1, 0.8, 0.5, 0.2, 0.1]
    gamma_reg = [1, 0.8, 0.5, 0.2, 0.1]

    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
                

    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],learning_rate,momentum,regularization,gamma_lr,gamma_reg)

def rank2_ctf_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_L15_ctf_test'

    L = 15
    n = 2048
    r = 2
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate()
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])

    learning_rate = [1e-4,1e-5]
    momentum = [0.9]
    regularization = [1e-4,1e-5,1e-3]
    gamma_lr = [1,0.8,0.5]
    gamma_reg = [1]

    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],learning_rate,momentum,regularization,gamma_lr,gamma_reg)
    
def rank4_resolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L64_test'

    L = 64
    n = 2048
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [1e-4,5e-5,1e-5,1e-6,5e-6]
    momentum = [0.9]
    regularization = [1e-5,1e-4]
    gamma_lr = [1]
    gamma_reg = [1,0.8,0.5]

    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],learning_rate,momentum,regularization,gamma_lr,gamma_reg)

if __name__ == "__main__":
    rank2_lr_params_test()
    rank2_gamma_params_test()
    rank2_ctf_test()
    rank4_resolution_test()
    