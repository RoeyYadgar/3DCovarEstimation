from utils import *
from covar_estimation import *
from covar_sgd import Covar

from aspire.volume import Volume,LegacyVolume
from aspire.utils import Rotation
from aspire.source import Simulation
from aspire.operators import RadialCTFFilter
from aspire.basis import FBBasis3D
from aspire.covariance import CovarianceEstimator
from aspire.noise import WhiteNoiseEstimator
from aspire.reconstruction import MeanEstimator
import scipy
import torch
import pickle
import os
import pandas as pd
from itertools import product

'''
def update_hyperparam_filenames(folder_name,default_params,filename_prefix = ''):
    df = pd.read_csv(os.path.join(folder_name,'results.csv'))
    for default_param_name,defualt_param_val in default_params.items():
        if(default_param_name not in df.columns):
            df[default_param_name] = defualt_param_val

    for i,df_row in df.iterrows():
        newfilename = filename_prefix + "_".join(f"{param_name}{param_vals[param_dict[param_name]]}" for param_name in param_dict if len(params[param_dict[param_name]]) > 1)
        #df.at[i,'filename'] = 
'''

def run_all_hyperparams(init_covar,folder_name,param_names,params,filename_prefix = ''):
    
    default_params = {'lr' : 5e-5,'momentum' : 0.9,'optimType' : 'SGD','reg' : 1e-5,'gammaLr' : 1,'gammaReg' : 1, 'batchSize' : 1, 'epochNum' : 10, 'orthogonalProjection' : False}
    for default_param_name,default_param_val in default_params.items(): #add default parameters if they aren't in param_names
        if(default_param_name not in param_names):
            param_names.append(default_param_name)
            params.append([default_param_val])

    param_dict = {param_name : index for index,param_name in enumerate(param_names)}
    param_combinations = list(product(*params))
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    for i,param_vals in enumerate(param_combinations):
        filename = filename_prefix + "_".join(f"{param_name}{param_vals[param_dict[param_name]]}" for param_name in param_dict)
        filepath = os.path.join(folder_name,filename)
        
        if(not os.path.isfile(filepath)):
            print(f'Running SGD on : {filepath}')
            open(filepath,'wb') #Create the 'decoy' file so other threads will not run the same training with the same parameters
            c = init_covar()
            try:
                c.train(batch_size = param_vals[param_dict['batchSize']],
                        epoch_num = param_vals[param_dict['epochNum']],
                        lr = param_vals[param_dict['lr']],
                        momentum = param_vals[param_dict['momentum']],
                        optim_type = param_vals[param_dict['optimType']],
                        reg = param_vals[param_dict['reg']],
                        gamma_lr = param_vals[param_dict['gammaLr']],
                        gamma_reg = param_vals[param_dict['gammaReg']],
                        orthogonal_projection= param_vals[param_dict['orthogonalProjection']]
                        )
                pickle.dump(c,open(filepath,'wb'))

                if(filename_prefix == ''):
                    df_row = pd.DataFrame([[filepath] + (list(param_vals))],
                                      columns = ['filename'] + (param_names))   
                else:
                    df_row = pd.DataFrame([[filepath] + (list(param_vals)) + [filename_prefix]],
                                      columns = ['filename'] + (param_names) + ['prefix'])
                appendCSV(df_row,
                          os.path.join(folder_name,'results.csv'))
            except Exception:
                print('Covar training got nan cost value, skipping training')
                os.remove(filepath) #Remove the 'decoy' file
    
    
def run_classic_alg(filepath,src,rank,basis = None):
    if(not os.path.isfile(filepath)):
        print(f'Running Classical Algo on : {filepath}')
        open(filepath,'wb') #Create the 'decoy' file so other threads will not run the same training with the same parameters

        if(basis == None):
            basis = FBBasis3D((src.L, src.L, src.L))

        noise_estimator = WhiteNoiseEstimator(src, batchSize=500)
        noise_variance = noise_estimator.estimate()
        mean_estimator = MeanEstimator(src)
        mean_est = mean_estimator.estimate()
        covar_estimator = CovarianceEstimator(src, basis, mean_kernel=mean_estimator.kernel)
        covar_est = covar_estimator.estimate(mean_est, noise_variance)
        eigs_est, lambdas_est = aspire.utils.eigs(covar_est, rank)
        covar_dict = {'covar' : covar_est , 'eigenvecs' : eigs_est, 'eigenvals' : lambdas_est}
        with open(filepath,'wb') as file:
           pickle.dump(covar_dict,file)



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
                    ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])

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
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])

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
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])
    
def rank2_resolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_L64_test'

    L = 64
    n = 2048
    r = 2
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [5e-5,1e-5,1e-6]
    momentum = [0.9]
    regularization = [1e-5,1e-4]
    gamma_lr = [1]
    gamma_reg = [1,0.8,0.5]

    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])
    
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
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])


def rank4_eigngap_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_eigengap_test'

    L = 15
    n = 2048
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)
    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))

    scaling_factor = [[1,1,1,1,1],[10,5,2,1,1]]
    filename_prefix = ['smallEigengap_' , 'largeEigengap_']

    for i in range(len(scaling_factor)):

        voxels *= np.array(scaling_factor[i],dtype = np.float32).reshape((r+1,1,1,1))        
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
        learning_rate = [1e-5 * 10]
        momentum = [0.9]
        regularization = [1e-5 , 1e-4 ,1e-3]
        gamma_lr = [1]
        gamma_reg = [0.8,0.5]
        orthogonal_projection = [True,False]

        covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
        run_all_hyperparams(covar_init,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg','orthogonalProjection'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg,orthogonal_projection],filename_prefix[i])

def rank4_lr_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_test'

    L = 15
    n = 2048
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [5e-4,1e-4,5e-5]
    momentum = [0.9]
    regularization = [1e-5,1e-6]
    gamma_lr = [1,0.8,0.5,0.3]
    gamma_reg = [0.8,0.5,0.3]

    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])
    
def rank4_optim_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_optim_test'

    L = 15
    n = 2048
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [1e-1,1e-2,1e-3,1e-4]
    momentum = [0.9]
    regularization = [1e-5,1e-6]
    gamma_lr = [1,0.8,0.5,0.3]
    gamma_reg = [0.8,0.5,0.3]
    optim = ['Adam']

    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg','optimType'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg,optim])

def rank2_cont_resolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_cont_L100_test'

    L = 100
    n = 2048
    r = 2
    voxels = Volume.from_vec(scipy.io.loadmat('data/vols.mat')['vols'].transpose())
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)

   
    learning_rate = [1e-4,5e-5,1e-5]
    momentum = [0.9]
    regularization = [1e-5,1e-6]
    gamma_lr = [1]
    gamma_reg = [1,0.8,0.5]

    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2))
    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])

def rank2_cont_ctf_resolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_cont_ctf_L100_test'

    L = 100
    n = 2048
    r = 2
    voxels = Volume.from_vec(scipy.io.loadmat('data/vols.mat')['vols'].transpose())
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])

   
    learning_rate = [1e-4,5e-5]
    momentum = [0.9]
    regularization = [1e-5]
    gamma_lr = [1 , 0.5 , 0.2]
    gamma_reg = [0.5]

    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2))
    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg])
    
def rank4_imsize_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_imsize_test'

    L = 15
    imnum = [2048 * (2 ** i) for i in range(4)]
    r = 4
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    

   
    learning_rate = [5e-4]
    momentum = [0.9]
    regularization = [1e-5]
    gamma_lr = [0.8]
    gamma_reg = [0.8]
    epochNum = [5]
    
    filename_prefix = ['numIm=' + str(n) + '_' for n in imnum]
    
    for i,n in enumerate(imnum): 
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0)
        covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
        run_all_hyperparams(covar_init,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg','epochNum'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg,epochNum],filename_prefix[i])
    
def rank2_cont_ctf_highresolution_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank2_cont_ctf_LHigh_test'

    from scipy.spatial.transform import Rotation as spRot
    vol_rots = np.single(spRot.from_euler('z', np.arange(0,100)/100*2*np.pi).as_matrix())
    
    resolutions = [256,512]
    
    for i,L in enumerate(resolutions):
    
        vol_highres = Volume.from_vec(scipy.io.loadmat('data/vols512.mat')['vols'].transpose()).downsample(L)
        high_res = vol_highres.resolution
        voxels = Volume(np.zeros((len(vol_rots),high_res,high_res,high_res)),dtype=np.single)
        for j in range(len(vol_rots)):
            voxels[j] = vol_highres.rotate(Rotation.from_matrix(vol_rots[j]))
        
        n = 2048
        r = 2
      
        voxels -= np.mean(voxels,axis=0)
    
        mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
        sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])
    
       
        learning_rate = [1e-4]
        momentum = [0.9]
        regularization = [1e-5]
        gamma_lr = [0.5]
        gamma_reg = [0.5]
    
        covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels,randomized_alg = True,max_eigennum = 2))
        run_all_hyperparams(covar_init,folder_name,
                            ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],filename_prefix = f'L={L}_')
        
def rank4_alg_cmp_test(folder_name = None):
    if(folder_name == None):
        folder_name = 'data/rank4_L15_alg_cmp'

    L = 15
    r = 4
    n = 2048
    voxels = LegacyVolume(L=L,C=r+1,dtype=np.float32,).generate() 
    voxels -= np.mean(voxels,axis=0)

    mean_voxel = Volume.from_vec(np.zeros((1,L**3),dtype=np.single))
    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)])

   
    learning_rate = [5e-4]
    momentum = [0.9]
    regularization = [1e-5]
    gamma_lr = [0.8]
    gamma_reg = [0.8]
    
    
    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],'SGD_')
    
    run_classic_alg(os.path.join(folder_name,'classical_covar.bin'),sim,r)

    #Sim with 'balanced' data
    num_reps = int(np.ceil(n/(r+1)))
    rotations = Rotation.generate_random_rotations(num_reps)
    rotations = np.repeat(rotations.angles, r+1,axis=0)[:n]
    states = np.tile(np.array([i+1 for i in range(r+1)]),num_reps)[:n]

    sim = Simulation(n = n , vols = voxels,amplitudes= 1,offsets = 0,unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],angles = rotations,states = states)
    covar_init = lambda : Covar(L,r,mean_voxel,sim,vectors= None,vectorsGD = volsCovarEigenvec(voxels))
    
    run_all_hyperparams(covar_init,folder_name,
                        ['lr','momentum','reg','gammaLr','gammaReg'],[learning_rate,momentum,regularization,gamma_lr,gamma_reg],'Balanced_SGD_')
    
    run_classic_alg(os.path.join(folder_name,'Balanced_classical_covar.bin'),sim,r)

if __name__ == "__main__":
    '''
    rank2_lr_params_test()
    rank2_gamma_params_test()
    rank2_ctf_test()
    rank2_resolution_test()
    rank4_resolution_test()
    rank4_eigngap_test()
    rank4_lr_test()
    rank4_optim_test()
    rank2_cont_resolution_test()
    rank2_cont_ctf_resolution_test()
    rank4_imsize_test()
    '''
    #rank2_cont_ctf_highresolution_test()
    rank4_alg_cmp_test()